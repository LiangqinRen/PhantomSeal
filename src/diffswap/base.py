from src.common_utils import cd, suppress_third_party_noise, use_project
from src.evaluate import Utility, Effectiveness, DistanceCloakSelector

import cv2
import dlib
import json
import numpy as np
import tempfile
import torch
import torch.nn.functional as F
import sys
from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path
from torch import Tensor


class Base:
    """
    Thin runtime wrapper around third_party/DiffSwap.
    """

    def __init__(self, logger, config):
        super().__init__()
        self.logger = logger
        self.config = config
        self.device = torch.device("cuda")
        self.utility = Utility(logger, config)
        self.effectiveness = Effectiveness(logger, config)
        self.cloak = DistanceCloakSelector(logger, config, self.effectiveness)

        third_party_config = self.config.third_party
        self.root_dir = Path(third_party_config.project_root)
        self.checkpoint_dir = Path(third_party_config.checkpoint_dir)
        self.checkpoint_path = Path(third_party_config.checkpoint_path)
        self.face_recognition_checkpoint_path = Path(
            third_party_config.face_recognition_checkpoint_path
        )
        self.shape_predictor_path = Path(third_party_config.shape_predictor_path)
        self.output_size = int(third_party_config.dataset.image_size)
        self.model_input_size = int(third_party_config.model.image_size)
        self.crop_size = int(third_party_config.model.crop_size)
        self.ddim_steps = int(third_party_config.origin.ddim_steps)
        self.ddim_eta = float(third_party_config.origin.ddim_eta)
        self.tgt_scale = float(third_party_config.origin.tgt_scale)
        self.detector_upsample = int(third_party_config.origin.detector_upsample)
        self.quiet_third_party = bool(third_party_config.origin.quiet_third_party)
        self.detector_resize_scales = tuple(
            float(scale) for scale in third_party_config.origin.detector_resize_scales
        )
        self.detection_failure_fallback = str(
            third_party_config.origin.detection_failure_fallback
        ).lower()

        self._check_required_files()
        self._ensure_project_checkpoint_compatibility()
        self._load_diffswap_modules()
        self._build_runtime_helpers()
        self._build_model()

    def _check_required_files(self) -> None:
        required_paths = [
            self.checkpoint_dir,
            self.checkpoint_path,
            self.face_recognition_checkpoint_path,
            self.shape_predictor_path,
            self.root_dir / "configs/diffswap/default-project.yaml",
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            missing_text = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "DiffSwap initialization is missing required files:\n" f"{missing_text}"
            )

    def _ensure_project_checkpoint_compatibility(self) -> None:
        """
        DiffSwap still hardcodes a few `checkpoints/...` paths internally.
        Keep the repo-level config centralized in `checkpoints/diffswap`, then
        create local compatibility symlinks inside `third_party/DiffSwap` so the
        untouched third-party code can still resolve its expected filenames.
        """

        project_checkpoint_dir = self.root_dir / "checkpoints"
        project_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        compatibility_map = {
            self.checkpoint_path: project_checkpoint_dir / self.checkpoint_path.name,
            self.face_recognition_checkpoint_path: (
                project_checkpoint_dir / self.face_recognition_checkpoint_path.name
            ),
            self.shape_predictor_path: project_checkpoint_dir
            / self.shape_predictor_path.name,
        }

        for source_path, target_path in compatibility_map.items():
            if target_path.exists():
                continue
            target_path.symlink_to(source_path.resolve())

    def _load_diffswap_modules(self) -> None:
        """
        DiffSwap ships its own top-level `src` package for ArcFace helpers.
        That clashes with this repository's `src` package, so we temporarily
        switch import context, import what we need, then restore the repo's
        original `src.*` modules.
        """

        original_src_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "src" or name.startswith("src.")
        }
        purge_prefixes = ("ldm", "utils", "data_preprocessing", "src")

        try:
            with suppress_third_party_noise(self.quiet_third_party):
                with use_project([self.root_dir], purge_prefixes=purge_prefixes), cd(
                    self.root_dir
                ):
                    from omegaconf import OmegaConf
                    from ldm.models.diffusion.ddim import DDIMSampler
                    from ldm.util import instantiate_from_config
                    from data_preprocessing.align.align_trans import (
                        get_reference_facial_points,
                        warp_and_crop_face,
                    )
                    from pipeline import crop_ffhq, get_lmk_256, get_lmk_ori
                    from utils.portrait import Portrait

                    self._OmegaConf = OmegaConf
                    self._DDIMSampler = DDIMSampler
                    self._instantiate_from_config = instantiate_from_config
                    self._get_reference_facial_points = get_reference_facial_points
                    self._warp_and_crop_face = warp_and_crop_face
                    self._crop_ffhq = crop_ffhq
                    self._get_lmk_256 = get_lmk_256
                    self._get_lmk_ori = get_lmk_ori
                    self._Portrait = Portrait
        finally:
            for name in list(sys.modules.keys()):
                if name == "src" or name.startswith("src."):
                    sys.modules.pop(name, None)
            sys.modules.update(original_src_modules)

    def _build_runtime_helpers(self) -> None:
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(str(self.shape_predictor_path))
        self.five_point_detector = MTCNN(
            image_size=160,
            device=self.device,
            selection_method="largest",
            keep_all=True,
            post_process=False,
        )
        self.reference_5pts = self._get_reference_facial_points(
            default_square=True
        ).astype(np.float32)

    def _build_model(self) -> None:
        config_path = self.root_dir / "configs/diffswap/default-project.yaml"
        original_src_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "src" or name.startswith("src.")
        }

        try:
            with suppress_third_party_noise(self.quiet_third_party):
                with use_project(
                    [self.root_dir],
                    purge_prefixes=("ldm", "utils", "data_preprocessing", "src"),
                ), cd(self.root_dir):
                    diff_config = self._OmegaConf.load(config_path)
                    self.model = self._instantiate_from_config(diff_config.model)
                    self.model.init_from_ckpt(str(self.checkpoint_path))
                    self.model = self.model.to(self.device).eval()
                    self.model.cond_stage_model.affine_crop = True
                    self.model.cond_stage_model.swap = True
                    self.ddim_sampler = self._DDIMSampler(
                        self.model,
                        tgt_scale=self.tgt_scale,
                    )
        finally:
            for name in list(sys.modules.keys()):
                if name == "src" or name.startswith("src."):
                    sys.modules.pop(name, None)
            sys.modules.update(original_src_modules)

    @staticmethod
    def _largest_face(rects) -> dlib.rectangle:
        return max(rects, key=lambda rect: rect.width() * rect.height())

    def _predict_landmarks(self, image_rgb_uint8: np.ndarray) -> np.ndarray | None:
        for scale in self.detector_resize_scales:
            if abs(scale - 1.0) < 1e-6:
                resized = image_rgb_uint8
            else:
                resized = cv2.resize(
                    image_rgb_uint8,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_LINEAR,
                )

            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector(gray, self.detector_upsample)
            if len(faces) == 0:
                continue

            face = self._largest_face(faces)
            shape = self.landmark_predictor(resized, face)
            points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
            points = points.astype(np.float32) / scale
            return points

        return None

    @staticmethod
    def _pick_largest_box_index(boxes: np.ndarray) -> int:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return int(np.argmax(areas))

    def _predict_five_points(self, image_rgb_uint8: np.ndarray) -> np.ndarray | None:
        image = Image.fromarray(image_rgb_uint8)
        boxes, _, landmarks = self.five_point_detector.detect(image, landmarks=True)
        if landmarks is None or len(landmarks) == 0:
            return None
        if boxes is not None and len(boxes) > 1:
            face_idx = self._pick_largest_box_index(boxes)
            return landmarks[face_idx].astype(np.float32)
        return landmarks[0].astype(np.float32)

    def _transform_to_theta(self, tfm: np.ndarray) -> np.ndarray:
        h1 = w1 = self.model_input_size
        h2 = w2 = self.crop_size
        a = np.array(
            [[2 / (w1 - 1), 0, -1], [0, 2 / (h1 - 1), -1], [0, 0, 1]],
            dtype=np.float32,
        )
        b = np.linalg.inv(
            np.array(
                [[2 / (w2 - 1), 0, -1], [0, 2 / (h2 - 1), -1], [0, 0, 1]],
                dtype=np.float32,
            )
        )
        c = np.array([[0, 0, 1]], dtype=np.float32)
        tfm_h = np.concatenate([tfm, c], axis=0)
        theta = a @ np.linalg.inv(tfm_h) @ b
        return theta[:2].astype(np.float32)

    def _build_affine_theta(self, facial_5pts: np.ndarray) -> np.ndarray:
        tfm = self._warp_and_crop_face(
            None,
            facial_5pts,
            self.reference_5pts,
            crop_size=(self.crop_size, self.crop_size),
            return_tfm=True,
        )
        return self._transform_to_theta(tfm)

    def _polygon_mask(
        self,
        landmarks: np.ndarray,
        indices: list[int],
        image_size: int,
    ) -> np.ndarray:
        polygon = np.round(landmarks[indices]).astype(np.int32)
        hull = cv2.convexHull(polygon)
        mask = np.zeros((image_size, image_size), dtype=np.float32)
        cv2.fillConvexPoly(mask, hull, 1.0)
        return mask

    def _build_target_masks(
        self, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        image_size = self.model_input_size
        organ_indices = [
            list(range(36, 42)),
            list(range(42, 48)),
            list(range(27, 36)),
            list(range(48, 68)),
        ]
        mask_organ = np.stack(
            [
                self._polygon_mask(landmarks, indices, image_size)
                for indices in organ_indices
            ]
        ).astype(np.float32)
        face_mask = self._polygon_mask(landmarks, list(range(68)), image_size)
        return mask_organ, face_mask

    def _prepare_image_batch(self, imgs: Tensor) -> Tensor:
        imgs = imgs.detach().to(self.device, dtype=torch.float32)
        imgs = F.interpolate(
            imgs,
            size=(self.model_input_size, self.model_input_size),
            mode="bilinear",
            align_corners=False,
        )
        return imgs

    def _get_detection_fallback(self, source_img: Tensor, target_img: Tensor) -> Tensor:
        return torch.zeros_like(target_img)

    def _build_condition_batch(
        self,
        source_imgs: Tensor,
        target_imgs: Tensor,
    ) -> tuple[dict[str, Tensor] | None, list[int]]:
        batch_size = source_imgs.shape[0]
        source_hwc = source_imgs.permute(0, 2, 3, 1).contiguous()
        target_hwc = target_imgs.permute(0, 2, 3, 1).contiguous()

        valid_indices: list[int] = []
        valid_source_hwc = []
        valid_target_hwc = []
        target_landmarks = []
        target_affines = []
        target_masks = []
        target_organ_masks = []
        source_affines = []
        source_organ_masks = []

        for index in range(batch_size):
            src_np = (
                source_imgs[index]
                .permute(1, 2, 0)
                .mul(255)
                .round()
                .clamp(0, 255)
                .byte()
                .cpu()
                .numpy()
            )
            tgt_np = (
                target_imgs[index]
                .permute(1, 2, 0)
                .mul(255)
                .round()
                .clamp(0, 255)
                .byte()
                .cpu()
                .numpy()
            )

            src_landmarks = self._predict_landmarks(src_np)
            tgt_landmarks = self._predict_landmarks(tgt_np)
            src_five_points = self._predict_five_points(src_np)
            tgt_five_points = self._predict_five_points(tgt_np)
            if (
                src_landmarks is None
                or tgt_landmarks is None
                or src_five_points is None
                or tgt_five_points is None
            ):
                self.logger.warning(
                    "DiffSwap face detection failed for batch item %s; "
                    "using '%s' fallback for that sample.",
                    index,
                    self.detection_failure_fallback,
                )
                continue

            valid_indices.append(index)
            valid_source_hwc.append(source_hwc[index])
            valid_target_hwc.append(target_hwc[index])

            src_mask_organ, _ = self._build_target_masks(src_landmarks)
            tgt_mask_organ, tgt_mask = self._build_target_masks(tgt_landmarks)

            target_landmarks.append(
                torch.from_numpy(
                    (tgt_landmarks / float(self.model_input_size)).astype(np.float32)
                )
            )
            target_affines.append(
                torch.from_numpy(self._build_affine_theta(tgt_five_points))
            )
            target_masks.append(torch.from_numpy(tgt_mask))
            target_organ_masks.append(torch.from_numpy(tgt_mask_organ))
            source_affines.append(
                torch.from_numpy(self._build_affine_theta(src_five_points))
            )
            source_organ_masks.append(torch.from_numpy(src_mask_organ))

        if not valid_indices:
            return None, valid_indices

        batch = {
            "image": torch.stack(valid_target_hwc, dim=0) * 2 - 1,
            "image_src": torch.stack(valid_source_hwc, dim=0) * 2 - 1,
            "landmark": torch.stack(target_landmarks, dim=0).to(self.device),
            "affine_theta": torch.stack(target_affines, dim=0).to(self.device),
            "affine_theta_src": torch.stack(source_affines, dim=0).to(self.device),
            "mask": torch.stack(target_masks, dim=0).to(self.device),
            "mask_organ": torch.stack(target_organ_masks, dim=0).to(self.device),
            "mask_organ_src": torch.stack(source_organ_masks, dim=0).to(self.device),
        }
        return batch, valid_indices

    def _encode_first_stage(self, image_hwc: Tensor) -> Tensor:
        image_chw = image_hwc.permute(0, 3, 1, 2).contiguous().float()
        encoder_posterior = self.model.encode_first_stage(image_chw)
        return self.model.get_first_stage_encoding(encoder_posterior).detach()

    @staticmethod
    def _tensor_to_uint8_image(img: Tensor) -> np.ndarray:
        image = (
            img.detach()
            .cpu()
            .permute(1, 2, 0)
            .mul(255)
            .round()
            .clamp(0, 255)
            .byte()
            .numpy()
        )
        return image

    def _build_original_portrait_batch(self, portrait_root: Path) -> dict[str, Tensor]:
        dataset = self._Portrait(str(portrait_root))
        sample = dataset[0]
        batch: dict[str, Tensor | list[str]] = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
            elif isinstance(value, torch.Tensor):
                tensor = value
            else:
                batch[key] = [value]
                continue
            batch[key] = tensor.unsqueeze(0).to(self.device)
        return batch  # type: ignore[return-value]

    def _write_original_affine_thetas(self, portrait_root: Path) -> None:
        affine_theta_all: dict[str, dict[str, list[list[float]]]] = {
            "source": {},
            "target": {},
        }
        for image_type in ("source", "target"):
            align_dir = portrait_root / "align" / image_type
            if not align_dir.exists():
                continue
            for image_path in sorted(align_dir.iterdir()):
                if not image_path.is_file():
                    continue
                image = np.array(Image.open(image_path).convert("RGB"))
                facial_5pts = self._predict_five_points(image)
                if facial_5pts is None:
                    continue
                theta = self._build_affine_theta(facial_5pts)
                affine_theta_all[image_type][image_path.name] = theta.tolist()

        with open(portrait_root / "affine_theta.json", "w", encoding="utf-8") as f:
            json.dump(affine_theta_all, f, indent=4)

    @torch.no_grad()
    def swap_face_original_pipeline(
        self, source_imgs: Tensor, target_imgs: Tensor
    ) -> Tensor:
        if source_imgs.ndim != 4 or target_imgs.ndim != 4:
            raise ValueError("DiffSwap expects 4D tensors shaped [B, 3, H, W].")
        if source_imgs.shape != target_imgs.shape:
            raise ValueError("Source and target tensors must share the same shape.")
        if source_imgs.shape[1] != 3:
            raise ValueError("DiffSwap expects RGB tensors with shape [B, 3, H, W].")
        if source_imgs.min() < -1e-5 or source_imgs.max() > 1 + 1e-5:
            raise ValueError("source_imgs must be normalized to [0, 1].")
        if target_imgs.min() < -1e-5 or target_imgs.max() > 1 + 1e-5:
            raise ValueError("target_imgs must be normalized to [0, 1].")

        results = []
        for source_img, target_img in zip(source_imgs, target_imgs):
            with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
                portrait_root = Path(tmp_dir) / "portrait"
                source_dir = portrait_root / "source"
                target_dir = portrait_root / "target"
                source_dir.mkdir(parents=True, exist_ok=True)
                target_dir.mkdir(parents=True, exist_ok=True)

                Image.fromarray(self._tensor_to_uint8_image(source_img)).save(
                    source_dir / "0000.png"
                )
                Image.fromarray(self._tensor_to_uint8_image(target_img)).save(
                    target_dir / "0000.png"
                )

                with suppress_third_party_noise(self.quiet_third_party):
                    with use_project(
                        [self.root_dir],
                        purge_prefixes=("ldm", "utils", "data_preprocessing", "src"),
                    ), cd(self.root_dir):
                        self._get_lmk_ori(
                            data_path=str(portrait_root),
                            save_path=str(portrait_root / "landmark"),
                        )
                        self._crop_ffhq(
                            data_path=str(portrait_root),
                            save_path=str(portrait_root / "align"),
                            affine_path=str(portrait_root / "affine_theta.json"),
                            landmark_path=str(
                                portrait_root / "landmark" / "landmark_ori.pkl"
                            ),
                            output_size=self.model_input_size,
                        )
                        self._get_lmk_256(
                            data_path=str(portrait_root / "align"),
                            save_path=str(portrait_root / "landmark"),
                            error_path=str(portrait_root / "error_img.json"),
                        )
                        self._write_original_affine_thetas(portrait_root)

                batch = self._build_original_portrait_batch(portrait_root)
                z, conditioning, _, _, _ = self.model.get_input(
                    batch,
                    self.model.first_stage_key,
                    return_first_stage_outputs=True,
                    force_c_encode=True,
                    return_original_cond=True,
                    swap=True,
                )
                latent_h, latent_w = z.shape[2], z.shape[3]
                preserve_background_mask = (1 - batch["mask"].float())[:, None]
                preserve_background_mask = F.interpolate(
                    preserve_background_mask,
                    size=(latent_h, latent_w),
                    mode="nearest",
                )
                preserve_background_mask[preserve_background_mask > 0] = 1
                preserve_background_mask[preserve_background_mask <= 0] = 0

                latent_shape = (
                    self.model.channels,
                    self.model.image_size,
                    self.model.image_size,
                )
                samples, _ = self.ddim_sampler.sample(
                    self.ddim_steps,
                    z.shape[0],
                    latent_shape,
                    conditioning,
                    eta=self.ddim_eta,
                    mask=preserve_background_mask,
                    x0=z,
                    verbose=False,
                )
                decoded = self.model.decode_first_stage(samples.to(self.device))
                decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
                results.append(decoded[0].detach().cpu())

        return torch.stack(results, dim=0).to(self.device)

    @torch.no_grad()
    def swap_face(self, source_imgs: Tensor, target_imgs: Tensor) -> Tensor:
        """
        Swap the identity of `source_imgs` onto `target_imgs`.

        Args:
            source_imgs: `[B, 3, H, W]` float tensor in `[0, 1]`
            target_imgs: `[B, 3, H, W]` float tensor in `[0, 1]`

        Returns:
            `[B, 3, output_size, output_size]` float tensor in `[0, 1]`

        Notes:
        - DiffSwap internally expects images in `[-1, 1]`, but this wrapper
          keeps the repo-wide public interface in `[0, 1]`.
        - DiffSwap also needs landmarks, masks, and affine crops. Those are
          generated on the fly from the input tensors inside this method.
        """

        if source_imgs.ndim != 4 or target_imgs.ndim != 4:
            raise ValueError("DiffSwap expects 4D tensors shaped [B, 3, H, W].")
        if source_imgs.shape != target_imgs.shape:
            raise ValueError("Source and target tensors must share the same shape.")
        if source_imgs.shape[1] != 3:
            raise ValueError("DiffSwap expects RGB tensors with shape [B, 3, H, W].")

        if source_imgs.min() < -1e-5 or source_imgs.max() > 1 + 1e-5:
            raise ValueError("source_imgs must be normalized to [0, 1].")
        if target_imgs.min() < -1e-5 or target_imgs.max() > 1 + 1e-5:
            raise ValueError("target_imgs must be normalized to [0, 1].")

        source_imgs = self._prepare_image_batch(source_imgs)
        target_imgs = self._prepare_image_batch(target_imgs)

        fallback_results = torch.stack(
            [
                self._get_detection_fallback(source_imgs[idx], target_imgs[idx])
                for idx in range(source_imgs.shape[0])
            ],
            dim=0,
        )

        batch, valid_indices = self._build_condition_batch(source_imgs, target_imgs)
        if batch is None:
            return fallback_results

        with suppress_third_party_noise(self.quiet_third_party):
            z = self._encode_first_stage(batch["image"])
            batch["z"] = z
            batch["z_src"] = torch.roll(z, shifts=1, dims=0)

            conditioning = self.model.get_learned_conditioning(batch)
            latent_h, latent_w = z.shape[2], z.shape[3]
            preserve_background_mask = (1 - batch["mask"].float())[:, None]
            preserve_background_mask = F.interpolate(
                preserve_background_mask,
                size=(latent_h, latent_w),
                mode="nearest",
            )
            preserve_background_mask[preserve_background_mask > 0] = 1
            preserve_background_mask[preserve_background_mask <= 0] = 0

            latent_shape = (
                self.model.channels,
                self.model.image_size,
                self.model.image_size,
            )
            samples, _ = self.ddim_sampler.sample(
                self.ddim_steps,
                z.shape[0],
                latent_shape,
                conditioning,
                eta=self.ddim_eta,
                mask=preserve_background_mask,
                x0=z,
                verbose=False,
            )
            valid_results = self.model.decode_first_stage(samples.to(self.device))
            valid_results = ((valid_results + 1.0) / 2.0).clamp(0.0, 1.0)

        results = fallback_results.clone()
        for output_index, batch_index in enumerate(valid_indices):
            results[batch_index] = valid_results[output_index]

        if (
            results.shape[-1] != self.output_size
            or results.shape[-2] != self.output_size
        ):
            results = F.interpolate(
                results,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return results
