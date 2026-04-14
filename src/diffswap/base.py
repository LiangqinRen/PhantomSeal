from src.common_utils import cd, suppress_third_party_noise, use_project
from src.evaluate import Effectiveness

import cv2
import dlib
import numpy as np
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from torch import Tensor


class Base:
    """
    Thin runtime wrapper around third_party/DiffSwap.

    Unified interface:
        swap_face(source_imgs, target_imgs) -> swapped_imgs

    Input contract:
    - `source_imgs` and `target_imgs` must be float tensors in `[0, 1]`
    - shape must be `[B, 3, H, W]`
    - each image should contain one reasonably clear frontal face

    Output contract:
    - returns a float tensor in `[0, 1]`
    - shape is `[B, 3, output_size, output_size]`

    Special initialization requirements:
    - `checkpoints/diffswap/diffswap.pth`
    - `checkpoints/diffswap/glint360k_r100.pth`
    - `checkpoints/diffswap/shape_predictor_68_face_landmarks.dat`

    The original DiffSwap test script relied on dataset-side json/pkl files for
    landmarks, affine transforms, and masks. This wrapper rebuilds those
    conditioning tensors directly from runtime tensors so the rest of the repo
    can use a normal tensor-based `swap_face(...)` API.
    """

    def __init__(self, logger, config):
        super().__init__()
        self.logger = logger
        self.config = config
        self.device = torch.device("cuda")

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

        self.effectiveness = Effectiveness(logger, config)

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

                    self._OmegaConf = OmegaConf
                    self._DDIMSampler = DDIMSampler
                    self._instantiate_from_config = instantiate_from_config
                    self._get_reference_facial_points = get_reference_facial_points
                    self._warp_and_crop_face = warp_and_crop_face
        finally:
            for name in list(sys.modules.keys()):
                if name == "src" or name.startswith("src."):
                    sys.modules.pop(name, None)
            sys.modules.update(original_src_modules)

    def _build_runtime_helpers(self) -> None:
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(str(self.shape_predictor_path))
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
    def _identity_theta(batch_size: int, device: torch.device) -> Tensor:
        theta = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        return theta.unsqueeze(0).repeat(batch_size, 1, 1)

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
    def _extract_five_points(landmarks: np.ndarray) -> np.ndarray:
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        nose = landmarks[30]
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        return np.stack([left_eye, right_eye, nose, mouth_left, mouth_right]).astype(
            np.float32
        )

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

    def _build_affine_theta(self, landmarks: np.ndarray) -> np.ndarray:
        facial_5pts = self._extract_five_points(landmarks)
        tfm = self._warp_and_crop_face(
            None,
            facial_5pts,
            self.reference_5pts,
            crop_size=(self.crop_size, self.crop_size),
            return_tfm=True,
        )
        return self._transform_to_theta(tfm)

    def _build_face_affine_theta_batch(self, imgs: Tensor) -> Tensor:
        imgs = self._prepare_image_batch(imgs)
        theta_list = []

        for index in range(imgs.shape[0]):
            img_np = (
                imgs[index]
                .permute(1, 2, 0)
                .mul(255)
                .round()
                .clamp(0, 255)
                .byte()
                .cpu()
                .numpy()
            )
            landmarks = self._predict_landmarks(img_np)
            if landmarks is None:
                self.logger.warning(
                    "DiffSwap face detection failed for identity crop %s; "
                    "falling back to the full-image crop.",
                    index,
                )
                theta_list.append(self._identity_theta(1, imgs.device)[0])
                continue

            theta_list.append(
                torch.from_numpy(self._build_affine_theta(landmarks)).to(imgs.device)
            )

        return torch.stack(theta_list, dim=0)

    def _crop_face_with_theta(self, imgs: Tensor, theta: Tensor) -> Tensor:
        grid = F.affine_grid(
            theta,
            size=(imgs.shape[0], 3, self.crop_size, self.crop_size),
            align_corners=False,
        )
        return F.grid_sample(
            imgs,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    @staticmethod
    def _free_gpu() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

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
            list(range(48, 68)),
            list(range(27, 36)),
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
        mode = self.detection_failure_fallback
        if mode == "target":
            return target_img
        if mode == "source":
            return source_img
        if mode == "zeros":
            return torch.zeros_like(target_img)
        if mode == "ones":
            return torch.ones_like(target_img)
        raise ValueError(
            "Unsupported detection_failure_fallback. "
            "Expected one of: target, source, zeros, ones."
        )

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
            if src_landmarks is None or tgt_landmarks is None:
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
                torch.from_numpy(self._build_affine_theta(tgt_landmarks))
            )
            target_masks.append(torch.from_numpy(tgt_mask))
            target_organ_masks.append(torch.from_numpy(tgt_mask_organ))
            source_affines.append(
                torch.from_numpy(self._build_affine_theta(src_landmarks))
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
            z_src = self._encode_first_stage(batch["image_src"])
            batch["z"] = z
            batch["z_src"] = z_src

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
