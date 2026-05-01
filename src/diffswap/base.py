from src.common_utils import cd, suppress_third_party_noise, use_project
from src.evaluate import Effectiveness

import cv2
import dlib
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from PIL import Image
from scipy.spatial import ConvexHull
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
        self.face_mask_dilate = int(third_party_config.origin.face_mask_dilate)
        self.repair_by_mask = bool(third_party_config.origin.repair_by_mask)
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
        purge_prefixes = (
            "ldm",
            "utils",
            "data_preprocessing",
            "src",
            "detector",
            "get_nets",
            "box_utils",
            "first_stage",
        )

        try:
            with suppress_third_party_noise(self.quiet_third_party):
                with use_project(
                    [self.root_dir, self.root_dir / "data_preprocessing/align"],
                    purge_prefixes=purge_prefixes,
                ), cd(self.root_dir / "data_preprocessing/align"):
                    import mtcnn
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
                    self._mtcnn_module = mtcnn
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
    def _largest_face(rects) -> dlib.rectangle:
        return max(rects, key=lambda rect: rect.width() * rect.height())

    def _get_detect(self, gray_image: np.ndarray, max_iter: int = 2):
        faces = []
        for upsample in range(max_iter + 1):
            faces = self.face_detector(gray_image, upsample)
            if len(faces) >= 1:
                break
        return faces

    @staticmethod
    def _shape_to_np(shape) -> np.ndarray:
        return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

    @staticmethod
    def _tensor_to_rgb_uint8(img: Tensor) -> np.ndarray:
        return (
            img.detach()
            .permute(1, 2, 0)
            .mul(255)
            .round()
            .clamp(0, 255)
            .byte()
            .cpu()
            .numpy()
        )

    def _detect_five_points_mtcnn(self, image_rgb_uint8: np.ndarray) -> np.ndarray | None:
        keys = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
        original_np_load = np.load

        def np_load_allow_pickle(*args, **kwargs):
            kwargs.setdefault("allow_pickle", True)
            return original_np_load(*args, **kwargs)

        try:
            np.load = np_load_allow_pickle
            with suppress_third_party_noise(self.quiet_third_party):
                if hasattr(self._mtcnn_module, "MTCNN"):
                    detector = self._mtcnn_module.MTCNN()
                    results = detector.detect_faces(image_rgb_uint8)
                elif hasattr(self._mtcnn_module, "detect_faces"):
                    boxes, landmarks = self._mtcnn_module.detect_faces(image_rgb_uint8)
                    if len(boxes) == 0 or len(landmarks) == 0:
                        results = []
                    else:
                        results = []
                        for box, landmark in zip(boxes, landmarks):
                            results.append(
                                {
                                    "box": [
                                        float(box[0]),
                                        float(box[1]),
                                        float(box[2] - box[0]),
                                        float(box[3] - box[1]),
                                    ],
                                    "keypoints": {
                                        "left_eye": [float(landmark[0]), float(landmark[5])],
                                        "right_eye": [float(landmark[1]), float(landmark[6])],
                                        "nose": [float(landmark[2]), float(landmark[7])],
                                        "mouth_left": [float(landmark[3]), float(landmark[8])],
                                        "mouth_right": [float(landmark[4]), float(landmark[9])],
                                    },
                                }
                            )
                else:
                    raise ImportError("Unsupported mtcnn package layout.")
        finally:
            np.load = original_np_load

        if len(results) == 0:
            return None

        def compute_area(item: dict) -> float:
            box = item.get("box", [0, 0, 0, 0])
            return float(box[-2] * box[-1])

        best = max(results, key=compute_area)
        keypoints = best.get("keypoints", {})
        if any(key not in keypoints for key in keys):
            return None
        return np.array([keypoints[key] for key in keys], dtype=np.float32)

    def _get_landmark_256_in_memory(self, image_rgb_uint8: np.ndarray) -> np.ndarray | None:
        faces = self._get_detect(cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2GRAY), 2)
        if len(faces) == 0:
            return None

        face = self._largest_face(faces)
        landmark = self.landmark_predictor(image_rgb_uint8, face)
        return self._shape_to_np(landmark).astype(np.float32)

    def _get_landmark_ori_in_memory(
        self, image_rgb_uint8: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        image = image_rgb_uint8
        while image.shape[0] > 2000 or image.shape[1] > 2000:
            image = cv2.resize(
                image,
                (0, 0),
                fx=0.5,
                fy=0.5,
                interpolation=cv2.INTER_CUBIC,
            )
        while image.shape[0] < 400 or image.shape[1] < 400:
            image = cv2.resize(
                image,
                (0, 0),
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_CUBIC,
            )

        faces = self._get_detect(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 2)
        if len(faces) == 0:
            return None

        face = self._largest_face(faces)
        landmark = self.landmark_predictor(image, face)
        return image, self._shape_to_np(landmark).astype(np.float32)

    def _crop_ffhq_in_memory(
        self,
        image_rgb_uint8: np.ndarray,
        landmark_ori: np.ndarray,
        output_size: int = 256,
        transform_size: int = 1024,
        enable_padding: bool = False,
        rotate_level: bool = True,
    ) -> np.ndarray:
        lm_eye_left = landmark_ori[36:42]
        lm_eye_right = landmark_ori[42:48]
        lm_mouth_outer = landmark_ori[48:60]

        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        if rotate_level:
            x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1
        else:
            x = np.array([1, 0], dtype=np.float64)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1

        image = Image.fromarray(image_rgb_uint8)
        quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
        qsize = np.hypot(*x) * 2

        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (
                int(np.rint(float(image.size[0]) / shrink)),
                int(np.rint(float(image.size[1]) / shrink)),
            )
            image = image.resize(rsize, Image.BICUBIC)
            quad /= shrink
            qsize /= shrink

        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        crop = (
            max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, image.size[0]),
            min(crop[3] + border, image.size[1]),
        )
        if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
            crop = tuple(map(round, crop))
            image = image.crop(crop)
            quad -= crop[0:2]

        pad = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        pad = (
            max(-pad[0] + border, 0),
            max(-pad[1] + border, 0),
            max(pad[2] - image.size[0] + border, 0),
            max(pad[3] - image.size[1] + border, 0),
        )
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            image_arr = np.pad(
                np.float32(image),
                ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                "reflect",
            )
            h, w, _ = image_arr.shape
            y_grid, x_grid, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(
                1.0
                - np.minimum(
                    np.float32(x_grid) / pad[0], np.float32(w - 1 - x_grid) / pad[2]
                ),
                1.0
                - np.minimum(
                    np.float32(y_grid) / pad[1], np.float32(h - 1 - y_grid) / pad[3]
                ),
            )
            blur = qsize * 0.02
            image_arr += (
                ndimage.gaussian_filter(image_arr, [blur, blur, 0]) - image_arr
            ) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            image_arr += (np.median(image_arr, axis=(0, 1)) - image_arr) * np.clip(
                mask, 0.0, 1.0
            )
            image = Image.fromarray(
                np.uint8(np.clip(np.rint(image_arr), 0, 255)), "RGB"
            )
            quad += pad[:2]

        quad = (quad + 0.5).flatten()
        affine = (
            -(quad[0] - quad[6]) / transform_size,
            -(quad[0] - quad[2]) / transform_size,
            quad[0],
            -(quad[1] - quad[7]) / transform_size,
            -(quad[1] - quad[3]) / transform_size,
            quad[1],
        )
        image = image.transform(
            (transform_size, transform_size), Image.AFFINE, affine, Image.BICUBIC
        )
        if output_size < transform_size:
            image = image.resize((output_size, output_size), Image.BICUBIC)

        return np.array(image).astype(np.uint8)

    def _extract_convex_hull_from_normalized_landmark(
        self, landmark_norm: np.ndarray, size: int = 256
    ) -> np.ndarray:
        landmark = landmark_norm * size
        hull = ConvexHull(landmark)
        image = np.zeros((size, size), dtype=np.float32)
        points = [landmark[hull.vertices, :1], landmark[hull.vertices, 1:]]
        points = np.concatenate(points, axis=-1).astype("int32")
        mask = cv2.fillPoly(image, pts=[points], color=(255, 255, 255))
        return mask > 0

    def _build_portrait_masks_in_memory(
        self, landmark_norm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        all_indices = np.arange(0, 68)
        landmark_indices = {
            "l_eye": all_indices[36:42].tolist(),
            "r_eye": all_indices[42:48].tolist(),
            "nose": all_indices[27:36].tolist(),
            "mouth": all_indices[48:68].tolist(),
        }
        mask_organ = []
        for _, indices in landmark_indices.items():
            mask_organ.append(
                self._extract_convex_hull_from_normalized_landmark(landmark_norm[indices])
            )
        face_mask = self._extract_convex_hull_from_normalized_landmark(landmark_norm)
        if self.face_mask_dilate > 0:
            kernel = np.ones(
                (self.face_mask_dilate, self.face_mask_dilate),
                dtype=np.uint8,
            )
            face_mask = cv2.dilate(face_mask.astype(np.uint8), kernel, iterations=1) > 0
        return np.stack(mask_organ).astype(np.float32), face_mask.astype(np.float32)

    def _build_portrait_item_in_memory(
        self, img: Tensor
    ) -> dict[str, Tensor | np.ndarray] | None:
        original_rgb_uint8 = self._tensor_to_rgb_uint8(img)
        landmark_ori_result = self._get_landmark_ori_in_memory(original_rgb_uint8)
        if landmark_ori_result is None:
            return None

        resized_original_rgb_uint8, landmark_ori = landmark_ori_result
        image_rgb_uint8 = self._crop_ffhq_in_memory(
            resized_original_rgb_uint8,
            landmark_ori,
            output_size=self.model_input_size,
        )

        landmark_256 = self._get_landmark_256_in_memory(image_rgb_uint8)
        if landmark_256 is None:
            return None

        five_points = self._detect_five_points_mtcnn(image_rgb_uint8)
        if five_points is None:
            five_points = self._extract_five_points(landmark_256)

        landmark_norm = (landmark_256 / float(self.model_input_size)).astype(np.float32)
        mask_organ, face_mask = self._build_portrait_masks_in_memory(landmark_norm)

        image_hwc = (
            torch.from_numpy(image_rgb_uint8)
            .to(self.device, dtype=torch.float32)
            / 127.5
            - 1.0
        )

        return {
            "align_image": image_rgb_uint8,
            "landmark_256": landmark_256,
            "image_hwc": image_hwc,
            "landmark_norm": torch.from_numpy(landmark_norm).to(
                self.device, dtype=torch.float32
            ),
            "affine_theta": torch.from_numpy(
                self._build_affine_theta_from_five_points(five_points)
            ).to(self.device, dtype=torch.float32),
            "mask": torch.from_numpy(face_mask).to(self.device, dtype=torch.float32),
            "mask_organ": torch.from_numpy(mask_organ).to(
                self.device, dtype=torch.float32
            ),
        }

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

    def _build_affine_theta_from_five_points(
        self, facial_5pts: np.ndarray
    ) -> np.ndarray:
        tfm = self._warp_and_crop_face(
            None,
            facial_5pts,
            self.reference_5pts,
            crop_size=(self.crop_size, self.crop_size),
            return_tfm=True,
        )
        return self._transform_to_theta(tfm)

    @staticmethod
    def _free_gpu() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @staticmethod
    def _im_reduce(img: np.ndarray) -> np.ndarray:
        filt = 1.0 / 20 * np.array([1, 5, 8, 5, 1])
        lowpass = ndimage.correlate1d(img, filt, 0)
        lowpass = ndimage.correlate1d(lowpass, filt, 1)
        return lowpass[::2, ::2, ...]

    @staticmethod
    def _im_expand(img: np.ndarray, template: np.ndarray) -> np.ndarray:
        expanded = np.zeros(template.shape, img.dtype)
        expanded[::2, ::2, ...] = img
        filt = 1.0 / 10 * np.array([1, 5, 8, 5, 1])
        lowpass = ndimage.correlate1d(expanded, filt, 0, mode="constant")
        lowpass = ndimage.correlate1d(lowpass, filt, 1, mode="constant")
        return lowpass

    def _gaussian_pyramid(self, image: np.ndarray, layers: int = 7) -> list[np.ndarray]:
        pyr = [image]
        temp = image
        for _ in range(layers):
            temp = self._im_reduce(temp)
            pyr.append(temp)
        return pyr

    def _laplacian_pyramid(self, gaussian_pyramid: list[np.ndarray]) -> list[np.ndarray]:
        pyr = []
        for i in range(len(gaussian_pyramid) - 1):
            g_k = gaussian_pyramid[i]
            g_k_plus_1 = gaussian_pyramid[i + 1]
            g_k_1_expand = self._im_expand(g_k_plus_1, g_k)
            pyr.append(g_k - g_k_1_expand)
        pyr.append(gaussian_pyramid[-1])
        return pyr

    def _laplacian_collapse(self, pyr: list[np.ndarray]) -> np.ndarray:
        partial = pyr[-1]
        for i in range(len(pyr) - 1):
            next_lowest = pyr[-2 - i]
            expanded = self._im_expand(partial, next_lowest)
            partial = expanded + next_lowest
        return partial

    @staticmethod
    def _laplacian_pyr_join(
        pyr1: list[np.ndarray], pyr2: list[np.ndarray], mask_gp: list[np.ndarray]
    ) -> list[np.ndarray]:
        pyr = []
        for i in range(len(pyr1)):
            mask = np.array([mask_gp[i], mask_gp[i], mask_gp[i]]).transpose(1, 2, 0)
            pyr.append(np.multiply(pyr1[i], mask) + np.multiply(pyr2[i], 1 - mask))
        return pyr

    def _repair_by_mask_batch(self, swap_results: Tensor) -> Tensor:
        if not hasattr(self, "last_pipeline_steps"):
            return swap_results

        repaired = []
        target_steps = self.last_pipeline_steps.get("target", [])
        for idx in range(swap_results.shape[0]):
            if idx >= len(target_steps):
                repaired.append(swap_results[idx])
                continue

            swap_img = (
                swap_results[idx]
                .detach()
                .permute(1, 2, 0)
                .mul(255.0)
                .clamp(0, 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            target_img = target_steps[idx]["align_image"].astype(np.uint8)
            mask = target_steps[idx]["mask"].detach().cpu().numpy().astype(np.uint8)

            im1 = np.int32(swap_img)
            im2 = np.int32(target_img)
            gp_1, gp_2 = [self._gaussian_pyramid(im) for im in [im1, im2]]
            mask_gp = [cv2.resize(mask, (gp.shape[1], gp.shape[0])) for gp in gp_1]
            lp_1, lp_2 = [self._laplacian_pyramid(gp) for gp in [gp_1, gp_2]]
            lp_join = self._laplacian_pyr_join(lp_1, lp_2, mask_gp)
            im_join = self._laplacian_collapse(lp_join)
            np.clip(im_join, 0, 255, out=im_join)
            repaired_img = (
                torch.from_numpy(np.uint8(im_join))
                .permute(2, 0, 1)
                .to(swap_results.device, dtype=torch.float32)
                / 255.0
            )
            repaired.append(repaired_img)

        return torch.stack(repaired, dim=0)

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

        valid_indices: list[int] = []
        valid_source_hwc = []
        valid_target_hwc = []
        target_landmarks = []
        target_affines = []
        target_masks = []
        target_organ_masks = []
        source_affines = []
        source_organ_masks = []
        debug_steps = {"target": []}

        for index in range(batch_size):
            src_item = self._build_portrait_item_in_memory(source_imgs[index])
            tgt_item = self._build_portrait_item_in_memory(target_imgs[index])
            if src_item is None or tgt_item is None:
                self.logger.warning(
                    "DiffSwap original in-memory preprocessing failed for batch item %s; "
                    "using '%s' fallback for that sample.",
                    index,
                    self.detection_failure_fallback,
                )
                continue

            valid_indices.append(index)
            valid_source_hwc.append(src_item["image_hwc"])
            valid_target_hwc.append(tgt_item["image_hwc"])
            target_landmarks.append(tgt_item["landmark_norm"])
            target_affines.append(tgt_item["affine_theta"])
            target_masks.append(tgt_item["mask"])
            target_organ_masks.append(tgt_item["mask_organ"])
            source_affines.append(src_item["affine_theta"])
            source_organ_masks.append(src_item["mask_organ"])
            debug_steps["target"].append(tgt_item)

        if not valid_indices:
            return None, valid_indices

        self.last_pipeline_steps = debug_steps
        batch = {
            "image": torch.stack(valid_target_hwc, dim=0),
            "image_src": torch.stack(valid_source_hwc, dim=0),
            "landmark": torch.stack(target_landmarks, dim=0),
            "affine_theta": torch.stack(target_affines, dim=0),
            "affine_theta_src": torch.stack(source_affines, dim=0),
            "mask": torch.stack(target_masks, dim=0),
            "mask_organ": torch.stack(target_organ_masks, dim=0),
            "mask_organ_src": torch.stack(source_organ_masks, dim=0),
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

        source_imgs = source_imgs.detach().to(self.device, dtype=torch.float32)
        target_imgs = target_imgs.detach().to(self.device, dtype=torch.float32)

        fallback_results = torch.stack(
            [
                self._get_detection_fallback(
                    self._prepare_image_batch(source_imgs[idx : idx + 1])[0],
                    self._prepare_image_batch(target_imgs[idx : idx + 1])[0],
                )
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
            if self.repair_by_mask:
                valid_results = self._repair_by_mask_batch(valid_results)

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
