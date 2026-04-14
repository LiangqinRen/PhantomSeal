from src.common_utils import cd, use_project
from src.evaluate import Utility, Effectiveness, DistanceCloakSelector
from third_party.FaceShifter.ModelC.face_modules.model import Backbone


import cv2
import torch
import warnings
import PIL.Image as Image
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torch import Tensor
from pathlib import Path


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config

        warnings.filterwarnings(
            "ignore", category=FutureWarning, module=".*matlab_cp2tform.*"
        )

        face_modules_dir = Path(self.config.third_party.project_root) / "face_modules"
        with use_project([face_modules_dir]):
            from third_party.FaceShifter.ModelC.face_modules.mtcnn import MTCNN
            from third_party.FaceShifter.ModelC.network.AEI_Net import AEI_Net
            from third_party.FaceShifter.ModelC.face_modules.mtcnn_pytorch.src.align_trans import (
                warp_and_crop_face,
            )

        self.utility = Utility(logger, config)
        self.effectiveness = Effectiveness(logger, config)
        self.cloak = DistanceCloakSelector(logger, config, self.effectiveness)

        self.device = torch.device("cuda")

        self.arcface = Backbone(50, 0.6, "ir_se").cuda()
        self.arcface.load_state_dict(
            torch.load(
                config.third_party.origin.model_path,
                weights_only=True,
            ),
            strict=False,
        )
        self.arcface = self.arcface.eval().cuda()

        with cd(Path(self.config.third_party.project_root)):
            self.detector = MTCNN()
        self._warp_and_crop_face = warp_and_crop_face

        self.G = AEI_Net(c_id=512)
        self.G.load_state_dict(
            torch.load(
                config.third_party.origin.G_path,
                weights_only=True,
            )
        )
        self.G = self.G.eval().cuda()

        self._normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self._blend_mask = self._build_blend_mask()

    def swap_face(self, src_img: Tensor, tgt_img: Tensor) -> Tensor:
        """
        Standard FaceShifter swap interface.

        Comparison against original ModelC/image_inference.py:
        - Original external input:
          source/target are image files read by cv2.imread as uint8 [0, 255]
          arrays with arbitrary HxW. Faces are then aligned to 256x256 before
          entering ArcFace / AEI-Net.
        - Current wrapper external input:
          src_img / tgt_img are tensors with shape [1, 3, H, W] in [-1, 1].
          H and W may vary, but batch size 1 is required.
        - Internal model input after alignment:
          source face -> [1, 3, 256, 256] in [-1, 1]
          ArcFace crop  -> [1, 3, 112, 112] in [-1, 1]
          target face -> [1, 3, 256, 256] in [-1, 1]
        - Output:
          returned tensor has shape [1, 3, H, W] at target resolution in [0, 1].

        Args:
            src_img:
                [1, 3, H, W] float tensor in [-1, 1]. Only batch size 1 is supported
                by the current FaceShifter wrapper.
            tgt_img:
                [1, 3, H, W] float tensor in [-1, 1]. Only batch size 1 is supported
                by the current FaceShifter wrapper.

        Returns:
            [1, 3, H, W] float tensor in [0, 1].
        """
        source_pil = Image.fromarray(self._to_ndarray(src_img))
        aligned_source = self._align_single_face(
            source_pil,
            crop_size=(256, 256),
            return_trans_inv=False,
            role="source",
        )
        if aligned_source is None:
            self.logger.warning(
                "FaceShifter source alignment failed after retries; "
                "falling back to a resized full-frame source crop."
            )
            aligned_source = source_pil.resize((256, 256), Image.Resampling.BILINEAR)
        # Match original ModelC preprocessing: aligned 256x256 RGB face -> tensor in [-1, 1].
        source_tensor = self._transform(aligned_source).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Match original ArcFace identity path: center crop 19:237 from the
            # aligned 256x256 face, then resize to 112x112.
            embeds = self.arcface(
                F.interpolate(
                    source_tensor[:, :, 19:237, 19:237],
                    (112, 112),
                    mode="bilinear",
                    align_corners=True,
                )
            )

        tgt_img_raw = self._to_ndarray(tgt_img).astype(np.float32) / 255.0
        aligned = self._align_single_face(
            Image.fromarray((tgt_img_raw * 255.0).astype(np.uint8)),
            crop_size=(256, 256),
            return_trans_inv=True,
            role="target",
        )
        if aligned is None:
            self.logger.warning(
                "FaceShifter target alignment failed after retries; "
                "falling back to full-frame generation."
            )
            tgt_pil = Image.fromarray((tgt_img_raw * 255.0).astype(np.uint8)).resize(
                (256, 256), Image.Resampling.BILINEAR
            )
            trans_inv = None
        else:
            tgt_pil, trans_inv = aligned
        # Match original AEI-Net target input: aligned 256x256 face in [-1, 1].
        tgt_img = self._transform(tgt_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            Yt, _ = self.G(tgt_img, embeds)
            Yt = Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0]) * 0.5 + 0.5
            if trans_inv is None:
                Yt_full = cv2.resize(
                    Yt,
                    (np.size(tgt_img_raw, 1), np.size(tgt_img_raw, 0)),
                    interpolation=cv2.INTER_LINEAR,
                )
                return torch.clamp(
                    transforms.ToTensor()(Yt_full).unsqueeze(0), min=0.0, max=1.0
                )
            Yt_trans_inv = cv2.warpAffine(
                Yt,
                trans_inv,
                (np.size(tgt_img_raw, 1), np.size(tgt_img_raw, 0)),
                borderValue=(0, 0, 0),
            )
            mask_ = cv2.warpAffine(
                self._blend_mask,
                trans_inv,
                (np.size(tgt_img_raw, 1), np.size(tgt_img_raw, 0)),
                borderValue=(0, 0, 0),
            )
            mask_ = np.expand_dims(mask_, 2)
            # Blend swapped face back into the original target-resolution image
            # using the same radial mask idea as original ModelC.
            Yt_trans_inv = mask_ * Yt_trans_inv + (1 - mask_) * tgt_img_raw

        return torch.clamp(
            transforms.ToTensor()(Yt_trans_inv).unsqueeze(0), min=0.0, max=1.0
        )

    def swapface(self, src_img: Tensor, tgt_img: Tensor) -> Tensor:
        return self.swap_face(src_img, tgt_img)

    def _to_ndarray(self, img: Tensor) -> np.ndarray:
        # Wrapper boundary conversion: tensor input is expected in [-1, 1],
        # while MTCNN / PIL utilities operate on uint8 image arrays in [0, 255].
        img = (img * 0.5 + 0.5) * 255
        img = img.squeeze(0)
        img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        img = img.astype(np.uint8)

        return img

    def _denormalize(self, img: Tensor) -> Tensor:
        return img * 0.5 + 0.5

    @staticmethod
    def _build_blend_mask() -> np.ndarray:
        coords = np.arange(256, dtype=np.float32)
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        dist = np.sqrt((yy - 128.0) ** 2 + (xx - 128.0) ** 2) / 128.0
        mask = 1.0 - np.minimum(dist, 1.0)
        return cv2.dilate(mask.astype(np.float32), None, iterations=20)

    def _align_single_face(
        self,
        img: Image.Image,
        crop_size: tuple[int, int] = (256, 256),
        return_trans_inv: bool = False,
        role: str = "face",
        min_face_size: float = 64.0,
        thresholds: tuple[float, float, float] = (0.6, 0.7, 0.8),
        retry_min_face_size: float = 20.0,
        retry_thresholds: tuple[float, float, float] = (0.6, 0.6, 0.6),
        retry_decay: float = 0.8,
        max_retry_steps: int = 10,
    ) -> Image.Image | tuple[Image.Image, np.ndarray] | None:
        boxes, landmarks = self._detect_faces_with_retry(
            img=img,
            role=role,
            min_face_size=min_face_size,
            thresholds=thresholds,
            retry_min_face_size=retry_min_face_size,
            retry_thresholds=retry_thresholds,
            retry_decay=retry_decay,
            max_retry_steps=max_retry_steps,
        )
        if landmarks is None or len(landmarks) == 0:
            return None

        face_idx = self._pick_largest_face_index(boxes)
        landmark = landmarks[face_idx]
        facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
        warped_face = self._warp_and_crop_face(
            np.array(img),
            facial5points,
            self.detector.refrence,
            crop_size=crop_size,
            return_trans_inv=return_trans_inv,
        )
        if return_trans_inv:
            face, trans_inv = warped_face
            return Image.fromarray(face), trans_inv
        return Image.fromarray(warped_face)

    def _detect_faces_with_retry(
        self,
        img: Image.Image,
        role: str,
        min_face_size: float,
        thresholds: tuple[float, float, float],
        retry_min_face_size: float,
        retry_thresholds: tuple[float, float, float],
        retry_decay: float,
        max_retry_steps: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        last_error: Exception | None = None
        attempts: list[tuple[float, tuple[float, float, float]]] = [
            (min_face_size, thresholds)
        ]

        current_min_face = retry_min_face_size
        current_thresholds = retry_thresholds
        for _ in range(max_retry_steps):
            attempts.append((current_min_face, current_thresholds))
            current_min_face *= retry_decay
            current_thresholds = tuple(x * retry_decay for x in current_thresholds)

        for attempt_idx, (attempt_min_face, attempt_thresholds) in enumerate(
            attempts, start=1
        ):
            try:
                boxes, landmarks = self.detector.detect_faces(
                    img,
                    min_face_size=attempt_min_face,
                    thresholds=list(attempt_thresholds),
                )
            except Exception as exc:
                last_error = exc
                boxes, landmarks = None, None

            if landmarks is not None and len(landmarks) > 0:
                return boxes, landmarks

        if last_error is not None:
            self.logger.warning(
                "FaceShifter %s alignment exhausted all retries with last error: %s",
                role,
                last_error,
            )
        else:
            self.logger.warning(
                "FaceShifter %s alignment exhausted all retries without detecting a face.",
                role,
            )
        return None, None

    @staticmethod
    def _pick_largest_face_index(boxes: np.ndarray | None) -> int:
        if boxes is None or len(boxes) == 0:
            return 0

        best_idx = 0
        best_area = -1.0
        for idx, box in enumerate(boxes):
            width = float(box[2] - box[0] + 1.0)
            height = float(box[3] - box[1] + 1.0)
            area = width * height
            if area > best_area:
                best_area = area
                best_idx = idx

        return best_idx
