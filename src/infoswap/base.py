from src.utils import cd, use_project

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from pathlib import Path
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config

        self.device = "cuda:0"

        origin_config = config.third_party.origin
        root_dir = Path(self.config.third_party.project_root)
        with use_project([root_dir, root_dir / "preprocess"]), cd(root_dir):
            from modules.encoder128 import Backbone128
            from modules.aii_generator import AII512
            from modules.decoder512 import UnetDecoder512
            from modules.iib import IIB
            from preprocess.mtcnn import MTCNN

            self.mtcnn = MTCNN()

            state_dict = torch.load(
                origin_config.encoder_path, map_location=self.device
            )
            self.encoder = Backbone128(50, 0.6, "ir_se").eval().to(self.device)
            self.encoder.load_state_dict(state_dict, strict=True)

            kernel_dir = (
                Path(origin_config.smooth_kernel_dir)
                if origin_config.smooth
                else Path(origin_config.no_smooth_kernel_dir)
            )
            kernel_name = (
                origin_config.smooth_kernel_name
                if origin_config.smooth
                else origin_config.no_smooth_kernel_name
            )
            nameG = kernel_name.replace("*", "G")
            nameE = kernel_name.replace("*", "E")
            nameI = kernel_name.replace("*", "I")

            self.G = AII512().eval().to(self.device)
            self.G.load_state_dict(
                torch.load(kernel_dir / nameG, map_location=self.device),
                strict=True,
            )
            self.decoder = UnetDecoder512().eval().to(self.device)
            self.decoder.load_state_dict(
                torch.load(kernel_dir / nameE, map_location=self.device), strict=True
            )

            self.N = 10
            _ = self.encoder(
                torch.rand(1, 3, 128, 128).to(self.device), cache_feats=True
            )
            _readout_feats = self.encoder.features[: (self.N + 1)]
            in_c = sum(map(lambda f: f.shape[-3], _readout_feats))
            out_c_list = [_readout_feats[i].shape[-3] for i in range(self.N)]

            self.iib = IIB(in_c, out_c_list, self.device, smooth=True, kernel_size=1)
            self.iib = self.iib.eval()
            self.iib.load_state_dict(
                torch.load(kernel_dir / nameI, map_location=self.device),
                strict=origin_config.smooth,
            )

            self.param_dict = []
            for i in range(self.N + 1):
                state = torch.load(
                    Path(origin_config.module_weights_dir) / f"readout_layer{i}.pth",
                    map_location=self.device,
                )
                n_samples = state["n_samples"].float()
                std = torch.sqrt(state["s"] / (n_samples - 1)).to(self.device)
                neuron_nonzero = state["neuron_nonzero"].float()
                active_neurons = (neuron_nonzero / n_samples) > 0.01
                self.param_dict.append(
                    [state["m"].to(self.device), std, active_neurons]
                )

    def swap_face(
        self,
        source_imgs: Tensor,
        target_imgs: Tensor,
    ) -> Tensor:
        aligned_source_imgs = self.align_batch_faces(source_imgs)
        aligned_target_imgs, original_targets, tfm_invs = (
            self.align_batch_targets_with_inverse(target_imgs)
        )

        X_id = self.encoder(
            F.interpolate(
                torch.cat((aligned_source_imgs, aligned_target_imgs), dim=0)[
                    :, :, 37:475, 37:475
                ],
                size=[128, 128],
                mode="bilinear",
                align_corners=True,
            ),
            cache_feats=True,
        )

        min_std = torch.tensor(0.01, device=self.device)
        readout_feats = [
            (self.encoder.features[i] - self.param_dict[i][0])
            / torch.max(self.param_dict[i][1], min_std)
            for i in range(self.N + 1)
        ]

        B = source_imgs.shape[0]
        X_id_restrict = torch.zeros_like(X_id).to(self.device)
        Xt_feats = []
        Xt_lambda = []

        for i in range(self.N):
            R = self.encoder.features[i]
            Z, lambda_, _ = getattr(self.iib, f"iba_{i}")(
                R,
                readout_feats,
                m_r=self.param_dict[i][0],
                std_r=self.param_dict[i][1],
                active_neurons=self.param_dict[i][2],
            )
            X_id_restrict += self.encoder.restrict_forward(Z, i)

            Rs, Rt = R[:B], R[B:]
            lambda_t = lambda_[B:]

            m_s = torch.mean(Rs, dim=0)
            std_s = torch.mean(Rs, dim=0)

            eps_s = torch.randn_like(Rt) * std_s + m_s
            feat_t = Rt * (1.0 - lambda_t) + lambda_t * eps_s

            Xt_feats.append(feat_t)
            Xt_lambda.append(lambda_t)

        X_id_restrict /= float(self.N)
        Xs_id = X_id_restrict[:B]
        Xt_feats[0] = aligned_target_imgs

        Xt_attr, Xt_attr_lamb = self.decoder(Xt_feats, lambs=Xt_lambda, use_lambda=True)
        Y = self.G(Xs_id, Xt_attr, Xt_attr_lamb)
        self.encoder.features = []

        blended = self.blend_back_to_targets(Y, original_targets, tfm_invs)

        # list[np.ndarray RGB uint8] -> tensor batch in [-1, 1]
        blended_tensors = []
        for img in blended:
            t = transforms.ToTensor()(img)  # [0,1]
            t = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(t)  # [-1,1]
            blended_tensors.append(t)

        return torch.stack(blended_tensors, dim=0).to(self.device)

    def align_batch_faces(
        self,
        imgs: Tensor,
        min_face_size: float = 64.0,
        thresholds: list[float] = [0.6, 0.7, 0.8],
        factor: float = 0.707,
        crop_size: tuple[int, int] = (512, 512),
        fallback_to_original: bool = True,
        retry_on_fail: bool = True,
        retry_min_face_size: float = 20.0,
        retry_thresholds: tuple[float, float, float] = (0.6, 0.6, 0.6),
        retry_decay: float = 0.8,
        max_retry_steps: int = 10,
    ) -> Tensor:
        """
        Align a batch of face images with MTCNN.

        Args:
            imgs: [B, 3, H, W], float tensor in [-1, 1]
            fallback_to_original:
                - False: raise error if any image has no detected face
                - True: keep the original image for that sample
            retry_on_fail:
                - Whether to retry with looser detection settings when initial align fails

        Returns:
            aligned_imgs: [B, 3, 512, 512], float tensor in [-1, 1]
        """
        if imgs.ndim != 4:
            raise ValueError(f"Expected imgs to be 4D [B, C, H, W], got {imgs.shape}")
        if imgs.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {imgs.shape}")

        device = imgs.device
        imgs_cpu = imgs.detach().cpu()

        to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        aligned_imgs = []

        for i in range(imgs_cpu.shape[0]):
            img = imgs_cpu[i]  # [3, H, W], [-1, 1]
            img_pil = to_pil_image(((img + 1) / 2).clamp(0, 1))

            faces = self.mtcnn.align_multi(
                img_pil,
                min_face_size=min_face_size,
                thresholds=thresholds,
                factor=factor,
                crop_size=crop_size,
            )

            # Retry with looser settings, similar to original target-side logic
            if (faces is None or len(faces) == 0) and retry_on_fail:
                mini = retry_min_face_size
                th1, th2, th3 = retry_thresholds

                for _ in range(max_retry_steps):
                    faces = self.mtcnn.align_multi(
                        img_pil,
                        min_face_size=mini,
                        thresholds=[th1, th2, th3],
                        factor=factor,
                        crop_size=crop_size,
                    )

                    if faces is not None and len(faces) > 0:
                        break

                    th1 *= retry_decay
                    th2 *= retry_decay
                    th3 *= retry_decay
                    mini *= retry_decay

            if faces is None or len(faces) == 0:
                if fallback_to_original:
                    if img.shape[-2:] != crop_size:
                        raise ValueError(
                            f"Sample {i} has no detected face, and original shape "
                            f"{img.shape[-2:]} != crop_size {crop_size}"
                        )
                    aligned_img = img
                else:
                    raise ValueError(f"No face detected for sample {i}")
            else:
                # Assume only one face, or use the first detected one
                aligned_img = to_tensor(faces[0])

            aligned_imgs.append(aligned_img)

        aligned_imgs = torch.stack(aligned_imgs, dim=0).to(device)
        return aligned_imgs

    def align_batch_targets_with_inverse(
        self,
        imgs: Tensor,
        min_face_size: float = 64.0,
        thresholds: list[float] = [0.6, 0.7, 0.7],
        factor: float = 0.707,
        crop_size: tuple[int, int] = (512, 512),
        fallback_to_original: bool = True,
        retry_on_fail: bool = True,
        retry_min_face_size: float = 20.0,
        retry_thresholds: tuple[float, float, float] = (0.6, 0.6, 0.6),
        retry_decay: float = 0.8,
        max_retry_steps: int = 10,
    ) -> tuple[Tensor, list[np.ndarray], list[np.ndarray | None]]:
        """
        Target-side align function.

        Args:
            imgs:
                [B, 3, H, W], float tensor in [-1, 1]

        Returns:
            aligned_imgs:
                [B, 3, 512, 512], float tensor in [-1, 1]
            original_imgs:
                list of HWC RGB uint8 target images
            tfm_invs:
                list of 2x3 inverse affine matrices, or None if fallback is used
        """
        if imgs.ndim != 4:
            raise ValueError(f"Expected imgs to be 4D [B, C, H, W], got {imgs.shape}")
        if imgs.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {imgs.shape}")

        device = imgs.device
        imgs_cpu = imgs.detach().cpu()

        to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        aligned_imgs: list[Tensor] = []
        original_imgs: list[np.ndarray] = []
        tfm_invs_out: list[np.ndarray | None] = []

        for i in range(imgs_cpu.shape[0]):
            img = imgs_cpu[i]  # [3, H, W], [-1, 1]

            img_01 = ((img + 1) / 2).clamp(0, 1)
            img_pil = to_pil_image(img_01)
            img_np = (img_01.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            original_imgs.append(img_np)

            out = self.mtcnn.align_multi(
                img_pil,
                min_face_size=min_face_size,
                thresholds=thresholds,
                factor=factor,
                crop_size=crop_size,
                reverse=True,
            )

            if out is None and retry_on_fail:
                mini = retry_min_face_size
                th1, th2, th3 = retry_thresholds
                for _ in range(max_retry_steps):
                    out = self.mtcnn.align_multi(
                        img_pil,
                        min_face_size=mini,
                        thresholds=[th1, th2, th3],
                        factor=factor,
                        crop_size=crop_size,
                        reverse=True,
                    )
                    if out is not None:
                        break
                    th1 *= retry_decay
                    th2 *= retry_decay
                    th3 *= retry_decay
                    mini *= retry_decay

            if out is None:
                if fallback_to_original:
                    if img.shape[-2:] != crop_size:
                        raise ValueError(
                            f"Sample {i} has no detected face, and original shape "
                            f"{img.shape[-2:]} != crop_size {crop_size}"
                        )
                    aligned_imgs.append(img)
                    tfm_invs_out.append(None)
                    continue
                raise ValueError(f"No face detected for target sample {i}")

            faces, tfm_invs, boxes = out
            if faces is None or len(faces) == 0:
                if fallback_to_original:
                    if img.shape[-2:] != crop_size:
                        raise ValueError(
                            f"Sample {i} has empty face result, and original shape "
                            f"{img.shape[-2:]} != crop_size {crop_size}"
                        )
                    aligned_imgs.append(img)
                    tfm_invs_out.append(None)
                    continue
                raise ValueError(f"Empty face result for target sample {i}")

            fi = 0
            if boxes is not None and len(boxes) > 1:
                ss = 0.0
                for j in range(len(boxes)):
                    box = boxes[j]
                    w = box[2] - box[0] + 1.0
                    h = box[3] - box[1] + 1.0
                    s = w * h
                    if s > ss:
                        ss = s
                        fi = j

            aligned_imgs.append(to_tensor(faces[fi]))
            tfm_invs_out.append(tfm_invs[fi])

        aligned_imgs = torch.stack(aligned_imgs, dim=0).to(device)
        return aligned_imgs, original_imgs, tfm_invs_out

    def blend_back_to_targets(
        self,
        Y: Tensor,
        original_targets: list[np.ndarray],
        tfm_invs: list[np.ndarray | None],
    ) -> list[np.ndarray]:
        """
        Blend aligned-space generated faces back to original target images.

        Args:
            Y:
                [B, 3, 512, 512], float tensor in [-1, 1]
            original_targets:
                list of HWC RGB uint8 images
            tfm_invs:
                list of 2x3 inverse affine matrices

        Returns:
            blended_results:
                list of HWC RGB uint8 images
        """
        if Y.ndim != 4 or Y.shape[1:] != (3, 512, 512):
            raise ValueError(f"Expected Y shape [B, 3, 512, 512], got {Y.shape}")

        if len(original_targets) != Y.shape[0] or len(tfm_invs) != Y.shape[0]:
            raise ValueError(
                "Batch size mismatch among Y, original_targets, and tfm_invs"
            )

        blended_results: list[np.ndarray] = []

        Y_cpu = Y.detach().cpu()

        for i in range(Y_cpu.shape[0]):
            y = Y_cpu[i]
            xt = original_targets[i]
            tfm_inv = tfm_invs[i]

            # 如果前面 fallback 了，没有 tfm_inv，就直接输出 aligned 结果或原图
            if tfm_inv is None:
                img_y = (y.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0
                img_y = np.clip(img_y, 0, 255).astype(np.uint8)
                blended_results.append(img_y)
                continue

            img_y = (y.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0
            img_y = np.clip(img_y, 0, 255).astype(np.uint8)

            H, W, _ = xt.shape

            frame = cv2.warpAffine(
                img_y.astype(np.float32),
                tfm_inv.astype(np.float32),
                dsize=(int(W), int(H)),
                borderValue=0,
            )

            # 原版的大 mask
            mask = np.zeros(img_y.shape, dtype=np.uint8)
            mask[37:475, 90:422, :] = 1
            mask = cv2.warpAffine(
                mask.astype(np.float32),
                tfm_inv.astype(np.float32),
                dsize=(int(W), int(H)),
                borderValue=0,
            )
            mask = (mask > 0).astype(np.uint8)

            try:
                src = np.array([255.0, 255.0, 1.0], dtype=np.float32).reshape(3, 1)
                x, y_center = np.matmul(tfm_inv, src)
                x = int(round(float(x.item())))
                y_center = int(round(float(y_center.item())))

                # 原版给 seamlessClone 用的更紧一些的 mask
                m = np.zeros(img_y.shape, dtype=np.uint8)
                m[40:472, 80:432, :] = 1
                m = cv2.warpAffine(
                    m.astype(np.float32),
                    tfm_inv.astype(np.float32),
                    dsize=(int(W), int(H)),
                    borderValue=0,
                )
                m = (m > 0).astype(np.uint8)

                res_poisson = cv2.seamlessClone(
                    frame.astype(np.uint8),
                    xt.astype(np.uint8),
                    m.astype(np.uint8) * 255,
                    p=(x, y_center),
                    flags=cv2.NORMAL_CLONE,
                )
                blended_results.append(res_poisson.astype(np.uint8))
            except Exception:
                # 简单 fallback：不用你原 repo 的 laplacian_blending，
                # 先做一个 alpha 混合，至少流程能跑通
                mask_f = mask.astype(np.float32)
                res = frame.astype(np.float32) * mask_f + xt.astype(np.float32) * (
                    1.0 - mask_f
                )
                blended_results.append(np.clip(res, 0, 255).astype(np.uint8))

        return blended_results
