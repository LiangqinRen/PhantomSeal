from src.common_utils import cd, use_project
from src.evaluate import Utility, Effectiveness, DistanceCloakSelector

import os
import copy
import cv2
import sys
import torch
import tempfile
import warnings
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

from pathlib import Path
from typing import Any, Sequence
from PIL import Image
from torch import Tensor
from skimage.transform import resize

warnings.filterwarnings("ignore")


class Base:
    def __init__(self, logger: Any, config: Any) -> None:
        super().__init__()
        self.logger = logger
        self.config = config

        self.device: str = "cuda:0"
        self.utility = Utility(logger, config)
        self.effectiveness = Effectiveness(logger, config)
        self.cloak = DistanceCloakSelector(logger, config, self.effectiveness)

        origin_config = config.third_party.origin
        root_dir = Path(self.config.third_party.project_root)

        with use_project(
            [root_dir, root_dir / "src"],
            purge_prefixes=("src",),
        ), cd(root_dir):
            from src.pretrained.face_vid2vid.driven_demo import (
                init_facevid2vid_pretrained_model,
                drive_source_demo,
            )
            from src.pretrained.gpen.gpen_demo import (
                init_gpen_pretrained_model,
                GPEN_demo,
            )
            from src.pretrained.face_parsing.face_parsing_demo import (
                init_faceParsing_pretrained_model,
                faceParsing_demo,
                vis_parsing_maps,
            )
            from src.utils.swap_face_mask import swap_head_mask_revisit_considerGlass
            from src.utils import torch_utils
            from src.utils.alignmengt import crop_faces, calc_alignment_coefficients
            from src.utils.morphology import dilation, erosion
            from src.utils.multi_band_blending import blending
            from src.options.swap_options import SwapFacePipelineOptions
            from src.models.networks import Net3
            from src.datasets.dataset import TO_TENSOR, NORMALIZE

            self.drive_source_demo = drive_source_demo
            self.GPEN_demo = GPEN_demo
            self.faceParsing_demo = faceParsing_demo
            self.vis_parsing_maps = vis_parsing_maps
            self.swap_head_mask_revisit_considerGlass = (
                swap_head_mask_revisit_considerGlass
            )
            self.torch_utils = torch_utils
            self.crop_faces = crop_faces
            self.calc_alignment_coefficients = calc_alignment_coefficients
            self.dilation = dilation
            self.erosion = erosion
            self.blending = blending
            self.SwapFacePipelineOptions = SwapFacePipelineOptions
            self.Net3 = Net3
            self.TO_TENSOR = TO_TENSOR
            self.NORMALIZE = NORMALIZE

            # --------------------------------------------------
            # Build opts
            # --------------------------------------------------
            # Reset argv so the third-party parser only reads its defaults.
            sys.argv = [sys.argv[0]]
            self.opts = SwapFacePipelineOptions().parse()
            self.opts.device = self.device

            # --------------------------------------------------
            # face_vid2vid
            # --------------------------------------------------
            (
                self.generator,
                self.kp_detector,
                self.he_estimator,
                self.estimate_jacobian,
            ) = init_facevid2vid_pretrained_model(
                origin_config.face_vid2vid_cfg_path,
                origin_config.face_vid2vid_ckpt_path,
            )

            # --------------------------------------------------
            # GPEN
            # --------------------------------------------------
            self.GPEN_model = init_gpen_pretrained_model(
                model_params={
                    "base_dir": origin_config.gpen_base_dir,
                    "in_size": 512,
                    "model": "GPEN-BFR-512",
                    "use_sr": True,
                    "sr_model": "realesrnet",
                    "sr_scale": 4,
                    "channel_multiplier": 2,
                    "narrow": 1,
                }
            )

            face_parser_ckpt = origin_config.face_parser_ckpt_path
            face_parser_config = ""

            self.faceParsing_model = init_faceParsing_pretrained_model(
                "default",
                face_parser_ckpt,
                face_parser_config,
            )

            # --------------------------------------------------
            # E4S model
            # --------------------------------------------------
            self.net = self.Net3(self.opts).to(self.device)
            save_dict = torch.load(
                origin_config.checkpoint_path, map_location=self.device
            )
            self.net.load_state_dict(
                self.torch_utils.remove_module_prefix(
                    save_dict["state_dict"], prefix="module."
                )
            )
            self.net.latent_avg = save_dict["latent_avg"].to(self.device)
            self.net.eval()

    # ==========================================================
    # public API
    # ==========================================================
    @torch.no_grad()
    def swap_face(
        self,
        source_imgs: Tensor,
        target_imgs: Tensor,
        target_masks: list[np.ndarray] | None = None,
        need_crop: bool = False,
        only_target_crop: bool = False,
        verbose: bool = False,
    ) -> Tensor:
        """
        Args:
            source_imgs: [B, 3, H, W], float tensor in [-1, 1]
            target_imgs: [B, 3, H, W], float tensor in [-1, 1]
            target_masks:
                optional list of 12-class segmentation maps, each [H, W] uint8 / int
            need_crop:
                whether to crop and align both source and target as original pipeline
            only_target_crop:
                whether only crop target and keep source resized to 1024x1024
            verbose:
                save debug results internally if needed

        Returns:
            swapped_imgs: [B, 3, H_out, W_out], float tensor in [-1, 1]

        Notes:
            The public E4S swap interface consumes tensors in [-1, 1], converts
            each image to PIL internally, runs the original E4S pipeline, and
            returns results normalized to [-1, 1].
        """
        if source_imgs.ndim != 4 or target_imgs.ndim != 4:
            raise ValueError(
                f"Expected 4D tensors, got {source_imgs.shape=} and {target_imgs.shape=}"
            )
        if source_imgs.shape[0] != target_imgs.shape[0]:
            raise ValueError(
                f"Batch size mismatch: {source_imgs.shape[0]} vs {target_imgs.shape[0]}"
            )
        if source_imgs.shape[1] != 3 or target_imgs.shape[1] != 3:
            raise ValueError("Expected 3-channel RGB tensors in [-1, 1]")

        B = source_imgs.shape[0]
        results: list[Tensor] = []

        if target_masks is None:
            normalized_target_masks: list[np.ndarray | None] = [None] * B
        elif len(target_masks) != B:
            raise ValueError(
                f"target_masks length mismatch: expected {B}, got {len(target_masks)}"
            )
        else:
            normalized_target_masks = target_masks

        for i in range(B):
            source_pil = self.tensor_to_pil(source_imgs[i])
            target_pil = self.tensor_to_pil(target_imgs[i])

            result_pil = self.swap_face_single(
                source_img=source_pil,
                target_img=target_pil,
                target_mask=normalized_target_masks[i],
                need_crop=need_crop,
                only_target_crop=only_target_crop,
                verbose=verbose,
            )
            results.append(self.pil_to_tensor_normalized(result_pil))

        # Keep the batch API simple by requiring consistent output sizes.
        return torch.stack(results, dim=0).to(self.device)

    # ==========================================================
    # single-pair pipeline
    # ==========================================================
    @torch.no_grad()
    def swap_face_single(
        self,
        source_img: Image.Image,
        target_img: Image.Image,
        target_mask: np.ndarray | None = None,
        need_crop: bool = False,
        only_target_crop: bool = False,
        verbose: bool = False,
    ) -> Image.Image:
        """
        Single-pair E4S pipeline.
        Keeps the original flow as much as possible.
        """
        source_img = source_img.convert("RGB")
        target_img = target_img.convert("RGB")

        # --------------------------------------------------
        # (1) crop & align
        # --------------------------------------------------
        if only_target_crop:
            crops, orig_images, _quads, inv_transforms = self.crop_and_align_face_pil(
                [target_img]
            )
            crops = [crop.convert("RGB") for crop in crops]
            T = crops[0]
            S = source_img.resize((1024, 1024), Image.Resampling.BILINEAR)

        elif need_crop:
            crops, orig_images, _quads, inv_transforms = self.crop_and_align_face_pil(
                [source_img, target_img]
            )
            crops = [crop.convert("RGB") for crop in crops]
            S, T = crops

        else:
            S = source_img.resize((1024, 1024), Image.Resampling.BILINEAR)
            T = target_img.resize((1024, 1024), Image.Resampling.BILINEAR)
            crops = [S, T]
            orig_images = None
            inv_transforms = None

        # --------------------------------------------------
        # (2) parsing target
        # --------------------------------------------------
        S_256 = resize(np.array(S) / 255.0, (256, 256))
        T_256 = resize(np.array(T) / 255.0, (256, 256))

        T_mask = (
            self.faceParsing_demo(
                self.faceParsing_model,
                T,
                convert_to_seg12=True,
                model_name=self.opts.faceParser_name,
            )
            if target_mask is None
            else target_mask
        )

        # --------------------------------------------------
        # (3) faceVid2Vid reenactment
        # --------------------------------------------------
        predictions = self.drive_source_demo(
            S_256,
            [T_256],
            self.generator,
            self.kp_detector,
            self.he_estimator,
            self.estimate_jacobian,
        )
        predictions = [(pred * 255).astype(np.uint8) for pred in predictions]

        # --------------------------------------------------
        # (4) GPEN refinement
        # --------------------------------------------------
        drivens = [
            self.GPEN_demo(pred[:, :, ::-1], self.GPEN_model, aligned=False)
            for pred in predictions
        ]
        D = Image.fromarray(drivens[0][:, :, ::-1]).convert("RGB")

        # --------------------------------------------------
        # (5) parsing D
        # --------------------------------------------------
        D_mask = self.faceParsing_demo(
            self.faceParsing_model,
            D,
            convert_to_seg12=True,
            model_name=self.opts.faceParser_name,
        )

        # --------------------------------------------------
        # (6) wrap tensors
        # --------------------------------------------------
        driven = transforms.Compose([self.TO_TENSOR, self.NORMALIZE])(D)
        driven = driven.to(self.device).float().unsqueeze(0)

        driven_mask = transforms.Compose([self.TO_TENSOR])(Image.fromarray(D_mask))
        driven_mask = (driven_mask * 255).long().to(self.device).unsqueeze(0)
        driven_onehot = self.torch_utils.labelMap2OneHot(
            driven_mask, num_cls=self.opts.num_seg_cls
        )

        target = transforms.Compose([self.TO_TENSOR, self.NORMALIZE])(T)
        target = target.to(self.device).float().unsqueeze(0)

        target_mask_tensor = transforms.Compose([self.TO_TENSOR])(
            Image.fromarray(T_mask)
        )
        target_mask_tensor = (
            (target_mask_tensor * 255).long().to(self.device).unsqueeze(0)
        )
        target_onehot = self.torch_utils.labelMap2OneHot(
            target_mask_tensor, num_cls=self.opts.num_seg_cls
        )

        # --------------------------------------------------
        # (7) extract style vectors
        # --------------------------------------------------
        driven_style_vector, _ = self.net.get_style_vectors(driven, driven_onehot)
        target_style_vector, _ = self.net.get_style_vectors(target, target_onehot)

        # --------------------------------------------------
        # (8) swap mask / style
        # --------------------------------------------------
        swapped_msk, hole_map = self.swap_head_mask_revisit_considerGlass(
            D_mask, T_mask
        )

        comp_indices = set(range(self.opts.num_seg_cls)) - {0, 4, 11, 10}
        swapped_style_vectors = self.swap_comp_style_vector(
            target_style_vector,
            driven_style_vector,
            list(comp_indices),
            belowFace_interpolation=False,
        )

        # --------------------------------------------------
        # (9) generate swapped face
        # --------------------------------------------------
        swapped_msk_img = Image.fromarray(swapped_msk).convert("L")
        swapped_msk_tensor = transforms.Compose([self.TO_TENSOR])(swapped_msk_img)
        swapped_msk_tensor = (
            (swapped_msk_tensor * 255).long().to(self.device).unsqueeze(0)
        )
        swapped_onehot = self.torch_utils.labelMap2OneHot(
            swapped_msk_tensor, num_cls=self.opts.num_seg_cls
        )

        swapped_style_codes = self.net.cal_style_codes(swapped_style_vectors)
        swapped_face, _, _ = self.net.gen_img(
            torch.zeros(1, 512, 32, 32, device=self.device),
            swapped_style_codes,
            swapped_onehot,
        )
        swapped_face_image = self.torch_utils.tensor2im(swapped_face[0])

        # --------------------------------------------------
        # (10) blend in aligned/cropped space
        # --------------------------------------------------
        swapped_and_pasted = self.blend_swapped_to_target(
            swapped_face_image=swapped_face_image,
            target_img=T,
            swapped_msk=swapped_msk_tensor,
            hole_map=hole_map,
            lap_bld=self.opts.lap_bld,
            outer_dilation=5,
        )

        # --------------------------------------------------
        # (11) paste back to original target if cropped
        # --------------------------------------------------
        if only_target_crop:
            assert orig_images is not None and inv_transforms is not None
            inv_trans_coeffs, orig_image = inv_transforms[0], orig_images[0]
            pasted_image = self.paste_back_to_original(
                swapped_and_pasted, orig_image, inv_trans_coeffs
            )
        elif need_crop:
            assert orig_images is not None and inv_transforms is not None
            inv_trans_coeffs, orig_image = inv_transforms[1], orig_images[1]
            pasted_image = self.paste_back_to_original(
                swapped_and_pasted, orig_image, inv_trans_coeffs
            )
        else:
            pasted_image = swapped_and_pasted

        return pasted_image.convert("RGB")

    # ==========================================================
    # helpers
    # ==========================================================
    def tensor_to_pil(self, img: Tensor) -> Image.Image:
        """
        img: [3,H,W] in [-1,1]
        """
        img = img.detach().cpu()
        img = ((img + 1) / 2).clamp(0, 1)
        img = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(img).convert("RGB")

    def pil_to_tensor_normalized(self, img: Image.Image) -> Tensor:
        """
        PIL RGB -> tensor in [-1,1]
        """
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )(img.convert("RGB"))

    def crop_and_align_face_pil(
        self, images: list[Image.Image]
    ) -> tuple[list[Image.Image], list[Image.Image], list[np.ndarray], list[Any]]:
        """
        Reuse original crop_faces pipeline as much as possible.
        """
        image_size = 1024
        scale = 1.0
        center_sigma = 0
        xy_sigma = 0
        use_fa = False

        with tempfile.TemporaryDirectory() as tmp_dir:
            target_files: list[tuple[str, str]] = []
            for i, img in enumerate(images):
                path = os.path.join(tmp_dir, f"{i}.png")
                img.save(path)
                target_files.append((str(i), path))

            crops, orig_images, quads = self.crop_faces(
                image_size,
                target_files,
                scale,
                center_sigma=center_sigma,
                xy_sigma=xy_sigma,
                use_fa=use_fa,
            )

            inv_transforms = [
                self.calc_alignment_coefficients(
                    quad + 0.5,
                    [
                        [0, 0],
                        [0, image_size],
                        [image_size, image_size],
                        [image_size, 0],
                    ],
                )
                for quad in quads
            ]

        return crops, orig_images, quads, inv_transforms

    def swap_comp_style_vector(
        self,
        style_vectors1: Tensor,
        style_vectors2: Tensor,
        comp_indices: Sequence[int],
        belowFace_interpolation: bool = False,
    ) -> Tensor:
        """
        Same as original:
        style_vectors1: target
        style_vectors2: source/driven
        """
        style_vectors = copy.deepcopy(style_vectors1)

        for comp_idx in comp_indices:
            style_vectors[:, comp_idx, :] = style_vectors2[:, comp_idx, :]

        # no ear region in source
        if torch.sum(style_vectors2[:, 7, :]) == 0:
            style_vectors[:, 7, :] = (
                style_vectors1[:, 7, :] + style_vectors2[:, 7, :]
            ) / 2

        # no teeth region in source
        if torch.sum(style_vectors2[:, 9, :]) == 0:
            style_vectors[:, 9, :] = style_vectors1[:, 9, :]

        # neck interpolation if enabled
        if belowFace_interpolation:
            style_vectors[:, 8, :] = (
                style_vectors1[:, 8, :] + style_vectors2[:, 8, :]
            ) / 2

        return style_vectors

    def create_masks(
        self,
        mask: Tensor,
        outer_dilation: int = 0,
        operation: str = "dilation",
    ) -> tuple[Tensor, Tensor, Tensor]:
        radius = outer_dilation
        temp = copy.deepcopy(mask)

        kernel = torch.ones(
            2 * radius + 1,
            2 * radius + 1,
            device=mask.device,
        )

        if operation == "dilation":
            full_mask = self.dilation(temp, kernel, engine="convolution")
            border_mask = full_mask - temp
        elif operation == "erosion":
            full_mask = self.erosion(temp, kernel, engine="convolution")
            border_mask = temp - full_mask
        elif operation == "expansion":
            full_mask = self.dilation(temp, kernel, engine="convolution")
            erosion_mask = self.erosion(temp, kernel, engine="convolution")
            border_mask = full_mask - erosion_mask
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        border_mask = border_mask.clip(0, 1)
        content_mask = mask

        return content_mask, border_mask, full_mask

    def logical_or_reduce(self, *tensors: Tensor) -> Tensor:
        return torch.stack(tensors, dim=0).any(dim=0)

    def logical_and_reduce(self, *tensors: Tensor) -> Tensor:
        return torch.stack(tensors, dim=0).all(dim=0)

    def smooth_face_boundry(
        self,
        image: Image.Image,
        dst_image: Image.Image,
        mask: Image.Image,
        radius: int = 0,
        sigma: float = 0.0,
    ) -> Image.Image:
        image_masked = image.copy().convert("RGBA")
        pasted_image = dst_image.copy().convert("RGBA")

        if radius != 0:
            mask_np = np.array(mask)
            kernel_size = (radius * 2 + 1, radius * 2 + 1)
            kernel = np.ones(kernel_size)
            eroded = cv2.erode(
                mask_np,
                kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            blurred_mask = cv2.GaussianBlur(eroded, kernel_size, sigmaX=sigma)
            blurred_mask = Image.fromarray(blurred_mask)
            image_masked.putalpha(blurred_mask)
        else:
            image_masked.putalpha(mask)

        pasted_image.alpha_composite(image_masked)
        return pasted_image

    def blend_swapped_to_target(
        self,
        swapped_face_image: Image.Image,
        target_img: Image.Image,
        swapped_msk: Tensor,
        hole_map: np.ndarray,
        lap_bld: bool,
        outer_dilation: int = 5,
    ) -> Image.Image:
        """
        Equivalent to the original step (6) in cropped/aligned space.
        """
        mask_bg = self.logical_or_reduce(*[swapped_msk == clz for clz in [0, 11, 4]])
        is_foreground = torch.logical_not(mask_bg)

        hole_index = hole_map[None, None] == 255
        hole_index = torch.from_numpy(hole_index).to(is_foreground.device)
        is_foreground[hole_index] = True
        foreground_mask = is_foreground.float()

        if lap_bld:
            content_mask, border_mask, full_mask = self.create_masks(
                foreground_mask,
                outer_dilation=outer_dilation,
                operation="expansion",
            )
        else:
            content_mask, border_mask, full_mask = self.create_masks(
                foreground_mask,
                outer_dilation=outer_dilation,
            )

        content_mask = F.interpolate(
            content_mask,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        content_mask_image = Image.fromarray(
            (255 * content_mask[0, 0].cpu().numpy()).astype(np.uint8)
        )

        full_mask = F.interpolate(
            full_mask,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        full_mask_image = Image.fromarray(
            (255 * full_mask[0, 0].cpu().numpy()).astype(np.uint8)
        )

        if lap_bld:
            content_mask_np = content_mask[0, 0, :, :, None].cpu().numpy()

            border_mask = F.interpolate(
                border_mask,
                (1024, 1024),
                mode="bilinear",
                align_corners=False,
            )
            border_mask = border_mask[0, 0, :, :, None].cpu().numpy()
            border_mask = np.repeat(border_mask, 3, axis=-1)

            swapped_and_pasted = np.array(swapped_face_image).astype(
                np.float32
            ) * content_mask_np + np.array(target_img).astype(np.float32) * (
                1 - content_mask_np
            )
            swapped_and_pasted = Image.fromarray(
                np.clip(swapped_and_pasted, 0, 255).astype(np.uint8)
            )
            swapped_and_pasted = Image.fromarray(
                self.blending(
                    np.array(target_img),
                    np.array(swapped_and_pasted),
                    mask=border_mask,
                )
            )
        else:
            if outer_dilation == 0:
                swapped_and_pasted = self.smooth_face_boundry(
                    swapped_face_image,
                    target_img,
                    content_mask_image,
                    radius=outer_dilation,
                )
            else:
                swapped_and_pasted = self.smooth_face_boundry(
                    swapped_face_image,
                    target_img,
                    full_mask_image,
                    radius=outer_dilation,
                )

        return swapped_and_pasted.convert("RGB")

    def paste_back_to_original(
        self,
        swapped_and_pasted: Image.Image,
        orig_image: Image.Image,
        inv_trans_coeffs: Sequence[float],
    ) -> Image.Image:
        """
        Restore cropped result back to original target image.
        """
        swapped_and_pasted = swapped_and_pasted.convert("RGBA")
        pasted_image = orig_image.convert("RGBA")

        swapped_and_pasted.putalpha(255)
        projected = swapped_and_pasted.transform(
            orig_image.size,
            Image.Transform.PERSPECTIVE,
            inv_trans_coeffs,
            Image.Resampling.BILINEAR,
        )
        pasted_image.alpha_composite(projected)
        return pasted_image.convert("RGB")
