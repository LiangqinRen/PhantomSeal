from src.simswap.base import Base
from src.utils import save_tensor_imgs

import os
import io
import random
import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm
from torch import tensor, Tensor
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
from pathlib import Path


class Robustness(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

    def load_logo(self) -> Tensor:
        logo = self._load_imgs([self.config.third_party.robustness.logo_img_path])
        return logo

    def gauss_noise(
        self, pert: Tensor, gauss_mean: float = 0, gauss_std: float = 0.1
    ) -> Tensor:
        gauss_noise = gauss_mean + gauss_std * torch.randn(pert.shape).cuda()
        noise_pert = pert + gauss_noise

        return noise_pert

    def webp_compress(self, imgs: Tensor, quality: float = 80):
        compressed_imgs = []
        for i in range(imgs.size(0)):
            img = imgs[i]
            pil_img = ToPILImage()(img)

            buffer = io.BytesIO()
            pil_img.save(buffer, format="WEBP", quality=quality)
            buffer.seek(0)

            compressed_img = Image.open(buffer)
            compressed_img = ToTensor()(compressed_img)

            compressed_imgs.append(compressed_img)

        return torch.stack(compressed_imgs).cuda()

    def crop(self, imgs: Tensor, thickness: float = 20) -> Tensor:
        crop_imgs = imgs.clone()
        crop_imgs[:, :, :thickness, :] = 0
        crop_imgs[:, :, -thickness:, :] = 0
        crop_imgs[:, :, :, :thickness] = 0
        crop_imgs[:, :, :, -thickness:] = 0

        return crop_imgs

    def logo(self, imgs: Tensor, logo: Tensor) -> Tensor:
        alpha = 0.75
        _, _, img_height, img_width = imgs.shape
        logo = F.interpolate(logo, size=(30, 90), mode="bilinear", align_corners=False)
        _, _, logo_height, logo_width = logo.shape

        x_offset = img_width - logo_width
        y_offset = img_height - logo_height

        logo_imgs = imgs.clone()

        logo_imgs[
            :, :, y_offset : y_offset + logo_height, x_offset : x_offset + logo_width
        ] = (
            imgs[
                :,
                :,
                y_offset : y_offset + logo_height,
                x_offset : x_offset + logo_width,
            ]
            * (1 - alpha)
            + logo * alpha
        )

        return logo_imgs

    def brightness(self, imgs, brightness_factor: float):
        adjusted_tensor = imgs * brightness_factor
        adjusted_tensor = torch.clamp(adjusted_tensor, 0, 1)

        return adjusted_tensor

    def get_gauss_noise_metrics(
        self,
        idx: int,
        imgs_A: Tensor,
        imgs_B: Tensor,
        x_imgs: Tensor,
        cloak_imgs: Tensor,
        image_dir: Path,
    ) -> tuple[dict, dict]:
        gauss_mean, gauss_std = 0, 0.1

        noise_imgs = self.gauss_noise(imgs_A, gauss_mean, gauss_std)
        noise_identity = self._get_imgs_identity(noise_imgs)
        noise_pert_imgs = self.gauss_noise(x_imgs, gauss_mean, gauss_std)
        noise_pert_identity = self._get_imgs_identity(noise_pert_imgs)

        imgs_B_identity = self._get_imgs_identity(imgs_B)
        noise_swap_imgs = self.target(None, imgs_B, noise_identity, None, True)
        reverse_noise_swap_imgs = self.target(
            None, noise_imgs, imgs_B_identity, None, True
        )
        noise_pert_swap_imgs = self.target(
            None, imgs_B, noise_pert_identity, None, True
        )
        reverse_noise_pert_swap_imgs = self.target(
            None, noise_pert_imgs, imgs_B_identity, None, True
        )

        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs_A,
            None,
            noise_swap_imgs,
            noise_pert_swap_imgs,
            cloak_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs_B, None, reverse_noise_swap_imgs, reverse_noise_pert_swap_imgs, None
        )

        save_tensor_imgs(
            image_dir,
            idx,
            [
                "imgs_A",
                "imgs_B",
                "noise_imgs",
                "noise_swap\nimgs",
                "reverse\nnoise_swap\nimgs",
                "x_imgs",
                "noise_pert_imgs",
                "noise\npert_swap\nimgs",
                "reverse\nnoise_pert_swap\nimgs",
            ],
            [
                imgs_A,
                imgs_B,
                noise_imgs,
                noise_swap_imgs,
                reverse_noise_swap_imgs,
                x_imgs,
                noise_pert_imgs,
                noise_pert_swap_imgs,
                reverse_noise_pert_swap_imgs,
            ],
            image_name="noise",
            only_save_summary=self.config.third_party.defense.only_save_summary,
        )

        return (
            source_effectivenesses,
            target_effectivenesses,
        )

    def get_compress_metrics(
        self,
        idx: int,
        imgs_A: Tensor,
        imgs_B: Tensor,
        x_imgs: Tensor,
        cloak_imgs: Tensor,
        image_dir: Path,
    ) -> tuple[dict, dict]:
        compress_rate = 80
        compress_imgs = self.webp_compress(imgs_A, compress_rate)
        compress_identity = self._get_imgs_identity(compress_imgs)
        compress_pert_imgs = self.webp_compress(x_imgs, compress_rate)
        compress_pert_identity = self._get_imgs_identity(compress_pert_imgs)

        imgs_B_identity = self._get_imgs_identity(imgs_B)
        compress_swap_imgs = self.target(None, imgs_B, compress_identity, None, True)
        reverse_compress_swap_imgs = self.target(
            None, compress_imgs, imgs_B_identity, None, True
        )
        compress_pert_swap_imgs = self.target(
            None, imgs_B, compress_pert_identity, None, True
        )
        reverse_compress_pert_swap_imgs = self.target(
            None, compress_pert_imgs, imgs_B_identity, None, True
        )

        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs_A,
            None,
            compress_swap_imgs,
            compress_pert_swap_imgs,
            cloak_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs_B,
            None,
            reverse_compress_swap_imgs,
            reverse_compress_pert_swap_imgs,
            None,
        )

        save_tensor_imgs(
            image_dir,
            idx,
            [
                "imgs_A",
                "imgs_B",
                "compress_imgs",
                "compress_swap\nimgs",
                "reverse\ncompress_swap\nimgs",
                "x_imgs",
                "compress_pert_imgs",
                "compress\npert_swap\nimgs",
                "reverse\ncompress_pert_swap\nimgs",
            ],
            [
                imgs_A,
                imgs_B,
                compress_imgs,
                compress_swap_imgs,
                reverse_compress_swap_imgs,
                x_imgs,
                compress_pert_imgs,
                compress_pert_swap_imgs,
                reverse_compress_pert_swap_imgs,
            ],
            image_name="compress",
            only_save_summary=self.config.third_party.defense.only_save_summary,
        )

        return (
            source_effectivenesses,
            target_effectivenesses,
        )

    def get_crop_metrics(
        self,
        idx: int,
        imgs_A: Tensor,
        imgs_B: Tensor,
        x_imgs: Tensor,
        cloak_imgs: Tensor,
        image_dir: Path,
    ) -> tuple[dict, dict]:
        border_thickness = 20

        crop_imgs = self.crop(imgs_A, border_thickness)
        crop_identity = self._get_imgs_identity(crop_imgs)
        crop_pert_imgs = self.crop(x_imgs, border_thickness)
        crop_pert_identity = self._get_imgs_identity(crop_pert_imgs)

        imgs_B_identity = self._get_imgs_identity(imgs_B)
        crop_swap_imgs = self.target(None, imgs_B, crop_identity, None, True)
        reverse_crop_swap_imgs = self.target(
            None, crop_imgs, imgs_B_identity, None, True
        )
        crop_pert_swap_imgs = self.target(None, imgs_B, crop_pert_identity, None, True)
        reverse_crop_pert_swap_imgs = self.target(
            None, crop_pert_imgs, imgs_B_identity, None, True
        )

        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs_A,
            None,
            crop_swap_imgs,
            crop_pert_swap_imgs,
            cloak_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs_B, None, reverse_crop_swap_imgs, reverse_crop_pert_swap_imgs, None
        )

        save_tensor_imgs(
            image_dir,
            idx,
            [
                "imgs_A",
                "imgs_B",
                "crop_imgs",
                "crop_swap\nimgs",
                "reverse\ncrop_swap\nimgs",
                "x_imgs",
                "crop_pert_imgs",
                "crop\npert_swap\nimgs",
                "reverse\ncrop_pert_swap\nimgs",
            ],
            [
                imgs_A,
                imgs_B,
                crop_imgs,
                crop_swap_imgs,
                reverse_crop_swap_imgs,
                x_imgs,
                crop_pert_imgs,
                crop_pert_swap_imgs,
                reverse_crop_pert_swap_imgs,
            ],
            image_name="crop",
            only_save_summary=self.config.third_party.defense.only_save_summary,
        )

        return (
            source_effectivenesses,
            target_effectivenesses,
        )

    def get_logo_metrics(
        self,
        idx: int,
        imgs_A: Tensor,
        imgs_B: Tensor,
        x_imgs: Tensor,
        logo: Tensor,
        cloak_imgs: Tensor,
        image_dir: Path,
    ) -> tuple[dict, dict]:
        logo_imgs = self.logo(imgs_A, logo)
        logo_identity = self._get_imgs_identity(logo_imgs)
        logo_pert_imgs = self.logo(x_imgs, logo)
        logo_pert_identity = self._get_imgs_identity(logo_pert_imgs)

        imgs_B_identity = self._get_imgs_identity(imgs_B)
        logo_swap_imgs = self.target(None, imgs_B, logo_identity, None, True)
        reverse_logo_swap_imgs = self.target(
            None, logo_imgs, imgs_B_identity, None, True
        )
        logo_pert_swap_imgs = self.target(None, imgs_B, logo_pert_identity, None, True)
        reverse_logo_pert_swap_imgs = self.target(
            None, logo_pert_imgs, imgs_B_identity, None, True
        )

        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs_A,
            None,
            logo_swap_imgs,
            logo_pert_swap_imgs,
            cloak_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs_B, None, reverse_logo_swap_imgs, reverse_logo_pert_swap_imgs, None
        )

        save_tensor_imgs(
            image_dir,
            idx,
            [
                "imgs_A",
                "imgs_B",
                "logo_imgs",
                "logo_swap\nimgs",
                "reverse\nlogo_swap\nimgs",
                "x_imgs",
                "logo_pert_imgs",
                "logo\npert_swap\nimgs",
                "reverse\nlogo_pert_swap\nimgs",
            ],
            [
                imgs_A,
                imgs_B,
                logo_imgs,
                logo_swap_imgs,
                reverse_logo_swap_imgs,
                x_imgs,
                logo_pert_imgs,
                logo_pert_swap_imgs,
                reverse_logo_pert_swap_imgs,
            ],
            image_name="logo",
            only_save_summary=self.config.third_party.defense.only_save_summary,
        )

        return (
            source_effectivenesses,
            target_effectivenesses,
        )

    def get_brightness_metrics(
        self,
        idx: int,
        imgs_A: Tensor,
        imgs_B: Tensor,
        x_imgs: Tensor,
        factor: float,
        cloak_imgs: Tensor,
        image_dir: Path,
    ) -> tuple[dict, dict]:
        brightness_imgs = self.brightness(imgs_A, factor)
        brightness_identity = self._get_imgs_identity(brightness_imgs)
        brightness_pert_imgs = self.brightness(x_imgs, factor)
        brightness_pert_identity = self._get_imgs_identity(brightness_pert_imgs)

        imgs_B_identity = self._get_imgs_identity(imgs_B)
        brightness_swap_imgs = self.target(
            None, imgs_B, brightness_identity, None, True
        )
        reverse_brightness_swap_imgs = self.target(
            None, brightness_imgs, imgs_B_identity, None, True
        )
        brightness_pert_swap_imgs = self.target(
            None, imgs_B, brightness_pert_identity, None, True
        )
        reverse_brightness_pert_swap_imgs = self.target(
            None, brightness_pert_imgs, imgs_B_identity, None, True
        )

        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs_A,
            None,
            brightness_swap_imgs,
            brightness_pert_swap_imgs,
            cloak_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs_B,
            None,
            reverse_brightness_swap_imgs,
            reverse_brightness_pert_swap_imgs,
            None,
        )

        save_tensor_imgs(
            image_dir,
            idx,
            [
                "imgs_A",
                "imgs_B",
                f"brightness_{factor}\nimgs",
                f"brightness_{factor}\nswap_imgs",
                f"reverse\nbrightness_{factor}\nswap_imgs",
                "x_imgs",
                f"brightness_{factor}\npert_imgs",
                f"brightness_{factor}\npert_swap\nimgs",
                f"reverse\nbrightness_{factor}\npert_swap\nimgs",
            ],
            [
                imgs_A,
                imgs_B,
                brightness_imgs,
                brightness_swap_imgs,
                reverse_brightness_swap_imgs,
                x_imgs,
                brightness_pert_imgs,
                brightness_pert_swap_imgs,
                reverse_brightness_pert_swap_imgs,
            ],
            image_name=f"brightness_{factor}",
            only_save_summary=self.config.third_party.defense.only_save_summary,
        )

        return (
            source_effectivenesses,
            target_effectivenesses,
        )

    def _load_imgs(self, imgs_path) -> Tensor:
        if self.config.third_party.dataset.use_224:
            transformer = transforms.Compose(
                [transforms.Resize(224), transforms.ToTensor()]
            )
        else:
            transformer = transforms.Compose(
                [transforms.Resize(256), transforms.ToTensor()]
            )

        imgs = []
        for path in imgs_path:
            img = transformer(Image.open(path).convert("RGB"))
            if not isinstance(img, torch.Tensor):
                img = ToTensor()(img)
            imgs.append(img)
        imgs = torch.stack(imgs)

        return imgs.cuda()
