from pathlib import Path
from types import MethodType
from typing import Any, Callable, cast

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

from src.common_utils import cd, use_project
from src.simswap.options import build_simswap_test_options

try:
    from torch.serialization import add_safe_globals
except ImportError:
    def add_safe_globals(_: list[Any]) -> None:
        return None


class SimSwapValidator:
    """
    Minimal SimSwap wrapper for denoiser internal validation.

    This intentionally avoids src.simswap.base.Base because that path also
    initializes cloak selection, which is not needed when evaluating denoised
    perturb images.
    """

    def __init__(self, config: Any) -> None:
        root_dir = Path(config.third_party.project_root)
        with use_project([root_dir]), cd(root_dir):
            from models.models import create_model
            from models import arcface_models

            options = build_simswap_test_options(config)

            add_safe_globals(
                [
                    nn.Conv2d,
                    nn.Linear,
                    nn.BatchNorm2d,
                    nn.BatchNorm1d,
                    nn.ReLU,
                    nn.PReLU,
                    nn.Sigmoid,
                    nn.Dropout,
                    nn.Sequential,
                    nn.MaxPool2d,
                    nn.AdaptiveAvgPool2d,
                ]
            )
            for _, obj in inspect.getmembers(arcface_models):
                if inspect.isfunction(obj):
                    add_safe_globals([obj])
                elif inspect.isclass(obj):
                    add_safe_globals([cast(Callable[..., Any], obj)])

            self.target = create_model(options)
            self.target.cuda().eval()

        self.transformer_arcface = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

        netG = cast(nn.Module, self.target.netG)
        setattr(netG, "encoder", MethodType(self.encoder, netG))

    def _get_imgs_identity(self, imgs: Tensor) -> Tensor:
        imgs = self.transformer_arcface(imgs)
        imgs_downsample = F.interpolate(imgs, size=(112, 112))
        netArc = cast(nn.Module, self.target.netArc)
        prior = netArc(imgs_downsample)
        prior = F.normalize(prior, p=2, dim=1)
        return prior.cuda()

    @torch.no_grad()
    def swap_face(self, source_imgs: Tensor, target_imgs: Tensor) -> Tensor:
        source_identity = self._get_imgs_identity(source_imgs)
        return torch.clamp(
            self.target(None, target_imgs, source_identity, None, True),
            min=0.0,
            max=1.0,
        )

    @staticmethod
    def encoder(this: Any, input: Tensor) -> Tensor:
        x = input
        x = this.first_layer(x)
        x = this.down1(x)
        x = this.down2(x)
        x = this.down3(x)
        if this.deep:
            x = this.down4(x)
        return x
