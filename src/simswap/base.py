from src.evaluate import (
    Utility,
    Effectiveness,
    AIEditing,
    Cloak,
    DistanceCloakSelector,
)
from src.common_utils import cd, use_project, check_tensor_info
from src.simswap.options import build_simswap_test_options

import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from types import MethodType
from torch.serialization import add_safe_globals
from torchvision import transforms
from pathlib import Path
from typing import Callable, Any, cast


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config

        root_dir = Path(config.third_party.project_root)
        with use_project([root_dir]), cd(root_dir):
            from models.models import create_model
            from models import arcface_models

            self.test_options = build_simswap_test_options(config)

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

            self.target = create_model(self.test_options)
            self.target.cuda().eval()

        self.transformer_Arcface = transforms.Compose(
            [
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        netG = cast(nn.Module, self.target.netG)
        setattr(netG, "encoder", MethodType(self.encoder, netG))

        self.utility = Utility(logger, config)
        self.effectiveness = Effectiveness(logger, config)
        self.aiediting = AIEditing(logger, config)
        self.cloak = DistanceCloakSelector(logger, config, self.effectiveness)

    def _get_imgs_identity(self, imgs: Tensor) -> Tensor:
        imgs = self.transformer_Arcface(imgs)
        imgs_downsample = F.interpolate(imgs, size=(112, 112))
        netArc = cast(nn.Module, self.target.netArc)
        prior = netArc(imgs_downsample)
        prior = F.normalize(prior, p=2, dim=1)

        return prior.cuda()

    def swap_face(self, source_imgs: Tensor, target_imgs: Tensor) -> Tensor:
        """
        Standard SimSwap swap interface.

        Args:
            source_imgs:
                [B, 3, H, W] float tensor in [0, 1]. Identity is extracted from these images and injected into the target images.
            target_imgs:
                [B, 3, H, W] float tensor in [0, 1]. Face/content comes from these
                images.

        Returns:
            [B, 3, H, W] float tensor in [0, 1].
        """

        source_identity = self._get_imgs_identity(source_imgs)
        return torch.clamp(
            self.target(None, target_imgs, source_identity, None, True),
            min=0.0,
            max=1.0,
        )

    @staticmethod
    def encoder(this, input: Tensor) -> Tensor:
        x = input

        x = this.first_layer(x)
        x = this.down1(x)
        x = this.down2(x)
        x = this.down3(x)
        if this.deep:
            x = this.down4(x)

        return x
