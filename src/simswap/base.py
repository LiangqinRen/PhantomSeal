from models.models import create_model
from evaluate import Utility, Effectiveness, AIEditing, Cloak
from models import arcface_models


import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from torch import Tensor
from types import MethodType
from torch.serialization import add_safe_globals


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config

        self.test_options = Namespace(
            gpu_ids=[0],
            isTrain=False,
            checkpoints_dir="third_party/SimSwap/checkpoints",
            name="people",
            resize_or_crop="scale_width",
            crop_size=224,
            Arc_path="third_party/SimSwap/arcface_model/arcface_checkpoint.tar",
            which_epoch="latest",
            verbose=False,
        )

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
                from typing import Callable, Any, cast

                add_safe_globals([cast(Callable[..., Any], obj)])

        self.target = create_model(self.test_options)
        self.target.cuda().eval()

        self.target.netG.encoder = MethodType(self.encoder, self.target.netG)

        self.utility = Utility(logger, config)
        self.effectiveness = Effectiveness(logger, config)
        self.aiediting = AIEditing(logger, config)
        self.cloak = Cloak(logger, config, self.effectiveness)

    def _get_imgs_identity(self, imgs: Tensor) -> Tensor:
        imgs_downsample = F.interpolate(imgs, size=(112, 112))
        prior = self.target.netArc(imgs_downsample)
        prior = prior / torch.norm(prior, p=2, dim=1)[0]

        return prior.cuda()

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
