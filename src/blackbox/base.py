from src.common_utils import cd, use_project
from src.evaluate import Utility, Effectiveness, DistanceCloakSelector

import cv2
import torch
import inspect
import warnings
import face_alignment
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch import Tensor
from argparse import Namespace
from types import MethodType
from omegaconf import OmegaConf
from pathlib import Path
from torchvision import transforms
from torch.serialization import add_safe_globals, SourceChangeWarning
from torch.nn.functional import mse_loss, l1_loss


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config

        warnings.filterwarnings("ignore", category=SourceChangeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message=r".*The parameter 'pretrained' is deprecated since 0\.13.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Arguments other than a weight enum or `None` for 'weights' are deprecated since 0\.13.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*`rcond` parameter will change to the default of machine precision.*",
            category=FutureWarning,
            module=r".*matlab_cp2tform",
        )

        self.device = torch.device("cuda")

        # simswap
        self._simswap_normalize = transforms.Compose(
            [
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        simswap_root = Path(self.config.third_party.simswap_dir)
        with use_project([simswap_root]), cd(simswap_root):
            from models.models import create_model
            from models import arcface_models

            self.test_options = Namespace(
                gpu_ids=[0],
                isTrain=False,
                checkpoints_dir="checkpoints",
                name="people",
                resize_or_crop="scale_width",
                crop_size=224,
                Arc_path="arcface_model/arcface_checkpoint.tar",
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

            setattr(
                self.target.netG, "encoder", MethodType(self.encoder, self.target.netG)
            )

        # hififace
        hififace_root = Path(self.config.third_party.hififace_dir)
        with use_project([hififace_root]), cd(hififace_root):
            from hififace_pl import HifiFace

            origin_config = OmegaConf.load(
                config.third_party.origin.hififace.config_path
            )

            self.net = HifiFace(origin_config)
            checkpoint = torch.load(
                config.third_party.origin.hififace.checkpoint_path, map_location="cpu"
            )
            self.net.load_state_dict(checkpoint["state_dict"])
            self.net = self.net.eval().to(self.device)

        # faceshifter
        faceshifter_root = Path(self.config.third_party.faceshifter_dir)
        with use_project([faceshifter_root, faceshifter_root / "face_modules"]), cd(
            faceshifter_root
        ):
            from face_modules.model import Backbone
            from face_modules.mtcnn import MTCNN
            from network.AEI_Net import AEI_Net

            self.arcface = Backbone(50, 0.6, "ir_se").cuda()
            self.arcface.load_state_dict(
                torch.load(
                    config.third_party.origin.faceshifter.model_path,
                    weights_only=True,
                ),
                strict=False,
            )
            self.arcface = self.arcface.eval().cuda()
            self.detector = MTCNN()

            self.G = AEI_Net(c_id=512)
            self.G.load_state_dict(
                torch.load(
                    config.third_party.origin.faceshifter.G_path,
                    weights_only=True,
                )
            )
            self.G = self.G.eval().cuda()

        def build_defense_target(config_name: str, model_class):
            target_config = OmegaConf.create(
                OmegaConf.to_container(self.config, resolve=False)
            )
            target_config.third_party = OmegaConf.load(
                Path(self.config.root_dir) / f"config/third_party/{config_name}.yaml"
            )
            return model_class(
                self.logger,
                OmegaConf.create(OmegaConf.to_container(target_config, resolve=True)),
            )

        if self.config.third_party.defense.target == "diffface":
            from src.diffface.base import Base as DiffFace

            self.defense_target = build_defense_target("diffface", DiffFace)
        elif self.config.third_party.defense.target == "diffswap":
            from src.diffswap.base import Base as DiffSwap

            self.defense_target = build_defense_target("diffswap", DiffSwap)
        elif self.config.third_party.defense.target == "uniface":
            from src.uniface.base import Base as UniFace

            self.defense_target = build_defense_target("uniface", UniFace)
        elif self.config.third_party.defense.target == "e4s":
            from src.e4s.base import Base as E4S

            self.defense_target = build_defense_target("e4s", E4S)
        elif self.config.third_party.defense.target == "infoswap":
            from src.infoswap.base import Base as InfoSwap

            self.defense_target = build_defense_target("infoswap", InfoSwap)
        else:
            raise ValueError(
                f"Unsupported defense target: {self.config.third_party.defense.target}"
            )

        # common
        self.utility = Utility(logger, config)
        self.effectiveness = Effectiveness(logger, config)
        self.cloak = DistanceCloakSelector(logger, config, self.effectiveness)

    @staticmethod
    def encoder(this, input):
        x = input

        x = this.first_layer(x)
        x = this.down1(x)
        x = this.down2(x)
        x = this.down3(x)
        if this.deep:
            x = this.down4(x)

        return x

    def get_simswap_identity(self, imgs: Tensor) -> Tensor:
        imgs = self._simswap_normalize(imgs)
        imgs_downsample = F.interpolate(imgs, size=(112, 112))
        prior = self.target.netArc(imgs_downsample)
        prior = prior / torch.norm(prior, p=2, dim=1)[0]

        return prior.cuda()

    def get_faceshifter_identity(self, imgs: Tensor) -> Tensor:
        return self.arcface(
            F.interpolate(
                imgs[:, :, 19:237, 19:237],
                (112, 112),
                mode="bilinear",
                align_corners=True,
            )
        )

    def get_hififace_identity(self, imgs: Tensor) -> Tensor:
        return F.normalize(
            self.net.generator.id_extractor.f_id(
                F.interpolate((imgs - 0.5) / 0.5, size=112, mode="bilinear")
            ),
            dim=-1,
            p=2,
        )
