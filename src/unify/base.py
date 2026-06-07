from src.common_utils import cd, use_project
from src.simswap.options import build_simswap_test_options
from src.evaluate import Utility, Effectiveness, DistanceCloakSelector

import cv2
import torch
import inspect
import warnings
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np
import torch.nn as nn
from torch import Tensor
from types import MethodType
from omegaconf import OmegaConf
from pathlib import Path
from torchvision import transforms
from torch.serialization import add_safe_globals


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config

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

            origin_config = OmegaConf.load(config.third_party.origin.config_path)

            self.net = HifiFace(origin_config)
            checkpoint = torch.load(
                config.third_party.origin.checkpoint_path, map_location="cpu"
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
                    config.third_party.origin.model_path,
                    weights_only=True,
                ),
                strict=False,
            )
            self.arcface = self.arcface.eval().cuda()
            self.detector = MTCNN()

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
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
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

    def _model_256_input(self, imgs: Tensor) -> Tensor:
        if imgs.shape[-2:] == (256, 256):
            return imgs
        return F.interpolate(imgs, size=(256, 256), mode="bilinear", align_corners=False)

    def get_simswap_identity(self, imgs: Tensor) -> Tensor:
        imgs = self._simswap_normalize(imgs)
        imgs_downsample = F.interpolate(imgs, size=(112, 112))
        prior = self.target.netArc(imgs_downsample)
        prior = prior / torch.norm(prior, p=2, dim=1)[0]

        return prior.cuda()

    def get_faceshifter_identity(self, imgs: Tensor) -> Tensor:
        imgs = self._model_256_input(imgs)
        return self.arcface(
            F.interpolate(
                imgs[:, :, 19:237, 19:237],
                (112, 112),
                mode="bilinear",
                align_corners=True,
            )
        )

    def get_hififace_identity(self, imgs: Tensor) -> Tensor:
        imgs = self._model_256_input(imgs)
        return F.normalize(
            self.net.generator.id_extractor.f_id(
                F.interpolate((imgs - 0.5) / 0.5, size=112, mode="bilinear")
            ),
            dim=-1,
            p=2,
        )

    def _simswap_swapface(self, src_img: Tensor, tgt_img: Tensor) -> Tensor:
        with torch.no_grad():
            src_id = self.get_simswap_identity(src_img)
            swap_result = self.target(None, tgt_img, src_id, None, True)

        return swap_result

    def _hififace_swapface(self, src_img: Tensor, tgt_img: Tensor) -> Tensor:
        return torch.clamp(self.net(src_img, tgt_img), 0, 1)

    def _faceshifter_swapface(self, src_img: Tensor, tgt_img: Tensor) -> Tensor:
        def swapface(src_img: Tensor, tgt_img: Tensor) -> Tensor:
            with torch.no_grad():
                embeds = self.arcface(
                    F.interpolate(
                        src_img[:, :, 19:237, 19:237],
                        (112, 112),
                        mode="bilinear",
                        align_corners=True,
                    )
                )

            tgt_img_raw = self._to_ndarray(tgt_img)
            tgt_img, trans_inv = self.detector.align(
                Image.fromarray(tgt_img_raw),
                crop_size=(256, 256),
                return_trans_inv=True,
            )
            tgt_img_raw = tgt_img_raw.astype(float) / 255.0
            tgt_img = self._transform(tgt_img).unsqueeze(0).cuda()

            mask = np.zeros([256, 256], dtype=float)
            for i in range(256):
                for j in range(256):
                    dist = np.sqrt((i - 128) ** 2 + (j - 128) ** 2) / 128
                    dist = np.minimum(dist, 1)
                    mask[i, j] = 1 - dist
            mask = cv2.dilate(mask, None, iterations=20)

            with torch.no_grad():
                Yt, _ = self.G(tgt_img, embeds)
                Yt = (
                    Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0]) * 0.5 + 0.5
                )
                Yt_trans_inv = cv2.warpAffine(
                    Yt,
                    trans_inv,
                    (np.size(tgt_img_raw, 1), np.size(tgt_img_raw, 0)),
                    borderValue=(0, 0, 0),
                )
                mask_ = cv2.warpAffine(
                    mask,
                    trans_inv,
                    (np.size(tgt_img_raw, 1), np.size(tgt_img_raw, 0)),
                    borderValue=(0, 0, 0),
                )
                mask_ = np.expand_dims(mask_, 2)
                Yt_trans_inv = mask_ * Yt_trans_inv + (1 - mask_) * tgt_img_raw

            return transforms.ToTensor()(Yt_trans_inv).unsqueeze(0)

        norm_src_img, norm_tgt_img = self._normalize(src_img), self._normalize(tgt_img)
        batch_size = self.config.third_party.defense.batch_size
        src_imgs = list(torch.chunk(norm_src_img, chunks=batch_size, dim=0))
        tgt_imgs = list(torch.chunk(norm_tgt_img, chunks=batch_size, dim=0))
        id_swap_list = []
        for i in range(len(src_imgs)):
            try:
                id_swap_list.append(swapface(src_imgs[i], tgt_imgs[i]))
            except Exception as e:
                id_swap_list.append(torch.zeros(1, 3, 256, 256))

        swap_results = torch.cat(id_swap_list, dim=0)
        return torch.clamp(swap_results, 0, 1).cuda()

    def _to_ndarray(self, img: Tensor) -> np.ndarray:
        img = (img * 0.5 + 0.5) * 255
        img = img.squeeze(0)
        img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        img = img.astype(np.uint8)

        return img

    def _denormalize(self, img: Tensor) -> Tensor:
        return img * 0.5 + 0.5
