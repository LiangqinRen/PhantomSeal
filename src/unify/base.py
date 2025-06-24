from utils import cd, use_project
from evaluate import Utility, Effectiveness, AIEditing, Cloak

import cv2
import torch
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np
from torch import tensor, nn
from argparse import Namespace
from types import MethodType
from omegaconf import OmegaConf
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config

        self.device = torch.device("cuda")

        # simswap
        with use_project([Path("third_party") / "SimSwap"]):
            from SimSwap.models.models import create_model

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

            self.target = create_model(self.test_options)
            self.target.cuda().eval()

            self.target.netG.encoder = MethodType(self.encoder, self.target.netG)

        # hififace
        with use_project([Path("third_party") / "HifiFace"]):
            with cd(Path("third_party") / "HifiFace"):
                from HifiFace.hififace_pl import HifiFace

                origin_config = OmegaConf.load(config.third_party.origin.config_path)

                self.net = HifiFace(origin_config)
                checkpoint = torch.load(
                    config.third_party.origin.checkpoint_path, map_location="cpu"
                )
                self.net.load_state_dict(checkpoint["state_dict"])
                self.net = self.net.eval().to(self.device)

        # faceshifter
        with use_project(
            [
                Path("third_party") / "FaceShifter" / "ModelC",
                Path("third_party") / "FaceShifter" / "ModelC" / "face_modules",
            ]
        ):
            from FaceShifter.ModelC.face_modules.model import Backbone
            from FaceShifter.ModelC.face_modules.mtcnn import MTCNN
            from FaceShifter.ModelC.network.AEI_Net import AEI_Net

            self.arcface = Backbone(50, 0.6, "ir_se").cuda()
            self.arcface.load_state_dict(
                torch.load(
                    config.third_party.origin.model_path,
                    weights_only=True,
                ),
                strict=False,
            )
            self.arcface = self.arcface.eval().cuda()

            with cd(Path("third_party") / "FaceShifter" / "ModelC"):
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
        self.aiediting = AIEditing(logger, config)
        self.cloak = Cloak(logger, config, self.effectiveness)

    def _get_imgs_identity(self, imgs: tensor) -> tensor:
        imgs_downsample = F.interpolate(imgs, size=(112, 112))
        prior = self.target.netArc(imgs_downsample)
        prior = prior / torch.norm(prior, p=2, dim=1)[0]

        return prior.cuda()

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

    def _simswap_swapface(self, src_img: tensor, tgt_img: tensor) -> tensor:
        with torch.no_grad():
            src_id = self._get_imgs_identity(src_img)
            swap_result = self.target(None, tgt_img, src_id, None, True)

        return swap_result

    def _hififace_swapface(self, src_img: tensor, tgt_img: tensor) -> tensor:
        return self.net(src_img, tgt_img)

    def _faceshifter_swapface(self, src_img: tensor, tgt_img: tensor) -> tensor:
        def swapface(src_img: tensor, tgt_img: tensor) -> tensor:
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
                self.logger.warning(f"faceshifter swap error: {e}")
                id_swap_list.append(torch.zeros(1, 3, 256, 256))

        swap_results = torch.cat(id_swap_list, dim=0)
        return torch.clamp(swap_results, 0, 1).cuda()

    def _to_ndarray(self, img: torch.tensor) -> np.ndarray:
        img = (img * 0.5 + 0.5) * 255
        img = img.squeeze(0)
        img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        img = img.astype(np.uint8)

        return img

    def _denormalize(self, img: tensor) -> tensor:
        return img * 0.5 + 0.5
