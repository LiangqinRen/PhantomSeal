from utils import cd
from evaluate import Utility, Effectiveness, AIEditing, Cloak
from face_modules.model import Backbone
from face_modules.mtcnn import MTCNN
from network.AEI_Net import AEI_Net


import cv2
import torch
import PIL.Image as Image
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torch import tensor
from pathlib import Path


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config

        self.utility = Utility(logger, config)
        self.effectiveness = Effectiveness(logger, config)
        self.aiediting = AIEditing(logger, config)
        self.cloak = Cloak(logger, config, self.effectiveness)

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

    def swapface(self, src_img: tensor, tgt_img: tensor) -> tensor:
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
            Yt = Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0]) * 0.5 + 0.5
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

    def _to_ndarray(self, img: torch.tensor) -> np.ndarray:
        img = (img * 0.5 + 0.5) * 255
        img = img.squeeze(0)
        img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        img = img.astype(np.uint8)

        return img

    def _denormalize(self, img: tensor) -> tensor:
        return img * 0.5 + 0.5
