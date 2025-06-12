# from evaluate import Utility, Effectiveness, Anchor, AIEditing
from models.models import create_model

# from options.test_options import TestOptions

import torch
import torch.nn.functional as F
from argparse import Namespace
from torch import tensor


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config

        self.test_options = Namespace(
            # minimal required configuration parameters from SimSwap
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

    def _get_imgs_identity(self, imgs: tensor) -> tensor:
        imgs_downsample = F.interpolate(imgs, size=(112, 112))
        prior = self.target.netArc(imgs_downsample)
        prior = prior / torch.norm(prior, p=2, dim=1)[0]

        return prior.cuda()
