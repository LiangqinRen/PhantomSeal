from src.utils import cd, use_project
from src.evaluate import Utility, Effectiveness, DistanceCloakSelector


import torch
from torch import nn, Tensor
from pathlib import Path


class Model(nn.Module):
    def __init__(self, Generator, Encoder, Colorize, origin_config):
        super(Model, self).__init__()
        self.g_ema = Generator(
            origin_config.size,
            origin_config.latent_channel_size,
            origin_config.latent_spatial_size,
            lr_mul=origin_config.lr_mul,
            channel_multiplier=origin_config.channel_multiplier,
            normalize_mode=origin_config.normalize_mode,
            small_generator=origin_config.small_generator,
        )
        self.e_ema = Encoder(
            origin_config.size,
            origin_config.latent_channel_size,
            origin_config.latent_spatial_size,
            channel_multiplier=origin_config.channel_multiplier,
        )

        self.Colorize = Colorize

    def tensor2label(self, label_tensor, n_label):
        label_tensor = label_tensor.cpu().float()
        if label_tensor.size()[0] > 1:
            label_tensor = label_tensor.max(0, keepdim=True)[1]
        label_tensor = self.Colorize(n_label)(label_tensor)
        label_numpy = label_tensor.numpy()

        return label_numpy

    def forward(self, input):
        trg = input[0]
        src = input[1]

        trg_src = torch.cat([trg, src], dim=0)
        w, w_feat = self.e_ema(trg_src)
        w_feat_tgt = [torch.chunk(f, 2, dim=0)[0] for f in w_feat][::-1]
        trg_w, src_w = torch.chunk(w, 2, dim=0)
        fake_img = self.g_ema([trg_w, src_w, w_feat_tgt])

        return trg, src, fake_img


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config

        # self.utility = Utility(logger, config)
        # self.effectiveness = Effectiveness(logger, config)
        # self.cloak = DistanceCloakSelector(logger, config, self.effectiveness)

        self.device = "cuda:0"

        origin_config = config.third_party.origin
        root_dir = Path(self.config.third_party.project_root)
        with use_project([root_dir]):
            from training.model import Generator_globalatt_return_32 as Generator
            from training.model import Encoder_return_32 as Encoder
            from generate_swap import Colorize

            checkpoint = torch.load(origin_config.checkpoint_path, weights_only=False)
            self.model = Model(Generator, Encoder, Colorize, origin_config).to(
                self.device
            )
            self.model.g_ema.load_state_dict(checkpoint["g_ema"])
            self.model.e_ema.load_state_dict(checkpoint["e_ema"])
            self.model.eval()

    def swap_face(self, source_imgs: Tensor, target_imgs: Tensor) -> Tensor:
        _, _, results = self.model([target_imgs, source_imgs])

        return results.clamp(-1, 1)
