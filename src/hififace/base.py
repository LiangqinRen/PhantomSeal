from utils import cd
from evaluate import Utility, Effectiveness, AIEditing, Cloak
from hififace_pl import HifiFace

import torch
from omegaconf import OmegaConf
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
        with cd(Path("third_party") / "HifiFace"):
            origin_config = OmegaConf.load(config.third_party.origin.config_path)

            self.net = HifiFace(origin_config)
            checkpoint = torch.load(
                config.third_party.origin.checkpoint_path, map_location="cpu"
            )
            self.net.load_state_dict(checkpoint["state_dict"])
            self.net = self.net.eval().to(self.device)
