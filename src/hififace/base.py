from src.utils import cd, use_project
from src.evaluate import Utility, Effectiveness, AIEditing, DistanceCloakSelector

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
        self.cloak = DistanceCloakSelector(logger, config, self.effectiveness)

        self.device = torch.device("cuda")
        root_dir = Path(self.config.third_party.project_root)
        with cd(root_dir), use_project([root_dir]):
            from hififace_pl import HifiFace

            origin_config = OmegaConf.load(config.third_party.origin.config_path)

            self.net = HifiFace(origin_config)
            checkpoint = torch.load(
                config.third_party.origin.checkpoint_path, map_location="cpu"
            )
            self.net.load_state_dict(checkpoint["state_dict"])
            self.net = self.net.eval().to(self.device)
