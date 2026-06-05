from src.common_utils import cd, use_project, check_tensor_info
from src.evaluate import Utility, Effectiveness, AIEditing, DistanceCloakSelector

import torch
import warnings
from omegaconf import OmegaConf
from pathlib import Path


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config

        warnings.filterwarnings(
            "ignore",
            message=".*parameter 'pretrained' is deprecated.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=".*Arguments other than a weight enum.*",
        )

        self.utility = Utility(logger, config)
        self.effectiveness = Effectiveness(logger, config)
        self.aiediting = AIEditing(logger, config)
        self.cloak = DistanceCloakSelector(logger, config, self.effectiveness)

        self.device = torch.device("cuda")
        root_dir = Path(self.config.third_party.project_root)
        with cd(root_dir), use_project([root_dir]):
            from hififace_pl import HifiFace

            config_path = Path(self.config.third_party.origin.config_path)
            if not config_path.is_absolute():
                config_path = root_dir / config_path
            checkpoint_path = Path(self.config.third_party.origin.checkpoint_path)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(self.config.root_dir) / checkpoint_path

            origin_config = OmegaConf.load(config_path)

            self.net = HifiFace(origin_config)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.net.load_state_dict(checkpoint["state_dict"])
            self.net = self.net.eval().to(self.device)

    def swap_face(
        self, source_imgs: torch.Tensor, target_imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard HifiFace swap interface.

        Args:
            source_imgs:
                [B, 3, H, W] float tensor in [0, 1]. Identity comes from these
                images.
            target_imgs:
                [B, 3, H, W] float tensor in [0, 1]. Pose/expression/background come
                from these images.

        Returns:
            [B, 3, H, W] float tensor in [0, 1].

        Notes:
            The upstream HifiFace model consumes inputs in [0, 1], but its raw
            forward output is not strictly bounded to [0, 1]. The official
            inference script clamps the generated image before saving, so this
            wrapper does the same and exposes a clamped [0, 1] tensor as the
            standard interface output.
        """
        return torch.clamp(self.net(source_imgs, target_imgs), min=0.0, max=1.0)
