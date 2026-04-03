from src.common_utils import cd, use_project

from pathlib import Path


class Base:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

        root_dir = Path(config.third_party.project_root)
        with use_project([root_dir]), cd(root_dir):
            from test_Dual_Mark import seed_torch
            from network.Dual_Mark import Network

            seed_torch(config.random_seed)

            self.device = "cuda"
            self.network = Network(
                config.third_party.origin.message_length,
                [],
                [],
                self.device,
                config.third_party.origin.batch_size,
                config.third_party.origin.lr,
                config.third_party.origin.beta1,
                config.third_party.origin.attention_encoder,
                config.third_party.origin.attention_decoder,
                config.third_party.origin.weight,
            )
            self.network.load_model_ed(config.third_party.defense.model_path)

            self.network.encoder_decoder.eval()
            self.network.discriminator.eval()
