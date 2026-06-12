import sys

import hydra
from omegaconf import DictConfig

from src import common_utils
from src.denoiser.train import test_with_config, train_with_config


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    logger = common_utils.get_customized_logger(config.log.record_level)
    common_utils.check_required_api_keys(config, logger)

    common_utils.check_cuda_availability(logger)
    common_utils.fix_random_seed(logger, config.random_seed)
    timer = common_utils.Timer(f"Denoiser {config.third_party.function}", logger)

    if config.third_party.function == "train":
        train_with_config(config, logger)
    elif config.third_party.function == "test":
        test_with_config(config, logger)
    else:
        sys.exit(f"Fail to find {config.third_party.function}")

    del timer


if __name__ == "__main__":
    main()
