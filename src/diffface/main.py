import src.utils
from src.diffface.defense import Defense

import sys
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    logger = src.utils.get_customized_logger(config.log.record_level)

    src.utils.check_cuda_availability(logger)
    src.utils.fix_random_seed(logger, config.random_seed)
    timer = src.utils.Timer("main", logger)

    defense = Defense(logger, config)
    defense_functions = {"metric": defense.metric}

    if config.third_party.function in defense_functions:
        defense_functions[config.third_party.function]()
    else:
        sys.exit(f"⚠️ Oops! Fail to find {config.third_party.function}")


if __name__ == "__main__":
    main()
