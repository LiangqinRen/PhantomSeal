import sys

import hydra
from omegaconf import DictConfig

from src import common_utils
from src.cuma.defense import Defense


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    logger = common_utils.get_customized_logger(config.log.record_level)

    common_utils.check_cuda_availability(logger)
    common_utils.fix_random_seed(logger, config.random_seed)
    timer = common_utils.Timer(f"CUMA {config.third_party.function}", logger)

    defense = Defense(logger, config)
    defense_function_list = ["metric"]
    defense_functions = {name: getattr(defense, name) for name in defense_function_list}

    if config.third_party.function in defense_functions:
        defense_functions[config.third_party.function]()
    else:
        sys.exit(f"Fail to find CUMA function: {config.third_party.function}")


if __name__ == "__main__":
    main()
