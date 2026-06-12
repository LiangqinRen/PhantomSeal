from src import common_utils
from src.sepmark.defense import Defense

import sys
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    logger = common_utils.get_customized_logger(config.log.record_level)
    common_utils.check_required_api_keys(config, logger)

    common_utils.check_cuda_availability(logger)
    timer = common_utils.Timer("main", logger)

    defense = Defense(logger, config)
    defense_function_list = ["forensics_robustness_metric"]
    defense_functions = {name: getattr(defense, name) for name in defense_function_list}

    if config.third_party.function in defense_functions:
        defense_functions[config.third_party.function]()
    else:
        sys.exit(f"⚠️ Oops! That function doesn't exist.")


if __name__ == "__main__":
    main()
