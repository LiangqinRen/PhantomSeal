import utils
from simswap.defense import Defense

import sys
import inspect
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    logger = utils.get_customized_logger(config.log.record_level)

    utils.check_cuda_availability(logger)
    utils.fix_random_seed(logger, config.random_seed)
    timer = utils.Timer("main", logger)

    defense = Defense(logger, config)
    defense_function_list = [
        "sample",
        "metric",
        "robustness_sample",
        "robustness_metric",
        "robustness_forensics_sample",
        "robustness_forensics_metric",
    ]
    defense_functions = {name: getattr(defense, name) for name in defense_function_list}

    if config.third_party.function in defense_functions:
        defense_functions[config.third_party.function]()
    else:
        sys.exit(f"⚠️ Oops! That function doesn't exist")


if __name__ == "__main__":
    main()
