import utils
from hififace.defense import Defense

import inspect
import hydra
import sys
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    logger = utils.get_customized_logger(config.log.record_level)

    utils.check_cuda_availability(logger)
    utils.fix_random_seed(logger, config.random_seed)
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

    defense = Defense(logger, config)
    defense_function_list = ["metric"]
    defense_functions = {name: getattr(defense, name) for name in defense_function_list}

    if config.third_party.function in defense_functions:
        defense_functions[config.third_party.function]()
    else:
        sys.exit(f"⚠️ Oops! Fail to find {config.third_party.function}")


if __name__ == "__main__":
    main()
