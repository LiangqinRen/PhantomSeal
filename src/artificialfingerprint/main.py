import utils
from artificialfingerprint.base import Base

import sys
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    logger = utils.get_customized_logger(config.log.record_level)

    utils.check_cuda_availability(logger)
    utils.fix_random_seed(logger, config.random_seed)
    timer = utils.Timer("main", logger)

    base = Base(logger, config)
    base_function_list = ["robust"]
    base_functions = {name: getattr(base, name) for name in base_function_list}

    if config.third_party.function in base_functions:
        base_functions[config.third_party.function]()
    else:
        sys.exit(f"⚠️ Oops! Fail to find {config.third_party.function}")


if __name__ == "__main__":
    main()
