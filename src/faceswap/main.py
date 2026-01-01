from src import utils
from src.faceswap.defense import Defense
from src.faceswap.worker import Worker

import hydra
import sys
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    logger = utils.get_customized_logger(config.log.record_level)

    utils.check_cuda_availability(logger)
    utils.fix_random_seed(logger, config.random_seed)
    timer = utils.Timer("main", logger)

    worker = Worker(logger, config)
    worker_function_list = ["extract", "train", "test"]
    worker_functions = {name: getattr(worker, name) for name in worker_function_list}

    defense = Defense(logger, config)
    defense_function_list = ["metric"]
    defense_functions = {name: getattr(defense, name) for name in defense_function_list}

    if config.third_party.function in worker_functions:
        worker_functions[config.third_party.function]()
    elif config.third_party.function in defense_functions:
        defense_functions[config.third_party.function]()
    else:
        sys.exit(f"⚠️ Oops! Fail to find {config.third_party.function}")


if __name__ == "__main__":
    main()
