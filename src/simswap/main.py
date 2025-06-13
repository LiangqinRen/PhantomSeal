import utils
from simswap.defense import Defense

import inspect
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    logger = utils.get_customized_logger()

    utils.check_cuda_availability(logger)
    utils.fix_random_seed(logger, config.random_seed)
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

    defense = Defense(logger, config)
    defense_function_list = ["sample", "metric"]
    defense_functions = {name: getattr(defense, name) for name in defense_function_list}

    if config.third_party.function in defense_functions:
        defense_functions[config.third_party.function]()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
