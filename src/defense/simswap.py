from utils import Timer, get_customized_logger

import inspect
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    logger = get_customized_logger()
    logger.info(config.third_party.slogan)

    timer = Timer(inspect.currentframe().f_code.co_name, logger)


if __name__ == "__main__":
    main()
