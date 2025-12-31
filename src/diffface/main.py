from src import utils
from src.diffface.defense import Defense
from src.diffface.patches.gaussian_diffusion import (
    patch_gaussian_diffusion_arcface_load,
)

import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    root_dir = Path(config.third_party.project_root)
    with utils.cd(root_dir), utils.use_project([root_dir]):
        patch_gaussian_diffusion_arcface_load(
            module_path="models.guided_diffusion.gaussian_diffusion"
        )

    logger = utils.get_customized_logger(config.log.record_level)

    utils.check_cuda_availability(logger)
    utils.fix_random_seed(logger, config.random_seed)
    timer = utils.Timer(f"Diffface {config.third_party.function}", logger)

    defense = Defense(logger, config)
    defense_functions = {"sample": defense.sample, "metric": defense.metric}

    if config.third_party.function in defense_functions:
        defense_functions[config.third_party.function]()
    else:
        sys.exit(f"⚠️ Oops! Fail to find {config.third_party.function}")


if __name__ == "__main__":
    main()
