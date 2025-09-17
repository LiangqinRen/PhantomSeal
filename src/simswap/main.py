import utils
from simswap.defense import Defense
from simswap.defense_compare import Lowkey

import sys
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    logger = utils.get_customized_logger(config.log.record_level)

    utils.check_cuda_availability(logger)
    utils.fix_random_seed(logger, config.random_seed)
    timer = utils.Timer("main", logger)

    defense = Defense(logger, config)
    lowkey_defense = Lowkey(logger, config)
    defense_functions = {
        "sample": defense.sample,
        "metric": defense.metric,
        "robustness_sample": defense.robustness_sample,
        "robustness_metric": defense.robustness_metric,
        "robustness_forensics_sample": defense.robustness_forensics_sample,
        "robustness_forensics_metric": defense.robustness_forensics_metric,
        "image_robustness_metric": defense.image_robustness_metric,
        "adaptive_attack": defense.adaptive_attack,
        "adaptive_attack_self": defense.adaptive_attack_self,
        "lowkey": lowkey_defense.metric,
    }

    if config.third_party.function in defense_functions:
        defense_functions[config.third_party.function]()
    else:
        sys.exit(f"⚠️ Oops! Fail to find {config.third_party.function}")


if __name__ == "__main__":
    main()
