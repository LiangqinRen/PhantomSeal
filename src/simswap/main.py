from src import common_utils
from src.simswap.defense import Defense

import sys
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    logger = common_utils.get_customized_logger(config.log.record_level)

    common_utils.check_cuda_availability(logger)
    common_utils.fix_random_seed(logger, config.random_seed)
    timer = common_utils.Timer(f"SimSwap {config.third_party.function}", logger)

    defense = Defense(logger, config)
    defense_functions = {
        "sample": defense.sample,
        "metric": defense.metric,
        "ai_beauty": defense.metric,
        "protection_robustness_sample": defense.protection_robustness_sample,
        "protection_robustness_metric": defense.protection_robustness_metric,
        "forensics_robustness_sample": defense.forensics_robustness_sample,
        "forensics_robustness_metric": defense.forensics_robustness_metric,
        "image_robustness_metric": defense.image_robustness_metric,
        "adaptive_attack_with_self_image": defense.adaptive_attack_with_self_image,
        "adaptive_attack_with_other_image": defense.adaptive_attack_with_other_image,
    }

    if config.third_party.function in defense_functions:
        defense_functions[config.third_party.function]()
    else:
        sys.exit(f"⚠️ Oops! Fail to find {config.third_party.function}")


if __name__ == "__main__":
    main()
