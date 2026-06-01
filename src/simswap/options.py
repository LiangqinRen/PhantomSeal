from argparse import Namespace
from typing import Any


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except (AttributeError, KeyError):
        return default


def build_simswap_test_options(config: Any) -> Namespace:
    origin = _get_attr(config.third_party, "origin")
    simswap_origin = _get_attr(origin, "simswap", origin)

    return Namespace(
        gpu_ids=[0],
        isTrain=False,
        checkpoints_dir=str(_get_attr(simswap_origin, "checkpoints_dir", "checkpoints")),
        name=str(_get_attr(simswap_origin, "name", "people")),
        resize_or_crop="scale_width",
        crop_size=int(_get_attr(simswap_origin, "crop_size", 224)),
        Arc_path=str(
            _get_attr(
                simswap_origin,
                "arcface_path",
                "arcface_model/arcface_checkpoint.tar",
            )
        ),
        which_epoch=str(_get_attr(simswap_origin, "which_epoch", "latest")),
        verbose=False,
    )
