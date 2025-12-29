from src.utils import cd, use_project
from third_party.DiffFace.models.guided_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
)

import os
import torch
import functools
import importlib
import torch.nn as nn
from typing import Any, Callable


def patch_gaussian_diffusion_arcface_load(
    module_path: str,
    new_arcface_ckpt: str = "./checkpoints/Arcface_model_only.tar",
) -> None:
    mod = importlib.import_module(module_path)
    GaussianDiffusion = mod.GaussianDiffusion

    if getattr(GaussianDiffusion.__init__, "_phantomseal_arcface_patched", False):
        return

    old_init: Callable[..., Any] = GaussianDiffusion.__init__

    @functools.wraps(old_init)
    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        old_load = mod.torch.load  # patch only this module's torch.load

        def redirected_load(path: Any, *a: Any, **k: Any):
            base = os.path.basename(str(path))

            # only intercept the original ArcFace checkpoint load
            if base == "Arcface.tar":
                # keep behavior compatible with old code
                k.setdefault("weights_only", False)
                return torch.load(new_arcface_ckpt, *a, **k)

            return old_load(path, *a, **k)

        mod.torch.load = redirected_load
        try:
            old_init(self, *args, **kwargs)
        finally:
            mod.torch.load = old_load  # restore immediately

    setattr(patched_init, "_phantomseal_arcface_patched", True)
    GaussianDiffusion.__init__ = patched_init
