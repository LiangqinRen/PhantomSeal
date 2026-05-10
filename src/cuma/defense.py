from argparse import Namespace
from pathlib import Path
from types import MethodType
from typing import Any, Callable, cast

import inspect
import random
import textwrap
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from src import metric
from src.common_utils import cd, save_tensor_imgs, use_project
from src.evaluate import Effectiveness, Utility

try:
    from torch.serialization import add_safe_globals
except ImportError:
    def add_safe_globals(_globals):
        return None


class CumaMetricDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root_dir = Path(
            config.third_party.dataset.metric_224_dir
            if config.third_party.dataset.use_224
            else config.third_party.dataset.metric_512_dir
        )
        image_size = 224 if config.third_party.dataset.use_224 else 256
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.images = sorted(path for path in self.root_dir.iterdir() if path.is_file())
        self.index_pairs = self._get_random_pairs()

    def _get_random_pairs(self) -> list[tuple[int, int]]:
        metric_pairs = self.config.third_party.dataset.metric_pairs
        image_count = len(self.images)
        index_pairs = []
        for _ in range(metric_pairs):
            i = random.randrange(image_count)
            j = random.randrange(image_count)
            while j == i:
                j = random.randrange(image_count)
            index_pairs.append((i, j))
        return index_pairs

    def __len__(self):
        return self.config.third_party.dataset.metric_pairs

    def __getitem__(self, idx):
        idx_a, idx_b = self.index_pairs[idx]
        img_A = self.transform(Image.open(self.images[idx_a]).convert("RGB"))
        img_B = self.transform(Image.open(self.images[idx_b]).convert("RGB"))
        return img_A, img_B


class Defense:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        notes_path = Path(self.config.notes_path)
        notes_path.touch(exist_ok=True)

        self.perturbation = self._load_perturbation(
            Path(self.config.third_party.defense.perturbation_path)
        )
        self.target = None
        self.transformer_Arcface = transforms.Compose(
            [
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.utility = None
        self.effectiveness = None

    @staticmethod
    def _load_perturbation(path: Path) -> Tensor:
        perturbation = torch.load(path, map_location="cpu")
        if not isinstance(perturbation, torch.Tensor):
            raise TypeError(f"{path} did not contain a torch.Tensor")

        if perturbation.ndim == 3:
            perturbation = perturbation.unsqueeze(0)
        if perturbation.ndim != 4 or perturbation.shape[1] != 3:
            raise ValueError(
                "Expected CUMA perturbation shape [3,H,W] or [1,3,H,W], "
                f"got {tuple(perturbation.shape)}"
            )

        return perturbation[:1].float()

    def _apply_cuma(self, imgs: Tensor) -> Tensor:
        perturbation = self.perturbation.to(device=imgs.device, dtype=imgs.dtype)
        if perturbation.shape[-2:] != imgs.shape[-2:]:
            perturbation = F.interpolate(
                perturbation,
                size=imgs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # CUMA stores perturbations for [-1, 1] normalized tensors. The SimSwap
        # pipeline uses [0, 1], so the equivalent pixel-space delta is half.
        pixel_delta = perturbation / 2.0
        pixel_delta = pixel_delta * self.config.third_party.defense.perturbation_scale
        return torch.clamp(imgs + pixel_delta, 0.0, 1.0)

    def _perturb_imgs(self, imgs: Tensor, cloak_imgs: Tensor | None = None) -> Tensor:
        return self._apply_cuma(imgs)

    def _init_simswap(self) -> None:
        if self.target is not None:
            return

        root_dir = Path(self.config.third_party.project_root)
        with use_project([root_dir]), cd(root_dir):
            from models.models import create_model
            from models import arcface_models

            test_options = Namespace(
                gpu_ids=[0],
                isTrain=False,
                checkpoints_dir="checkpoints",
                name="people",
                resize_or_crop="scale_width",
                crop_size=224,
                Arc_path="arcface_model/arcface_checkpoint.tar",
                which_epoch="latest",
                verbose=False,
            )

            add_safe_globals(
                [
                    nn.Conv2d,
                    nn.Linear,
                    nn.BatchNorm2d,
                    nn.BatchNorm1d,
                    nn.ReLU,
                    nn.PReLU,
                    nn.Sigmoid,
                    nn.Dropout,
                    nn.Sequential,
                    nn.MaxPool2d,
                    nn.AdaptiveAvgPool2d,
                ]
            )

            for _, obj in inspect.getmembers(arcface_models):
                if inspect.isfunction(obj):
                    add_safe_globals([obj])
                elif inspect.isclass(obj):
                    add_safe_globals([cast(Callable[..., Any], obj)])

            self.target = create_model(test_options)
            self.target.cuda().eval()

        netG = cast(nn.Module, self.target.netG)
        setattr(netG, "encoder", MethodType(self.encoder, netG))

    def _init_evaluators(self) -> None:
        if self.utility is None:
            self.utility = Utility(self.logger, self.config)
        if self.effectiveness is None:
            self.effectiveness = Effectiveness(self.logger, self.config)

    def _get_imgs_identity(self, imgs: Tensor) -> Tensor:
        self._init_simswap()
        imgs = self.transformer_Arcface(imgs)
        imgs_downsample = F.interpolate(imgs, size=(112, 112))
        netArc = cast(nn.Module, self.target.netArc)
        prior = netArc(imgs_downsample)
        prior = F.normalize(prior, p=2, dim=1)
        return prior.cuda()

    def _get_full_swap_results(
        self, imgs_A: Tensor, imgs_B: Tensor, pert_imgs_A: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self._init_simswap()

        imgs_A_identity = self._get_imgs_identity(imgs_A)
        source_swap = self.target(None, imgs_B, imgs_A_identity, None, True)

        pert_imgs_A_identity = self._get_imgs_identity(pert_imgs_A)
        pert_source_swap = self.target(None, imgs_B, pert_imgs_A_identity, None, True)

        imgs_B_identity = self._get_imgs_identity(imgs_B)
        target_swap = self.target(None, imgs_A, imgs_B_identity, None, True)
        pert_target_swap = self.target(None, pert_imgs_A, imgs_B_identity, None, True)

        return (
            torch.clamp(source_swap, 0.0, 1.0),
            torch.clamp(pert_source_swap, 0.0, 1.0),
            torch.clamp(target_swap, 0.0, 1.0),
            torch.clamp(pert_target_swap, 0.0, 1.0),
        )

    def _get_context_swap_results(
        self, imgs_A: Tensor, imgs_B: Tensor, pert_imgs_A: Tensor
    ) -> tuple[Tensor, Tensor]:
        self._init_simswap()

        imgs_B_identity = self._get_imgs_identity(imgs_B)
        target_swap = self.target(None, imgs_A, imgs_B_identity, None, True)
        pert_target_swap = self.target(None, pert_imgs_A, imgs_B_identity, None, True)

        return (
            torch.clamp(target_swap, 0.0, 1.0),
            torch.clamp(pert_target_swap, 0.0, 1.0),
        )

    @staticmethod
    def encoder(this, input: Tensor) -> Tensor:
        x = input
        x = this.first_layer(x)
        x = this.down1(x)
        x = this.down2(x)
        x = this.down3(x)
        if this.deep:
            x = this.down4(x)
        return x

    @staticmethod
    def _get_context_metric_data_template(effectiveness) -> dict:
        data = {
            "utility": (0, 0, 0, 0),
            "pert_target_utility": (0, 0, 0, 0),
            "pert_target_effectiveness": {},
        }

        for function in effectiveness.candi_funcs.keys():
            data["pert_target_effectiveness"][function] = {}

        return data

    @staticmethod
    def _merge_tuple_metric(prev: tuple, item: dict) -> tuple:
        return tuple(
            x + y
            for x, y in zip(
                prev,
                (
                    item["mse"],
                    item["psnr"],
                    item["ssim"],
                    item["lpips"],
                ),
            )
        )

    @staticmethod
    def _merge_context_effectiveness(metrics: dict, target_effectiveness: dict) -> None:
        for effec, values in target_effectiveness.items():
            for name, value in values.items():
                prev = metrics["pert_target_effectiveness"][effec].get(name, (0, 0))
                metrics["pert_target_effectiveness"][effec][name] = (
                    prev[0] + value[0],
                    prev[1] + value[1],
                )

    def metric(self) -> None:
        self._init_simswap()
        self._init_evaluators()
        metrics = self._get_context_metric_data_template(self.effectiveness)

        dataset = CumaMetricDataset(self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.third_party.defense.batch_size,
            shuffle=True,
        )

        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            with torch.no_grad():
                imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
                total_count += len(imgs_A)

                pert_imgs = self._perturb_imgs(imgs_A, None)
                target_swap, pert_target_swap = self._get_context_swap_results(
                    imgs_A, imgs_B, pert_imgs
                )

                utility = self.utility.calculate_utility(imgs_A, pert_imgs)
                target_utility = self.utility.calculate_utility(
                    target_swap, pert_target_swap
                )
                target_effectiveness = self.effectiveness.calculate_effectiveness(
                    imgs_B,
                    None,
                    target_swap,
                    pert_target_swap,
                    None,
                )

                metrics["utility"] = self._merge_tuple_metric(
                    metrics["utility"], utility
                )
                metrics["pert_target_utility"] = self._merge_tuple_metric(
                    metrics["pert_target_utility"], target_utility
                )
                self._merge_context_effectiveness(metrics, target_effectiveness)

                save_tensor_imgs(
                    self.image_dir,
                    idx,
                    [
                        "imgs_A",
                        "imgs_B",
                        "cuma_imgs_A",
                        "target_swap",
                        "cuma_target_swap",
                    ],
                    [
                        imgs_A,
                        imgs_B,
                        pert_imgs,
                        target_swap,
                        pert_target_swap,
                    ],
                    only_save_summary=self.config.third_party.defense.only_save_summary,
                )

            iter_log_str = textwrap.dedent(
                f"""
            utility (mse, psnr, ssim, lpips), effectiveness ({', '.join(self.effectiveness.candi_funcs.keys())})
            perturbation scale: {self.config.third_party.defense.perturbation_scale:.3f}
            utility: {metric.generate_iter_utility_log(utility)}
            target utility: {metric.generate_iter_utility_log(target_utility)}
            target effectiveness (clean target swap ASR, CUMA target swap ASR): {metric.generate_iter_effectiveness_log(target_effectiveness)}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            utility: {metric.generate_summary_utility_log(metrics, 'utility', idx)}
            target utility: {metric.generate_summary_utility_log(metrics, 'pert_target_utility', idx)}
            target effectiveness (clean target swap ASR, CUMA target swap ASR): {metric.generate_summary_effectiveness_log(metrics, 'pert_target_effectiveness')}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

            del (
                imgs_A,
                imgs_B,
                pert_imgs,
                target_swap,
                pert_target_swap,
            )
            self._free_gpu()

    @staticmethod
    def _free_gpu() -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
