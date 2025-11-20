import logging
import time
import torch
import random
import os
import sys
import numpy as np
from torch import Tensor
from torchvision.utils import save_image, make_grid
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, ImageFont
from pathlib import Path
from contextlib import contextmanager


class Timer:
    def __init__(self, function_name, logger):
        self.function_name = function_name
        self.begin_time = time.time()
        self.logger = logger

    def __del__(self):
        elapsed = time.time() - self.begin_time
        self.logger.info(
            f"{self.function_name} took {self.format_seconds(elapsed)} ({elapsed:.3f} seconds)"
        )

    def format_seconds(self, seconds: float) -> str:
        seconds = round(seconds, 3)
        days, rem = divmod(int(seconds), 86400)
        hours, rem = divmod(rem, 3600)
        minutes, whole_seconds = divmod(rem, 60)
        fractional = round(seconds - int(seconds), 3)
        final_seconds = whole_seconds + fractional

        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if final_seconds or not parts:
            parts.append(f"{final_seconds:.3f}s")
        return " ".join(parts)


def get_customized_logger(log_level: str) -> logging.Logger:
    level_str = log_level.upper()
    if not hasattr(logging, level_str):
        raise ValueError(f"Invalid log level: {level_str}")

    level = getattr(logging, level_str)
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d][%(filename)10s:%(lineno)4d][%(levelname)5s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    return logger


def check_cuda_availability(logger):
    if torch.cuda.is_available():
        logger.info(f"Use GPU {torch.cuda.get_device_name()}")
    else:
        raise SystemExit("CUDA is not available!")


def fix_random_seed(logger, random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    logger.info(f"Fix random, numpy and torch random seed to {random_seed}")


def save_tensor_imgs(
    image_dir: Path,
    idx: int,
    img_labels: list[str],
    img_tensors: list[Tensor],
    image_name: str = "summary",
    only_save_summary: bool = True,
) -> None:
    def prepend_white_row(imgs: Tensor) -> Tensor:
        B, C, H, W = imgs.shape
        white_img = torch.ones((B + 1, C, H, W), device=imgs.device, dtype=imgs.dtype)
        return white_img

    def prepend_white_column(imgs: Tensor) -> Tensor:
        _, C, H, W = imgs.shape
        white_img = torch.ones((1, C, H, W), device=imgs.device, dtype=imgs.dtype)
        return torch.cat([white_img, imgs], dim=0)

    index_row = prepend_white_row(img_tensors[0])
    summary_imgs = torch.cat(
        [prepend_white_column(img) for img in img_tensors],
        dim=0,
    )
    summary_imgs = torch.cat([index_row, summary_imgs], dim=0)

    nrow = len(img_tensors[0]) + 1
    grid = make_grid(summary_imgs, nrow=nrow, padding=2)
    grid = to_pil_image(torch.clamp(grid, 0, 1))

    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default().font_variant(size=40)

    cell_w = cell_h = grid.height // (len(summary_imgs) // nrow)
    for i in range(1, index_row.shape[0]):
        x = i * cell_w
        draw.text((x + 5, 5), str(i), fill="black", font=font)
    for i, label in enumerate(img_labels, start=1):  # skip the first line
        y = i * cell_h
        draw.text((5, y + 5), label, fill="black", font=font)

    grid.save(image_dir / f"{image_name}_{idx}.png")

    if not only_save_summary:
        for label, imgs in zip(img_labels, img_tensors):
            for i, img in enumerate(imgs, start=1):
                title_label = label.replace("\n", "_")
                save_image(img, image_dir / f"{title_label}_{idx}_{i}.png")


@contextmanager
def cd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@contextmanager
def use_project(project_paths: list[Path]):
    abs_paths = [str(p.resolve()) for p in project_paths]
    for path in abs_paths:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        for path in abs_paths:
            if path in sys.path:
                sys.path.remove(path)
