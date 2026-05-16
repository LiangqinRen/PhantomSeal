import logging
import time
import torch
import random
import os
import shutil
import sys
import warnings
import numpy as np
from torch import Tensor
from torchvision.utils import save_image, make_grid
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, ImageFont
from pathlib import Path
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Iterable, Iterator, TypeAlias


FontLike: TypeAlias = ImageFont.ImageFont | ImageFont.FreeTypeFont
_SOURCE_SNAPSHOT_DONE = False


class Timer:
    def __init__(self, function_name: str, logger: logging.Logger):
        self.function_name = function_name
        self.begin_time = time.time()
        self.logger = logger

    def __del__(self) -> None:
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

    snapshot_source_tree(logger)

    return logger


def snapshot_source_tree(logger: logging.Logger) -> None:
    global _SOURCE_SNAPSHOT_DONE
    if _SOURCE_SNAPSHOT_DONE:
        return

    try:
        from hydra.core.hydra_config import HydraConfig

        runtime = HydraConfig.get().runtime
        project_root = Path(runtime.cwd)
        log_dir = Path(runtime.output_dir)
    except Exception as exc:
        logger.debug("Skip source snapshot: Hydra runtime is unavailable: %s", exc)
        return

    source_dir = project_root / "src"
    snapshot_dir = log_dir / "src"
    if not source_dir.is_dir() or snapshot_dir.exists():
        _SOURCE_SNAPSHOT_DONE = True
        return

    try:
        shutil.copytree(
            source_dir,
            snapshot_dir,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                "*.pyo",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
            ),
        )
        _SOURCE_SNAPSHOT_DONE = True
        logger.info("Snapshot current src to %s", snapshot_dir)
    except Exception as exc:
        logger.warning("Failed to snapshot current src to %s: %s", snapshot_dir, exc)


def check_cuda_availability(logger: logging.Logger) -> None:
    if torch.cuda.is_available():
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        logger.info(f"Use GPU {torch.cuda.get_device_name()}")
    else:
        raise SystemExit("CUDA is not available!")


def fix_random_seed(logger: logging.Logger, random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    logger.info(f"Fix random, numpy and torch random seed to {random_seed}")


def save_tensor_imgs(
    image_dir: Path,
    idx: int | str | None,
    image_labels: list[str],
    image_tensors: list[Tensor],
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

    save_image_tensors: list[Tensor] = []
    for image in image_tensors:
        if image.min() < -1.05 or image.max() > 1.05:
            raise ValueError("Tensor value out of expected [-1, 1] range")

        if image.min() < -0.5:
            image_to_save = (image + 1) / 2
        else:
            image_to_save = image

        save_image_tensors.append(image_to_save)

    index_row = prepend_white_row(save_image_tensors[0])
    summary_imgs = torch.cat(
        [prepend_white_column(img) for img in save_image_tensors],
        dim=0,
    )
    summary_imgs = torch.cat([index_row, summary_imgs], dim=0)

    nrow = save_image_tensors[0].shape[0] + 1
    padding = 2
    grid = make_grid(summary_imgs, nrow=nrow, padding=padding)
    grid = to_pil_image(torch.clamp(grid, 0, 1))

    draw = ImageDraw.Draw(grid)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    _, _, cell_h, cell_w = summary_imgs.shape

    def get_cell_box(row: int, col: int) -> tuple[int, int, int, int]:
        x0 = padding + col * (cell_w + padding)
        y0 = padding + row * (cell_h + padding)
        return x0, y0, cell_w, cell_h

    def get_text_bbox(
        text: str, font: FontLike, spacing: int
    ) -> tuple[float, float, float, float]:
        if "\n" in text:
            return draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
        return draw.textbbox((0, 0), text, font=font)

    def fit_font(
        text: str,
        box_w: int,
        box_h: int,
        target_ratio: float,
        min_size: int = 10,
    ) -> tuple[FontLike, int]:
        max_size = max(min_size, int(min(box_w, box_h) * target_ratio))
        for size in range(max_size, min_size - 1, -1):
            spacing = max(2, size // 6)
            try:
                font = ImageFont.truetype(font_path, size)
            except OSError:
                return ImageFont.load_default(), spacing

            bbox = get_text_bbox(text, font, spacing)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            if text_w <= box_w * 0.82 and text_h <= box_h * 0.82:
                return font, spacing

        return ImageFont.load_default(), max(2, min_size // 6)

    def draw_center_text(row: int, col: int, text: str, target_ratio: float) -> None:
        x0, y0, w, h = get_cell_box(row, col)
        font, spacing = fit_font(text, w, h, target_ratio)
        bbox = get_text_bbox(text, font, spacing)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x0 + (w - text_w) / 2 - bbox[0]
        text_y = y0 + (h - text_h) / 2 - bbox[1]

        if "\n" in text:
            draw.multiline_text(
                (text_x, text_y),
                text,
                fill="black",
                font=font,
                spacing=spacing,
                align="center",
            )
        else:
            draw.text((text_x, text_y), text, fill="black", font=font)

    for i in range(1, index_row.shape[0]):
        draw_center_text(0, i, str(i), target_ratio=0.55)

    for i, label in enumerate(image_labels, start=1):
        label = label.replace("_", "\n")
        draw_center_text(i, 0, label, target_ratio=0.32)

    if idx is None or (isinstance(idx, str) and len(idx) == 0):
        grid.save(image_dir / f"{image_name}.png")
    else:
        grid.save(image_dir / f"{image_name}_{idx}.png")

    if not only_save_summary:
        for label, imgs in zip(image_labels, save_image_tensors):
            for i, img in enumerate(imgs, start=1):
                if idx is None or (isinstance(idx, str) and len(idx) == 0):
                    save_image(img, image_dir / f"{label}_{i}.png")
                else:
                    save_image(img, image_dir / f"{label}_{idx}_{i}.png")


def check_tensor_info(logger: logging.Logger, x: Tensor, name: str = "tensor") -> None:
    logger.info(
        f"{name}: "
        f"shape={tuple(x.shape)}, "
        f"dtype={x.dtype}, "
        f"device={x.device}, "
        f"requires_grad={x.requires_grad}, "
        f"is_contiguous={x.is_contiguous()}, "
        f"max={x.max().item()}, "
        f"min={x.min().item()}"
    )


@contextmanager
def cd(path: str | os.PathLike[str]) -> Iterator[None]:
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _infer_prefixes_from_project_roots(
    project_paths: Iterable[Path],
) -> tuple[str, ...]:
    prefixes: set[str] = set()

    for root in project_paths:
        root = root.resolve()
        if not root.exists() or not root.is_dir():
            continue

        for p in root.iterdir():
            name = p.name

            if name.startswith(".") or name in {"__pycache__", "build", "dist"}:
                continue

            # top-level module: xxx.py
            if p.is_file() and p.suffix == ".py":
                prefixes.add(p.stem)
                continue

            # top-level package: xxx/__init__.py
            if p.is_dir() and (p / "__init__.py").exists():
                prefixes.add(p.name)
                continue

            # top-level namespace-style package: xxx/*.py without __init__.py
            if p.is_dir() and any(child.suffix == ".py" for child in p.iterdir()):
                prefixes.add(p.name)
                continue

    return tuple(sorted(prefixes))


@contextmanager
def use_project(
    project_paths: list[Path], purge_prefixes: Iterable[str] | None = None
) -> Iterator[None]:
    abs_paths = [str(p.resolve()) for p in project_paths]
    old_sys_path = list(sys.path)
    for path in reversed(abs_paths):
        if path not in sys.path:
            sys.path.insert(0, path)

    if purge_prefixes is None:
        purge_prefixes = _infer_prefixes_from_project_roots(project_paths)
    else:
        purge_prefixes = tuple(purge_prefixes)

    for name in list(sys.modules.keys()):
        if any(name == p or name.startswith(p + ".") for p in purge_prefixes):
            sys.modules.pop(name, None)

    try:
        yield
    finally:
        sys.path[:] = old_sys_path


@contextmanager
def suppress_third_party_noise(enabled: bool = True) -> Iterator[None]:
    if not enabled:
        yield
        return

    devnull = open(os.devnull, "w")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Default grid_sample and affine_grid behavior has changed.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="`torch.cuda.amp.autocast\\(args\\.\\.\\.\\)` is deprecated.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*GPU device discovery failed.*",
                category=UserWarning,
            )
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
    finally:
        devnull.close()
