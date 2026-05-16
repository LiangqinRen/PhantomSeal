from __future__ import annotations

import os
import sys
import logging
import textwrap
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

from src.evaluate import Utility


class Defense:
    """Create perturbations for aligned faces using upstream faceswap internals.

    This deliberately operates on extracted/aligned face images, not full frames. The upstream
    faceswap converter should still be used later to paste faces back onto original frames.
    """

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

        self.faceswap_dir = Path(self.config.third_party.faceswap_dir).resolve()
        self.model_dir = Path(self.config.third_party.defense.model_dir).resolve()
        self.input_dir = Path(self.config.third_party.dataset.input_dir).resolve()
        self.output_dir = Path(self.config.third_party.dataset.output_dir).resolve()
        self.clean_swap_dir = self._optional_path(
            self.config.third_party.dataset.get("clean_swap_dir", None)
        )
        self.protected_swap_dir = self._optional_path(
            self.config.third_party.dataset.get("protected_swap_dir", None)
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._add_faceswap_to_path()
        self.encoder = self._load_encoder()
        self.encoder_input_size = int(self.encoder.input_shape[1])
        self.utility = Utility(logger, config)

    def perturb(self) -> None:
        image_paths = self._get_image_paths(self.input_dir)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.input_dir}")

        batch_size = int(self.config.third_party.defense.batch_size)
        utility_total = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        swap_utility_total = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        swap_utility_batches = 0
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Perturbing"):
            batch_paths = image_paths[start : start + batch_size]
            imgs = self._load_imgs(batch_paths)
            pert_imgs = self._perturb_batch(imgs)
            self._save_imgs(batch_paths, pert_imgs)
            utility = self.utility.calculate_utility(
                self._imgs_to_tensor(imgs),
                self._imgs_to_tensor(pert_imgs),
            )
            for key in utility_total:
                utility_total[key] += float(utility[key])

            swap_utility = self._calculate_swap_utility(batch_paths)
            if swap_utility is not None:
                swap_utility_batches += 1
                for key in swap_utility_total:
                    swap_utility_total[key] += float(swap_utility[key])

            batch_idx = start // batch_size + 1
            total_batches = int(np.ceil(len(image_paths) / batch_size))
            iter_log = textwrap.dedent(
                f"""
                Batch {batch_idx:4}/{total_batches:4} perturbed: {self.output_dir}
                protected image utility: {self._format_utility(utility)}
                summary protected image utility: {self._format_utility_average(utility_total, batch_idx)}
                """
            )
            if swap_utility is not None:
                iter_log += (
                    f"                swap result utility: {self._format_utility(swap_utility)}\n"
                    f"                summary swap result utility: "
                    f"{self._format_utility_average(swap_utility_total, swap_utility_batches)}\n"
                )
            elif self.clean_swap_dir is not None or self.protected_swap_dir is not None:
                iter_log += (
                    "                swap result utility: unavailable "
                    "(missing clean/protected swap pairs for this batch)\n"
                )
            self.logger.info(textwrap.indent(iter_log, "    "))

    @staticmethod
    def _optional_path(value) -> Path | None:
        if value in (None, "", "null"):
            return None
        return Path(value).resolve()

    def _add_faceswap_to_path(self) -> None:
        self._patch_logging_for_faceswap()
        faceswap_path = str(self.faceswap_dir)
        if faceswap_path not in sys.path:
            sys.path.insert(0, faceswap_path)

    @staticmethod
    def _patch_logging_for_faceswap() -> None:
        if not hasattr(logging, "getLevelNamesMapping"):
            logging.getLevelNamesMapping = lambda: logging._nameToLevel.copy()

    def _load_encoder(self) -> tf.keras.Model:
        self._patch_keras_for_faceswap_model()
        from lib.serializer import get_serializer
        from lib.utils import FaceswapError, get_folder
        from plugins.plugin_loader import PluginLoader

        model_dir = get_folder(str(self.model_dir), make_folder=False)
        if not model_dir:
            raise FaceswapError(f"{self.model_dir} does not exist.")

        state_files = [
            fname for fname in os.listdir(model_dir) if fname.endswith("_state.json")
        ]
        if len(state_files) != 1:
            raise FaceswapError(
                f"There should be 1 state file in {model_dir}. Found {len(state_files)}."
            )

        state = get_serializer("json").load(os.path.join(model_dir, state_files[0]))
        trainer = state.get("name", None)
        if not trainer:
            raise FaceswapError("Trainer name could not be read from state file.")

        args = self._get_faceswap_args()
        model = PluginLoader.get_model(trainer)(model_dir, args, predict=False)
        model.build()

        try:
            encoder = model.model.get_layer("encoder")
        except ValueError as exc:
            raise FaceswapError(
                "Could not find an 'encoder' submodel in the loaded faceswap model. "
                "This perturbation path currently supports encoder/decoder models like "
                "'original'."
            ) from exc

        self.logger.info(
            "Loaded upstream faceswap model '%s' from %s; encoder input=%s output=%s",
            trainer,
            model_dir,
            encoder.input_shape,
            encoder.output_shape,
        )
        return encoder

    @staticmethod
    def _patch_keras_for_faceswap_model() -> None:
        dense_cls = tf.keras.layers.Dense
        if getattr(dense_cls, "_phantomseal_quantization_patch", False):
            return

        original_from_config = dense_cls.from_config

        @classmethod
        def from_config(cls, config):
            config = dict(config)
            if config.get("quantization_config", None) is None:
                config.pop("quantization_config", None)
            return original_from_config(config)

        dense_cls.from_config = from_config
        dense_cls._phantomseal_quantization_patch = True

    def _get_faceswap_args(self) -> Namespace:
        defense_cfg = self.config.third_party.defense
        config_file = defense_cfg.get("config_file", None)
        if config_file is not None:
            config_file = str(Path(config_file).resolve())

        return Namespace(
            config_file=config_file,
            no_logs=True,
            summary=False,
            warmup=0,
            freeze_weights=False,
            load_weights=None,
            redirect_gui=False,
        )

    def _perturb_batch(self, imgs: np.ndarray) -> np.ndarray:
        x_orig = tf.convert_to_tensor(imgs, dtype=tf.float32)
        x = tf.clip_by_value(x_orig + 1.0e-6, 0.0, 1.0)
        latent_orig = tf.stop_gradient(self.encoder(self._encoder_input(x_orig), training=False))

        limits = np.array(
            [
                self.config.third_party.defense.limit.B,
                self.config.third_party.defense.limit.G,
                self.config.third_party.defense.limit.R,
            ],
            dtype=np.float32,
        ).reshape((1, 1, 1, 3))
        limits = tf.convert_to_tensor(limits, dtype=tf.float32)

        step_size = float(self.config.third_party.defense.epsilon)
        perturb_weight = float(self.config.third_party.defense.weight.perturb)
        latent_weight = float(self.config.third_party.defense.weight.latent)

        best_x = tf.identity(x)
        best_loss = float("inf")
        best_latent_mse = float("-inf")
        for epoch in range(1, int(self.config.third_party.defense.epochs) + 1):
            with tf.GradientTape() as tape:
                tape.watch(x)
                latent = self.encoder(self._encoder_input(x), training=False)
                perturb_loss = tf.reduce_mean(tf.square(x - x_orig))
                latent_mse = tf.reduce_mean(tf.square(latent - latent_orig))
                loss = perturb_weight * perturb_loss - latent_weight * latent_mse

            grad = tape.gradient(loss, x)
            if grad is None:
                grad = tf.zeros_like(x)
            x = x - step_size * tf.sign(grad)
            x = tf.clip_by_value(x, x_orig - limits, x_orig + limits)
            x = tf.clip_by_value(x, 0.0, 1.0)

            latent_mse_value = float(latent_mse.numpy())
            loss_value = float(loss.numpy())
            best_latent_mse = max(best_latent_mse, latent_mse_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best_x = tf.identity(x)

            if epoch % int(self.config.third_party.defense.log_interval) == 0:
                delta = x - x_orig
                perturb_mse = float(tf.reduce_mean(tf.square(delta)).numpy())
                perturb_mse_255 = perturb_mse * 255.0 * 255.0
                rmse_255 = float(np.sqrt(perturb_mse_255))
                max_abs_255 = float(tf.reduce_max(tf.abs(delta)).numpy() * 255.0)
                grad_abs = tf.abs(grad)
                grad_mean = float(tf.reduce_mean(grad_abs).numpy())
                grad_max = float(tf.reduce_max(grad_abs).numpy())
                self.logger.info(
                    "[%4d/%4d] loss=%.6f best_loss=%.6f perturb_mse=%.6f perturb_mse_255=%.3f latent_mse=%.6f best_latent_mse=%.6f rmse_255=%.3f max_abs_255=%.3f grad_mean=%.3e grad_max=%.3e",
                    epoch,
                    int(self.config.third_party.defense.epochs),
                    loss_value,
                    best_loss,
                    perturb_mse,
                    perturb_mse_255,
                    latent_mse_value,
                    best_latent_mse,
                    rmse_255,
                    max_abs_255,
                    grad_mean,
                    grad_max,
                )

        return best_x.numpy()

    def _encoder_input(self, imgs: tf.Tensor) -> tf.Tensor:
        """Resize loaded aligned faces to the trained faceswap model input size."""
        return tf.image.resize(
            imgs,
            (self.encoder_input_size, self.encoder_input_size),
            method="bilinear",
        )

    def _load_imgs(self, image_paths: list[Path]) -> np.ndarray:
        imgs = []
        height = int(self.config.third_party.dataset.image_size)
        width = int(self.config.third_party.dataset.image_size)
        for image_path in image_paths:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            imgs.append(img.astype("float32") / 255.0)
        return np.stack(imgs, axis=0)

    def _save_imgs(self, image_paths: list[Path], imgs: np.ndarray) -> None:
        for source_path, img in zip(image_paths, imgs):
            relative_path = source_path.relative_to(self.input_dir)
            output_path = self.output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img_uint8 = np.clip(img * 255.0, 0, 255).astype("uint8")
            cv2.imwrite(str(output_path), img_uint8)

    def _calculate_swap_utility(self, image_paths: list[Path]) -> dict | None:
        if self.clean_swap_dir is None or self.protected_swap_dir is None:
            return None

        clean_swap_imgs = []
        protected_swap_imgs = []
        for source_path in image_paths:
            relative_path = source_path.relative_to(self.input_dir)
            clean_swap_path = self.clean_swap_dir / relative_path
            protected_swap_path = self.protected_swap_dir / relative_path
            if not clean_swap_path.exists() or not protected_swap_path.exists():
                continue

            clean_swap = cv2.imread(str(clean_swap_path), cv2.IMREAD_COLOR)
            protected_swap = cv2.imread(str(protected_swap_path), cv2.IMREAD_COLOR)
            if clean_swap is None or protected_swap is None:
                continue
            if clean_swap.shape[:2] != protected_swap.shape[:2]:
                protected_swap = cv2.resize(
                    protected_swap,
                    (clean_swap.shape[1], clean_swap.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            clean_swap_imgs.append(clean_swap.astype("float32") / 255.0)
            protected_swap_imgs.append(protected_swap.astype("float32") / 255.0)

        if not clean_swap_imgs:
            return None

        return self.utility.calculate_utility(
            self._imgs_to_tensor(np.stack(clean_swap_imgs, axis=0)),
            self._imgs_to_tensor(np.stack(protected_swap_imgs, axis=0)),
        )

    @staticmethod
    def _imgs_to_tensor(imgs: np.ndarray) -> torch.Tensor:
        rgb = imgs[..., ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().clamp(0, 1)
        return tensor.cuda()

    @staticmethod
    def _format_utility(utility: dict) -> str:
        return (
            f"mse_255={float(utility['mse']):.3f}, "
            f"psnr={float(utility['psnr']):.3f}, "
            f"ssim={float(utility['ssim']):.3f}, "
            f"lpips={float(utility['lpips']):.3f}"
        )

    def _format_utility_average(self, total: dict, count: int) -> str:
        if count <= 0:
            return "n/a"
        return self._format_utility({key: total[key] / count for key in total})

    @staticmethod
    def _get_image_paths(input_dir: Path) -> list[Path]:
        suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
        return sorted(
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        )
