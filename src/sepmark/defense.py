from src.sepmark.base import Base
from src.simswap.robustness import Robustness
from src.dataset import SepMarkDataset
from src.common_utils import save_tensor_imgs

import copy
import torch
import textwrap
import io
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch import Tensor
from scipy.stats import binom
from typing import cast
from PIL import Image
from torchvision import transforms


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        notes_path = Path(self.config.notes_path)
        notes_path.touch(exist_ok=True)

        simswap_config = copy.deepcopy(config)
        simswap_config.third_party.project_root = config.third_party.simswap_root
        self.robustness = Robustness(logger, simswap_config)

        model = cast(nn.Module, self.network.encoder_decoder.module)
        self.encoder = cast(nn.Module, model.encoder)
        self.decoder_C = cast(nn.Module, model.decoder_C)
        self.decoder_RF = cast(nn.Module, model.decoder_RF)

    def forensics_robustness_metric(self) -> None:
        torch.set_grad_enabled(False)
        metrics = {  # BER, accuracy
            "origin": (0, 0),
            "none": (0, 0),
            "noise": (0, 0),
            "compress": (0, 0),
            "crop": (0, 0),
            "logo": (0, 0),
            "brighten": (0, 0),
            "darken": (0, 0),
        }

        logo = self.robustness.load_logo()
        dataset = SepMarkDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        origin_config = self.config.third_party.origin

        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()

            total_count += len(imgs_A)

            fingerprints = torch.Tensor(
                np.random.choice(
                    [-origin_config.message_range, origin_config.message_range],
                    (len(imgs_A), origin_config.message_length),
                )
            ).to(self.device)

            fingerprinted_imgs_B = self.encoder(imgs_B, fingerprints)

            origin_BER, origin_accuracy = self._calculate_metric(
                fingerprinted_imgs_B, fingerprints
            )
            self._merge_metrics(metrics, "origin", (origin_BER, origin_accuracy))

            none_results = self._simswap_faceswap(imgs_A, fingerprinted_imgs_B)
            none_BER, none_accuracy = self._calculate_metric(none_results, fingerprints)
            self._merge_metrics(metrics, "none", (none_BER, none_accuracy))

            noise_results = self.robustness.gauss_noise(none_results)
            noise_BER, noise_accuracy = self._calculate_metric(
                noise_results, fingerprints
            )
            self._merge_metrics(metrics, "noise", (noise_BER, noise_accuracy))

            compress_results = self.robustness.webp_compress(none_results)
            compress_BER, compress_accuracy = self._calculate_metric(
                compress_results, fingerprints
            )
            self._merge_metrics(metrics, "compress", (compress_BER, compress_accuracy))

            crop_results = self.robustness.crop(none_results)
            crop_BER, crop_accuracy = self._calculate_metric(crop_results, fingerprints)
            self._merge_metrics(metrics, "crop", (crop_BER, crop_accuracy))

            logo_results = self.robustness.logo(none_results, logo)
            logo_BER, logo_accuracy = self._calculate_metric(logo_results, fingerprints)
            self._merge_metrics(metrics, "logo", (logo_BER, logo_accuracy))

            brighten_results = self.robustness.brightness(none_results, 1.25)
            brighten_BER, brighten_accuracy = self._calculate_metric(
                brighten_results, fingerprints
            )
            self._merge_metrics(metrics, "brighten", (brighten_BER, brighten_accuracy))

            darken_results = self.robustness.brightness(none_results, 0.75)
            darken_BER, darken_accuracy = self._calculate_metric(
                darken_results, fingerprints
            )
            self._merge_metrics(metrics, "darken", (darken_BER, darken_accuracy))

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "finger\nprinted\nimgs_B",
                    "none\nresults",
                    "noise\nresults",
                    "compress\nresults",
                    "crop\nresults",
                    "logo\nresults",
                    "brighten\nresults",
                    "darken\nresults",
                ],
                [
                    imgs_A,
                    imgs_B,
                    fingerprinted_imgs_B,
                    none_results,
                    noise_results,
                    compress_results,
                    crop_results,
                    logo_results,
                    brighten_results,
                    darken_results,
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            iter_log_str = textwrap.dedent(
                f"""
            BER (bit error rate) and accuracy
            origin: {origin_BER:.3f}, {origin_accuracy:.3f}
            simswap -> no operation: {none_BER:.3f}, {none_accuracy:.3f}
            simswap -> noise: {noise_BER:.3f}, {noise_accuracy:.3f}
            simswap -> compress: {compress_BER:.3f}, {compress_accuracy:.3f}
            simswap -> crop: {crop_BER:.3f}, {crop_accuracy:.3f}
            simswap -> logo: {logo_BER:.3f}, {logo_accuracy:.3f}
            simswap -> brighten: {brighten_BER:.3f}, {brighten_accuracy:.3f}
            simswap -> darken: {darken_BER:.3f}, {darken_accuracy:.3f}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            origin: {metrics['origin'][0]/idx:.3f}, {metrics['origin'][1]/idx:.3f}
            simswap -> no operation: {metrics['none'][0]/idx:.3f}, {metrics['none'][1]/idx:.3f}
            simswap -> noise: {metrics['noise'][0]/idx:.3f}, {metrics['noise'][1]/idx:.3f}
            simswap -> compress: {metrics['compress'][0]/idx:.3f}, {metrics['compress'][1]/idx:.3f}
            simswap -> crop: {metrics['crop'][0]/idx:.3f}, {metrics['crop'][1]/idx:.3f}
            simswap -> logo: {metrics['logo'][0]/idx:.3f}, {metrics['logo'][1]/idx:.3f}
            simswap -> brighten: {metrics['brighten'][0]/idx:.3f}, {metrics['brighten'][1]/idx:.3f}
            simswap -> darken: {metrics['darken'][0]/idx:.3f}, {metrics['darken'][1]/idx:.3f}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

    def _simswap_faceswap(self, source_imgs: Tensor, target_imgs: Tensor) -> Tensor:
        source_identity = self.robustness._get_imgs_identity(source_imgs)
        results = self.robustness.target(None, target_imgs, source_identity, None, True)

        return results

    def _calculate_metric(
        self, fingerprinted_imgs: Tensor, fingerprints: Tensor
    ) -> tuple[float, float]:
        decoder_output = self.decoder_C(fingerprinted_imgs)
        BER = self.network.decoded_message_error_rate_batch(
            fingerprints, decoder_output
        )
        accuracy = float(
            binom.cdf(
                self.config.third_party.defense.max_wrong_bits,
                self.config.third_party.origin.message_length,
                BER,
            )
        )

        return BER * 100, accuracy * 100

    def _merge_metrics(self, metrics: dict, iter: str, iter_metric: tuple) -> None:
        metrics[iter] = (
            metrics[iter][0] + iter_metric[0],
            metrics[iter][1] + iter_metric[1],
        )
