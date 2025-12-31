from src.utils import cd, use_project
from src.artificialfingerprint.base import Base
from src.simswap.robustness import Robustness
from src import metric
from src.dataset import MetricDataset
from src.utils import save_tensor_imgs
from src.evaluate import Effectiveness

import textwrap
import torch
import copy
from torch import tensor, Tensor
from torch.utils.data import DataLoader
from pathlib import Path
from typing import cast


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.effectiveness = Effectiveness(logger, config)

        simswap_config = copy.deepcopy(config)
        simswap_config.third_party.project_root = config.third_party.simswap_root
        self.robustness = Robustness(logger, simswap_config)

    def forensics_robustness_metric(self) -> None:
        metrics = metric.get_robustness_metric_data_template(
            self.config, self.effectiveness
        )

        logo = self.robustness.load_logo()
        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)

            fingerprinted_imgs_A = imgs_A
            break
            # (
            #     noise_source_effectivenesses,
            #     noise_target_effectivenesses,
            # ) = self.robustness.get_gauss_noise_metrics(
            #     idx, imgs_A, imgs_B, pert_imgs, cloak_imgs, self.image_dir
            # )
            # metric.merge_single_robustness_metric(
            #     metrics,
            #     noise_source_effectivenesses,
            #     noise_target_effectivenesses,
            #     "noise",
            # )

            # (
            #     compress_source_effectivenesses,
            #     compress_target_effectivenesses,
            # ) = self.robustness.get_compress_metrics(
            #     idx, imgs_A, imgs_B, pert_imgs, cloak_imgs, self.image_dir
            # )
            # metric.merge_single_robustness_metric(
            #     metrics,
            #     compress_source_effectivenesses,
            #     compress_target_effectivenesses,
            #     "compress",
            # )

            # (
            #     crop_source_effectivenesses,
            #     crop_target_effectivenesses,
            # ) = self.robustness.get_crop_metrics(
            #     idx, imgs_A, imgs_B, pert_imgs, cloak_imgs, self.image_dir
            # )
            # metric.merge_single_robustness_metric(
            #     metrics,
            #     crop_source_effectivenesses,
            #     crop_target_effectivenesses,
            #     "crop",
            # )

            # (
            #     logo_source_effectivenesses,
            #     logo_target_effectivenesses,
            # ) = self.robustness.get_logo_metrics(
            #     idx, imgs_A, imgs_B, pert_imgs, logo, cloak_imgs, self.image_dir
            # )
            # metric.merge_single_robustness_metric(
            #     metrics,
            #     logo_source_effectivenesses,
            #     logo_target_effectivenesses,
            #     "logo",
            # )

            # (
            #     brighten_source_effectivenesses,
            #     brighten_target_effectivenesses,
            # ) = self.robustness.get_brightness_metrics(
            #     idx, imgs_A, imgs_B, pert_imgs, 1.25, cloak_imgs, self.image_dir
            # )
            # metric.merge_single_robustness_metric(
            #     metrics,
            #     brighten_source_effectivenesses,
            #     brighten_target_effectivenesses,
            #     "brighten",
            # )

            # (
            #     darken_source_effectivenesses,
            #     darken_target_effectivenesses,
            # ) = self.robustness.get_brightness_metrics(
            #     idx, imgs_A, imgs_B, pert_imgs, 0.75, cloak_imgs, self.image_dir
            # )
            # metric.merge_single_robustness_metric(
            #     metrics,
            #     darken_source_effectivenesses,
            #     darken_target_effectivenesses,
            #     "darken",
            # )

            # self._free_gpu()

            # iter_log_str = textwrap.dedent(
            #     f"""
            # utility (mse, psnr, ssim, lpips), effectiveness ({', '.join(self.effectiveness.candi_funcs.keys())}), identity ({', '.join(next(iter(noise_source_effectivenesses.values())).keys())}), context ({', '.join(next(iter(noise_source_effectivenesses.values())).keys())})
            # utility: {metric.generate_iter_utility_log(utility)}
            # noise: {metric.generate_iter_robustness_log(noise_source_effectivenesses,noise_target_effectivenesses)}
            # compress: {metric.generate_iter_robustness_log(compress_source_effectivenesses,compress_target_effectivenesses)}
            # crop: {metric.generate_iter_robustness_log(crop_source_effectivenesses,crop_target_effectivenesses)}
            # logo: {metric.generate_iter_robustness_log(logo_source_effectivenesses,logo_target_effectivenesses)}
            # brighten: {metric.generate_iter_robustness_log(brighten_source_effectivenesses,brighten_target_effectivenesses)}
            # darken: {metric.generate_iter_robustness_log(darken_source_effectivenesses,darken_target_effectivenesses)}
            # """
            # )

            # summary_log_str = textwrap.dedent(
            #     f"""
            # Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            # utility: {metric.generate_summary_robustness_utility_log(metrics['utility'], idx)}
            # noise: {metric.generate_summary_robustness_log(metrics['noise'])}
            # compress: {metric.generate_summary_robustness_log(metrics['compress'])}
            # crop: {metric.generate_summary_robustness_log(metrics['crop'])}
            # logo: {metric.generate_summary_robustness_log(metrics['logo'])}
            # brighten: {metric.generate_summary_robustness_log(metrics['brighten'])}
            # darken: {metric.generate_summary_robustness_log(metrics['darken'])}
            # """
            # )

            # self.logger.info(textwrap.indent(iter_log_str, "    "))
            # self.logger.info(textwrap.indent(summary_log_str, "    "))
