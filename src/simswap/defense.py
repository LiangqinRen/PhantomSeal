import metric
from simswap.base import Base
from dataset import SampleDataset, MetricDataset, AdaptiveMetricDataset
from robustness import Robustness
from utils import save_tensor_imgs


import textwrap
import torch
from torch import tensor, Tensor, nn
from torch.utils.data import DataLoader
from pathlib import Path


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.robustness = Robustness(logger, config)

        if self.config.third_party.robustness.ai_beauty:
            logger.info(
                f"AI Beauty enabled with {self.config.third_party.robustness.ai_beauty_tool}"
            )

    def sample(self) -> None:
        dataset = SampleDataset(self.config.third_party.dataset.sample_dir)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs, silent=False)

            (
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            ) = self._get_full_swap_results(imgs_A, imgs_B, x_imgs)

            (
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            ) = metric.get_defense_metric(
                self.utility,
                self.effectiveness,
                imgs_A,
                imgs_B,
                x_imgs,
                cloak_imgs,
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            )

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "pert_imgs",
                    "cloak_imgs",
                    "swap",
                    "pert_swap",
                    "rev\nswap",
                    "rev\npert_swap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    x_imgs,
                    cloak_imgs,
                    imgs_A_src_swap,
                    pert_imgs_A_src_swap,
                    imgs_A_tgt_swap,
                    pert_imgs_A_tgt_swap,
                ],
                only_save_summary=False,
            )

            self.logger.info(
                textwrap.dedent(
                    f"""
            pert utility(mse, psnr, ssim, lpips): {metric.generate_iter_utility_log(pert_utilities)}
            pert source utility(mse, psnr, ssim, lpips): {metric.generate_iter_utility_log(pert_as_src_swap_utilities)}
            pert target utility(mse, psnr, ssim, lpips): {metric.generate_iter_utility_log(pert_as_tgt_swap_utilities)}
            effectiveness tools: {list(self.effectiveness.candi_funcs.keys())}, source(pert, swap, pert_swap, cloak), target(swap, pert_swap)
            pert as swap source effectiveness: {metric.generate_iter_effectiveness_log(source_effectivenesses)}
            pert as swap target effectiveness: {metric.generate_iter_effectiveness_log(target_effectivenesses)}
            """
                )
            )

    def metric(
        self,
    ) -> None:
        data = metric.get_metric_data_template(self.effectiveness)

        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            if self.config.third_party.robustness.ai_beauty:
                ai_beauty_tool = self.config.third_party.robustness.ai_beauty_tool
                if ai_beauty_tool == "ai_lab_tools":
                    imgs_A, count = self.aiediting.face_beauty_via_ailabtools(imgs_A)
                elif ai_beauty_tool == "tencent_cloud":
                    imgs_A, count = self.aiediting.face_beauty_via_tencentcloud(imgs_A)
                else:
                    self.logger.critical(f"Unknown ai_beauty_tool: {ai_beauty_tool}")
                    raise ValueError(f"Unknown ai_beauty_tool: {ai_beauty_tool}")
                total_count += count
            elif not self.config.third_party.defense.failure_defense_tracing:
                total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs, silent=True)

            torch.set_grad_enabled(False)
            (
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            ) = self._get_full_swap_results(imgs_A, imgs_B, x_imgs)

            if self.config.third_party.defense.failure_defense_tracing:
                success_swap_indices = self.effectiveness.get_success_swap_indices(
                    imgs_A, pert_imgs_A_src_swap
                )
                if len(success_swap_indices) == 0:
                    continue
                else:
                    total_count += len(success_swap_indices)
                    idx_cpu = success_swap_indices.to("cpu")

                    imgs_A_cpu = imgs_A.detach().to("cpu")
                    del imgs_A
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    imgs_A = self._pick_on_cpu_then_to_gpu(
                        imgs_A_cpu, idx_cpu, device="cuda"
                    )
                    imgs_B_cpu = imgs_B.detach().to("cpu")
                    del imgs_B
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    imgs_B = self._pick_on_cpu_then_to_gpu(
                        imgs_B_cpu, idx_cpu, device="cuda"
                    )
                    x_imgs_cpu = x_imgs.detach().to("cpu")
                    del x_imgs
                    self._free_gpu()
                    x_imgs = self._pick_on_cpu_then_to_gpu(
                        x_imgs_cpu, idx_cpu, device="cuda"
                    )
                    cloak_imgs_cpu = cloak_imgs.detach().to("cpu")
                    del cloak_imgs
                    self._free_gpu()
                    cloak_imgs = self._pick_on_cpu_then_to_gpu(
                        cloak_imgs_cpu, idx_cpu, device="cuda"
                    )
                    imgs_A_src_swap_cpu = imgs_A_src_swap.detach().to("cpu")
                    del imgs_A_src_swap
                    self._free_gpu()
                    imgs_A_src_swap = self._pick_on_cpu_then_to_gpu(
                        imgs_A_src_swap_cpu, idx_cpu, device="cuda"
                    )
                    pert_imgs_A_src_swap_cpu = pert_imgs_A_src_swap.detach().to("cpu")
                    del pert_imgs_A_src_swap
                    self._free_gpu()
                    pert_imgs_A_src_swap = self._pick_on_cpu_then_to_gpu(
                        pert_imgs_A_src_swap_cpu, idx_cpu, device="cuda"
                    )
                    imgs_A_tgt_swap_cpu = imgs_A_tgt_swap.detach().to("cpu")
                    del imgs_A_tgt_swap
                    self._free_gpu()
                    imgs_A_tgt_swap = self._pick_on_cpu_then_to_gpu(
                        imgs_A_tgt_swap_cpu, idx_cpu, device="cuda"
                    )
                    pert_imgs_A_tgt_swap_cpu = pert_imgs_A_tgt_swap.detach().to("cpu")
                    del pert_imgs_A_tgt_swap
                    self._free_gpu()
                    pert_imgs_A_tgt_swap = self._pick_on_cpu_then_to_gpu(
                        pert_imgs_A_tgt_swap_cpu, idx_cpu, device="cuda"
                    )

                    del (
                        idx_cpu,
                        imgs_A_cpu,
                        imgs_B_cpu,
                        x_imgs_cpu,
                        cloak_imgs_cpu,
                        imgs_A_src_swap_cpu,
                        imgs_A_tgt_swap_cpu,
                        pert_imgs_A_src_swap_cpu,
                        pert_imgs_A_tgt_swap_cpu,
                    )
                    self._free_gpu()

            (
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            ) = metric.get_defense_metric(
                self.utility,
                self.effectiveness,
                imgs_A,
                imgs_B,
                x_imgs,
                cloak_imgs,
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            )

            metric.merge_metric(
                self.effectiveness,
                data,
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            )

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "pert_imgs",
                    "cloak_imgs",
                    "swap",
                    "pert_swap",
                    "rev\nswap",
                    "rev\npert_swap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    x_imgs,
                    cloak_imgs,
                    imgs_A_src_swap,
                    pert_imgs_A_src_swap,
                    imgs_A_tgt_swap,
                    pert_imgs_A_tgt_swap,
                ],
                only_save_summary=True,
            )

            del imgs_A, imgs_B, x_imgs, cloak_imgs
            del (
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            )
            self._free_gpu()

            self.logger.info(
                f"""
            utility(mse, psnr, ssim, lpips), effectiveness{self.effectiveness.candi_funcs.keys()} source(pert, swap, pert_swap, anchor) target(swap, pert_swap)
            pert utility: {metric.generate_iter_utility_log(pert_utilities)}
            pert as swap source utility: {metric.generate_iter_utility_log(pert_as_src_swap_utilities)}
            pert as swap target utility: {metric.generate_iter_utility_log(pert_as_tgt_swap_utilities)}
            pert as swap source effectiveness: {metric.generate_iter_effectiveness_log(source_effectivenesses)}
            pert as swap target effectiveness: {metric.generate_iter_effectiveness_log(target_effectivenesses)}
            """
            )

            self.logger.info(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            {metric.generate_summary_utility_log(data, 'pert_utility', idx)}
            {metric.generate_summary_utility_log(data, 'src_pert_swap_utility', idx)}
            {metric.generate_summary_utility_log(data, 'tgt_pert_swap_utility', idx)}
            {metric.generate_summary_effectiveness_log(data, 'src_pert_swap_effectiveness')}
            {metric.generate_summary_effectiveness_log(data, 'tgt_pert_swap_effectiveness')}
            """
            )

    def robustness_sample(self) -> None:
        dataset = SampleDataset(self.config.third_party.dataset.sample_dir)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        logo = self.robustness.load_logo()
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs, silent=False)
            x_imgs = self.robustness.webp_compress(x_imgs, 80)
            x_imgs = self._perturb_imgs(x_imgs, cloak_imgs, silent=False)
            x_imgs = self.robustness.webp_compress(x_imgs, 80)
            x_imgs = self._perturb_imgs(x_imgs, cloak_imgs, silent=False)

            pert_utilities = self.utility.calculate_utility(imgs_A, x_imgs)

            (
                noise_source_effectivenesses,
                noise_target_effectivenesses,
            ) = self.robustness.getgauss_noise_metrics(
                idx, imgs_A, imgs_B, x_imgs, cloak_imgs, self.image_dir
            )

            (
                compress_source_effectivenesses,
                compress_target_effectivenesses,
            ) = self.robustness.get_compress_metrics(
                idx, imgs_A, imgs_B, x_imgs, cloak_imgs, self.image_dir
            )

            (
                crop_source_effectivenesses,
                crop_target_effectivenesses,
            ) = self.robustness.get_crop_metrics(
                idx, imgs_A, imgs_B, x_imgs, cloak_imgs, self.image_dir
            )

            (
                logo_source_effectivenesses,
                logo_target_effectivenesses,
            ) = self.robustness.get_logo_metrics(
                idx, imgs_A, imgs_B, x_imgs, logo, cloak_imgs, self.image_dir
            )

            (
                inc_bright_source_effectivenesses,
                inc_bright_target_effectivenesses,
            ) = self.robustness.get_brightness_metrics(
                idx, imgs_A, imgs_B, x_imgs, 1.25, cloak_imgs, self.image_dir
            )

            (
                dec_bright_source_effectivenesses,
                dec_bright_target_effectivenesses,
            ) = self.robustness.get_brightness_metrics(
                idx, imgs_A, imgs_B, x_imgs, 0.75, cloak_imgs, self.image_dir
            )

            self._free_gpu()
            self.logger.info(
                textwrap.dedent(
                    f"""
                pert utility(mse, psnr, ssim, lpips): {metric.generate_iter_utility_log(pert_utilities)}
                effectiveness tools: {list(self.effectiveness.candi_funcs.keys())}, noise, compress, crop, overlay, increase and decrease the brightness
                source(robust swap, robust pert swap, robust cloak), target(robust swap, robust pert swap)
                noise, compress, crop, logo, inc_bright, dec_bright
                {metric.generate_iter_robustness_log(noise_source_effectivenesses,noise_target_effectivenesses)}
                {metric.generate_iter_robustness_log(compress_source_effectivenesses,compress_target_effectivenesses)}
                {metric.generate_iter_robustness_log(crop_source_effectivenesses,crop_target_effectivenesses)}
                {metric.generate_iter_robustness_log(logo_source_effectivenesses,logo_target_effectivenesses)}
                {metric.generate_iter_robustness_log(inc_bright_source_effectivenesses,inc_bright_target_effectivenesses)}
                {metric.generate_iter_robustness_log(dec_bright_source_effectivenesses,dec_bright_target_effectivenesses)}
                """
                )
            )

    def robustness_metric(self) -> None:
        data = metric.get_robustness_metric_data_template(self.effectiveness)

        logo = self.robustness.load_logo()
        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs, silent=True)
            x_imgs = self.robustness.webp_compress(x_imgs, 80)
            x_imgs = self._perturb_imgs(x_imgs, cloak_imgs, silent=True)
            x_imgs = self.robustness.webp_compress(x_imgs, 80)
            x_imgs = self._perturb_imgs(x_imgs, cloak_imgs, silent=True)

            torch.set_grad_enabled(False)
            pert_utilities = self.utility.calculate_utility(imgs_A, x_imgs)
            metric.merge_single_dict(data["utility"], pert_utilities)

            (
                noise_source_effectivenesses,
                noise_target_effectivenesses,
            ) = self.robustness.getgauss_noise_metrics(
                idx, imgs_A, imgs_B, x_imgs, cloak_imgs, self.image_dir
            )
            metric.merge_single_robustness_metric(
                data,
                noise_source_effectivenesses,
                noise_target_effectivenesses,
                "noise",
            )

            (
                compress_source_effectivenesses,
                compress_target_effectivenesses,
            ) = self.robustness.get_compress_metrics(
                idx, imgs_A, imgs_B, x_imgs, cloak_imgs, self.image_dir
            )
            metric.merge_single_robustness_metric(
                data,
                compress_source_effectivenesses,
                compress_target_effectivenesses,
                "compress",
            )

            (
                crop_source_effectivenesses,
                crop_target_effectivenesses,
            ) = self.robustness.get_crop_metrics(
                idx, imgs_A, imgs_B, x_imgs, cloak_imgs, self.image_dir
            )
            metric.merge_single_robustness_metric(
                data,
                crop_source_effectivenesses,
                crop_target_effectivenesses,
                "crop",
            )

            (
                logo_source_effectivenesses,
                logo_target_effectivenesses,
            ) = self.robustness.get_logo_metrics(
                idx, imgs_A, imgs_B, x_imgs, logo, cloak_imgs, self.image_dir
            )
            metric.merge_single_robustness_metric(
                data,
                logo_source_effectivenesses,
                logo_target_effectivenesses,
                "logo",
            )

            (
                inc_bright_source_effectivenesses,
                inc_bright_target_effectivenesses,
            ) = self.robustness.get_brightness_metrics(
                idx, imgs_A, imgs_B, x_imgs, 1.25, cloak_imgs, self.image_dir
            )
            metric.merge_single_robustness_metric(
                data,
                inc_bright_source_effectivenesses,
                inc_bright_target_effectivenesses,
                "inc_bright",
            )

            (
                dec_bright_source_effectivenesses,
                dec_bright_target_effectivenesses,
            ) = self.robustness.get_brightness_metrics(
                idx, imgs_A, imgs_B, x_imgs, 0.75, cloak_imgs, self.image_dir
            )
            metric.merge_single_robustness_metric(
                data,
                dec_bright_source_effectivenesses,
                dec_bright_target_effectivenesses,
                "dec_bright",
            )

            self._free_gpu()
            self.logger.info(
                f"""
            utility: {metric.generate_iter_utility_log(pert_utilities)}
            noise, compress, crop, overlay, increase and decrease the brightness {self.effectiveness.candi_funcs.keys()}
            source(robust swap, robust pert swap, cloak), target(robust swap, robust pert swap)
            {metric.generate_iter_robustness_log(noise_source_effectivenesses,noise_target_effectivenesses)}
            {metric.generate_iter_robustness_log(compress_source_effectivenesses,compress_target_effectivenesses)}
            {metric.generate_iter_robustness_log(crop_source_effectivenesses,crop_target_effectivenesses)}
            {metric.generate_iter_robustness_log(logo_source_effectivenesses,logo_target_effectivenesses)}
            {metric.generate_iter_robustness_log(inc_bright_source_effectivenesses,inc_bright_target_effectivenesses)}
            {metric.generate_iter_robustness_log(dec_bright_source_effectivenesses,dec_bright_target_effectivenesses)}
            """
            )

            self.logger.info(
                f"""[{idx}/{len(dataloader)}]Average of {total_count} pictures
            utility: {metric.generate_summary_robustness_utility_log(data['utility'], idx)}
            {metric.generate_summary_robustness_log(data['noise'])}
            {metric.generate_summary_robustness_log(data['compress'])}
            {metric.generate_summary_robustness_log(data['crop'])}
            {metric.generate_summary_robustness_log(data['logo'])}
            {metric.generate_summary_robustness_log(data['inc_bright'])}
            {metric.generate_summary_robustness_log(data['dec_bright'])}
            """
            )

    def robustness_forensics_sample(self):
        dataset = SampleDataset(self.config.third_party.dataset.sample_dir)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        logo = self.robustness.load_logo()
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs, silent=False)
            x_imgs = self.robustness.webp_compress(x_imgs, 80)
            x_imgs = self._perturb_imgs(x_imgs, cloak_imgs, silent=False)
            x_imgs = self.robustness.webp_compress(x_imgs, 80)
            x_imgs = self._perturb_imgs(x_imgs, cloak_imgs, silent=False)

            imgs_A_identity = self._get_imgs_identity(imgs_A)
            imgs_A_src_swap = self.target(None, imgs_B, imgs_A_identity, None, True)
            pert_imgs_A_identity = self._get_imgs_identity(x_imgs)
            pert_imgs_A_src_swap = self.target(
                None, imgs_B, pert_imgs_A_identity, None, True
            )

            results = {"clean": pert_imgs_A_src_swap}
            results["noise"] = self.robustness.gauss_noise(pert_imgs_A_src_swap)
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "imgs_A_src_swap",
                    "pert_imgs_A_src_swap",
                    "noise",
                ],
                [
                    imgs_A,
                    imgs_B,
                    imgs_A_src_swap,
                    pert_imgs_A_src_swap,
                    results["noise"],
                ],
                only_save_summary=True,
            )

            results["compress"] = self.robustness.webp_compress(pert_imgs_A_src_swap)
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "imgs_A_src_swap",
                    "pert_imgs_A_src_swap",
                    "compress",
                ],
                [
                    imgs_A,
                    imgs_B,
                    imgs_A_src_swap,
                    pert_imgs_A_src_swap,
                    results["compress"],
                ],
                only_save_summary=True,
            )

            results["crop"] = self.robustness.crop(pert_imgs_A_src_swap)
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "imgs_A_src_swap",
                    "pert_imgs_A_src_swap",
                    "crop",
                ],
                [
                    imgs_A,
                    imgs_B,
                    imgs_A_src_swap,
                    pert_imgs_A_src_swap,
                    results["crop"],
                ],
                only_save_summary=True,
            )

            results["logo"] = self.robustness.logo(pert_imgs_A_src_swap, logo)
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "imgs_A_src_swap",
                    "pert_imgs_A_src_swap",
                    "logo",
                ],
                [
                    imgs_A,
                    imgs_B,
                    imgs_A_src_swap,
                    pert_imgs_A_src_swap,
                    results["logo"],
                ],
                only_save_summary=True,
            )

            results["inc_bright"] = self.robustness.brightness(
                pert_imgs_A_src_swap, 1.25
            )
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "imgs_A_src_swap",
                    "pert_imgs_A_src_swap",
                    "inc_bright",
                ],
                [
                    imgs_A,
                    imgs_B,
                    imgs_A_src_swap,
                    pert_imgs_A_src_swap,
                    results["inc_bright"],
                ],
                only_save_summary=True,
            )

            results["dec_bright"] = self.robustness.brightness(
                pert_imgs_A_src_swap, 0.75
            )
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "imgs_A_src_swap",
                    "pert_imgs_A_src_swap",
                    "dec_bright",
                ],
                [
                    imgs_A,
                    imgs_B,
                    imgs_A_src_swap,
                    pert_imgs_A_src_swap,
                    results["dec_bright"],
                ],
                only_save_summary=True,
            )

            forensic_metrics = metric.get_robustness_forensics_metric(
                self.effectiveness, cloak_imgs, results
            )

            self._free_gpu()
            self.logger.info(
                textwrap.dedent(
                    f"""
            effectiveness tools: {list(self.effectiveness.candi_funcs.keys())}, cloak
            original: {metric.generate_forensics_robustness_log(forensic_metrics['clean'])}
            noise: {metric.generate_forensics_robustness_log(forensic_metrics['noise'])}
            compress: {metric.generate_forensics_robustness_log(forensic_metrics['compress'])}
            crop: {metric.generate_forensics_robustness_log(forensic_metrics['crop'])}
            overlay: {metric.generate_forensics_robustness_log(forensic_metrics['logo'])}
            increase the brightness: {metric.generate_forensics_robustness_log(forensic_metrics['inc_bright'])}
            decrease the brightness: {metric.generate_forensics_robustness_log(forensic_metrics['dec_bright'])}
            """
                )
            )

    def robustness_forensics_metric(self):
        data = metric.get_robustness_forensics_metric_data_template(self.effectiveness)

        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        logo = self.robustness.load_logo()
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs, silent=True)
            x_imgs = self.robustness.webp_compress(x_imgs, 80)
            x_imgs = self._perturb_imgs(x_imgs, cloak_imgs, silent=True)
            x_imgs = self.robustness.webp_compress(x_imgs, 80)
            x_imgs = self._perturb_imgs(x_imgs, cloak_imgs, silent=True)

            torch.set_grad_enabled(False)
            pert_imgs_A_identity = self._get_imgs_identity(x_imgs)
            pert_imgs_A_src_swap = self.target(
                None, imgs_B, pert_imgs_A_identity, None, True
            )

            swap_results = {"clean": pert_imgs_A_src_swap}
            swap_results["noise"] = self.robustness.gauss_noise(pert_imgs_A_src_swap)
            swap_results["compress"] = self.robustness.webp_compress(
                pert_imgs_A_src_swap
            )
            swap_results["crop"] = self.robustness.crop(pert_imgs_A_src_swap)
            swap_results["logo"] = self.robustness.logo(pert_imgs_A_src_swap, logo)
            swap_results["inc_bright"] = self.robustness.brightness(
                pert_imgs_A_src_swap, 1.25
            )
            swap_results["dec_bright"] = self.robustness.brightness(
                pert_imgs_A_src_swap, 0.75
            )

            forensic_metrics = metric.get_robustness_forensics_metric(
                self.effectiveness, cloak_imgs, swap_results
            )
            metric.merge_single_dict(data, forensic_metrics)

            self._free_gpu()
            self.logger.info(
                f"""
            noise, compress, crop, overlay, increase and decrease the brightness {self.effectiveness.candi_funcs.keys()}
            cloak images
            {metric.generate_forensics_robustness_log(forensic_metrics['clean'])}
            {metric.generate_forensics_robustness_log(forensic_metrics['noise'])}
            {metric.generate_forensics_robustness_log(forensic_metrics['compress'])}
            {metric.generate_forensics_robustness_log(forensic_metrics['crop'])}
            {metric.generate_forensics_robustness_log(forensic_metrics['logo'])}
            {metric.generate_forensics_robustness_log(forensic_metrics['inc_bright'])}
            {metric.generate_forensics_robustness_log(forensic_metrics['dec_bright'])}
            """
            )

            self.logger.info(
                f"""[{idx}/{len(dataloader)}]Average of {total_count} pictures
            {metric.generate_forensics_robustness_log(data['clean'])}
            {metric.generate_forensics_robustness_log(data['noise'])}
            {metric.generate_forensics_robustness_log(data['compress'])}
            {metric.generate_forensics_robustness_log(data['crop'])}
            {metric.generate_forensics_robustness_log(data['logo'])}
            {metric.generate_forensics_robustness_log(data['inc_bright'])}
            {metric.generate_forensics_robustness_log(data['dec_bright'])}
            """
            )

    def image_robustness_metric(self):
        data = metric.get_image_robustness_data_template(self.effectiveness)
        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        logo = self.robustness.load_logo()
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)

            noise_imgs_A = self.robustness.gauss_noise(imgs_A)
            self._get_single_robustness_operate_metric(
                imgs_B,
                noise_imgs_A,
                cloak_imgs,
                data["noise"],
            )

            compress_imgs_A = self.robustness.webp_compress(imgs_A)
            self._get_single_robustness_operate_metric(
                imgs_B,
                compress_imgs_A,
                cloak_imgs,
                data["compress"],
            )

            crop_imgs_A = self.robustness.crop(imgs_A)
            self._get_single_robustness_operate_metric(
                imgs_B,
                crop_imgs_A,
                cloak_imgs,
                data["crop"],
            )

            logo_imgs_A = self.robustness.logo(imgs_A, logo)
            self._get_single_robustness_operate_metric(
                imgs_B,
                logo_imgs_A,
                cloak_imgs,
                data["logo"],
            )

            inc_bright_imgs_A = self.robustness.brightness(imgs_A, 1.25)
            self._get_single_robustness_operate_metric(
                imgs_B,
                inc_bright_imgs_A,
                cloak_imgs,
                data["inc_bright"],
            )

            dec_bright_imgs_A = self.robustness.brightness(imgs_A, 0.75)
            self._get_single_robustness_operate_metric(
                imgs_B,
                dec_bright_imgs_A,
                cloak_imgs,
                data["dec_bright"],
            )

            torch.cuda.empty_cache()
            self.logger.info(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            noise
            {metric.generate_summary_utility_log(data['noise'], 'pert_utility', idx)}
            {metric.generate_summary_utility_log(data['noise'], 'src_pert_swap_utility', idx)}
            {metric.generate_summary_utility_log(data['noise'], 'tgt_pert_swap_utility', idx)}
            {metric.generate_summary_effectiveness_log(data['noise'], 'src_pert_swap_effectiveness')}
            {metric.generate_summary_effectiveness_log(data['noise'], 'tgt_pert_swap_effectiveness')}

            compress
            {metric.generate_summary_utility_log(data['compress'], 'pert_utility', idx)}
            {metric.generate_summary_utility_log(data['compress'], 'src_pert_swap_utility', idx)}
            {metric.generate_summary_utility_log(data['compress'], 'tgt_pert_swap_utility', idx)}
            {metric.generate_summary_effectiveness_log(data['compress'], 'src_pert_swap_effectiveness')}
            {metric.generate_summary_effectiveness_log(data['compress'], 'tgt_pert_swap_effectiveness')}

            crop
            {metric.generate_summary_utility_log(data['crop'], 'pert_utility', idx)}
            {metric.generate_summary_utility_log(data['crop'], 'src_pert_swap_utility', idx)}
            {metric.generate_summary_utility_log(data['crop'], 'tgt_pert_swap_utility', idx)}
            {metric.generate_summary_effectiveness_log(data['crop'], 'src_pert_swap_effectiveness')}
            {metric.generate_summary_effectiveness_log(data['crop'], 'tgt_pert_swap_effectiveness')}

            logo
            {metric.generate_summary_utility_log(data['logo'], 'pert_utility', idx)}
            {metric.generate_summary_utility_log(data['logo'], 'src_pert_swap_utility', idx)}
            {metric.generate_summary_utility_log(data['logo'], 'tgt_pert_swap_utility', idx)}
            {metric.generate_summary_effectiveness_log(data['logo'], 'src_pert_swap_effectiveness')}
            {metric.generate_summary_effectiveness_log(data['logo'], 'tgt_pert_swap_effectiveness')}

            inc_bright
            {metric.generate_summary_utility_log(data['inc_bright'], 'pert_utility', idx)}
            {metric.generate_summary_utility_log(data['inc_bright'], 'src_pert_swap_utility', idx)}
            {metric.generate_summary_utility_log(data['inc_bright'], 'tgt_pert_swap_utility', idx)}
            {metric.generate_summary_effectiveness_log(data['inc_bright'], 'src_pert_swap_effectiveness')}
            {metric.generate_summary_effectiveness_log(data['inc_bright'], 'tgt_pert_swap_effectiveness')}

            dec_bright
            {metric.generate_summary_utility_log(data['dec_bright'], 'pert_utility', idx)}
            {metric.generate_summary_utility_log(data['dec_bright'], 'src_pert_swap_utility', idx)}
            {metric.generate_summary_utility_log(data['dec_bright'], 'tgt_pert_swap_utility', idx)}
            {metric.generate_summary_effectiveness_log(data['dec_bright'], 'src_pert_swap_effectiveness')}
            {metric.generate_summary_effectiveness_log(data['dec_bright'], 'tgt_pert_swap_effectiveness')}
            """
            )

    def adaptive_attack(self) -> None:
        data = metric.get_metric_data_template(self.effectiveness)

        dataset = AdaptiveMetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B, imgs_C) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B, imgs_C = imgs_A.cuda(), imgs_B.cuda(), imgs_C.cuda()
            total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs_A = self._perturb_imgs(imgs_A, cloak_imgs, silent=True)
            x_imgs_B = self._perturb_imgs(imgs_B, cloak_imgs, silent=True)

            torch.set_grad_enabled(False)
            x_imgs = x_imgs_A - (x_imgs_B - imgs_B)
            (
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            ) = self._get_full_swap_results(imgs_A, imgs_C, x_imgs)

            (
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            ) = metric.get_defense_metric(
                self.utility,
                self.effectiveness,
                imgs_A,
                imgs_C,
                x_imgs,
                cloak_imgs,
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            )

            metric.merge_metric(
                self.effectiveness,
                data,
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            )

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "imgs_C",
                    "cloak_imgs",
                    "x_imgs",
                    "imgs_A\ndiff",
                    "imgs_B\ndiff",
                ],
                [
                    imgs_A,
                    imgs_B,
                    imgs_C,
                    cloak_imgs,
                    x_imgs,
                    (x_imgs_B - imgs_B) * 10,
                    (x_imgs_A - imgs_A) * 10,
                ],
                only_save_summary=True,
            )

            del imgs_A, imgs_B, imgs_C, x_imgs, cloak_imgs
            del (
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            )
            self._free_gpu()

            self.logger.info(
                f"""
            utility(mse, psnr, ssim, lpips), effectiveness{self.effectiveness.candi_funcs.keys()} source(pert, swap, pert_swap, anchor) target(swap, pert_swap)
            pert utility: {metric.generate_iter_utility_log(pert_utilities)}
            pert as swap source utility: {metric.generate_iter_utility_log(pert_as_src_swap_utilities)}
            pert as swap target utility: {metric.generate_iter_utility_log(pert_as_tgt_swap_utilities)}
            pert as swap source effectiveness: {metric.generate_iter_effectiveness_log(source_effectivenesses)}
            pert as swap target effectiveness: {metric.generate_iter_effectiveness_log(target_effectivenesses)}
            """
            )

            self.logger.info(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            {metric.generate_summary_utility_log(data, 'pert_utility', idx)}
            {metric.generate_summary_utility_log(data, 'src_pert_swap_utility', idx)}
            {metric.generate_summary_utility_log(data, 'tgt_pert_swap_utility', idx)}
            {metric.generate_summary_effectiveness_log(data, 'src_pert_swap_effectiveness')}
            {metric.generate_summary_effectiveness_log(data, 'tgt_pert_swap_effectiveness')}
            """
            )

    def adaptive_attack_self(self) -> None:
        data = metric.get_metric_data_template(self.effectiveness)

        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs_A = self._perturb_imgs(imgs_A, cloak_imgs, silent=True)
            x_x_imgs_A = self._perturb_imgs(x_imgs_A, cloak_imgs, silent=True)

            torch.set_grad_enabled(False)
            x_imgs = x_imgs_A - (x_x_imgs_A - x_imgs_A)
            (
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            ) = self._get_full_swap_results(imgs_A, imgs_B, x_imgs)

            (
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            ) = metric.get_defense_metric(
                self.utility,
                self.effectiveness,
                imgs_A,
                imgs_B,
                x_imgs,
                cloak_imgs,
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            )

            metric.merge_metric(
                self.effectiveness,
                data,
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            )

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "cloak_imgs",
                    "x_imgs",
                    "pert_imgs_A\nsrc_swap",
                    "pert_imgs_A\ntgt_swap",
                    "protect\ndiff",
                    "attack\ndiff",
                ],
                [
                    imgs_A,
                    imgs_B,
                    cloak_imgs,
                    x_imgs,
                    pert_imgs_A_src_swap,
                    pert_imgs_A_tgt_swap,
                    (x_imgs_A - imgs_A) * 10,
                    (x_x_imgs_A - x_imgs_A) * 10,
                ],
                only_save_summary=False,
            )

            del imgs_A, imgs_B, cloak_imgs, x_imgs
            del (
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            )
            self._free_gpu()

            self.logger.info(
                f"""
            utility(mse, psnr, ssim, lpips), effectiveness{self.effectiveness.candi_funcs.keys()} source(pert, swap, pert_swap, anchor) target(swap, pert_swap)
            pert utility: {metric.generate_iter_utility_log(pert_utilities)}
            pert as swap source utility: {metric.generate_iter_utility_log(pert_as_src_swap_utilities)}
            pert as swap target utility: {metric.generate_iter_utility_log(pert_as_tgt_swap_utilities)}
            pert as swap source effectiveness: {metric.generate_iter_effectiveness_log(source_effectivenesses)}
            pert as swap target effectiveness: {metric.generate_iter_effectiveness_log(target_effectivenesses)}
            """
            )

            self.logger.info(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            {metric.generate_summary_utility_log(data, 'pert_utility', idx)}
            {metric.generate_summary_utility_log(data, 'src_pert_swap_utility', idx)}
            {metric.generate_summary_utility_log(data, 'tgt_pert_swap_utility', idx)}
            {metric.generate_summary_effectiveness_log(data, 'src_pert_swap_effectiveness')}
            {metric.generate_summary_effectiveness_log(data, 'tgt_pert_swap_effectiveness')}
            """
            )

    def _perturb_imgs(
        self, imgs: Tensor, cloak_imgs: Tensor, silent: bool = False
    ) -> Tensor:
        l2_loss = nn.MSELoss().cuda()
        x_imgs = imgs.clone().detach() + torch.randn_like(imgs) * 1e-5
        cloak_identity = self._get_imgs_identity(cloak_imgs)
        imgs_latent_code = self.target.netG.encoder(x_imgs)
        epsilon = (
            self.config.third_party.defense.epsilon
            * (torch.max(x_imgs) - torch.min(x_imgs))
            / 2
        )
        limits = (
            tensor(
                [
                    self.config.third_party.defense.limit.R,
                    self.config.third_party.defense.limit.G,
                    self.config.third_party.defense.limit.B,
                ]
            )
            .view(1, 3, 1, 1)
            .cuda()
        )

        best_imgs, best_loss = torch.ones_like(imgs), float("inf")
        for epoch in range(self.config.third_party.defense.epochs):
            x_imgs = x_imgs.clone().detach().requires_grad_(True)

            pert_diff_loss = self.config.third_party.defense.weight.perturb * l2_loss(
                x_imgs, imgs.detach()
            )

            x_identity = self._get_imgs_identity(x_imgs)
            identity_diff_loss = (
                self.config.third_party.defense.weight.identity
                * l2_loss(x_identity, cloak_identity.detach())
            )

            x_latent_code = self.target.netG.encoder(x_imgs)
            context_diff_loss = (
                self.config.third_party.defense.weight.context
                * -torch.clamp(
                    l2_loss(x_latent_code, imgs_latent_code.detach()),
                    0,
                    self.config.third_party.defense.limit.context,
                )
            )

            loss = pert_diff_loss + identity_diff_loss + context_diff_loss
            loss.backward()

            if x_imgs.grad is not None:
                grad_sign = x_imgs.grad.sign().clone().detach()
            else:
                grad_sign = torch.zeros_like(x_imgs)

            x_imgs = x_imgs.clone().detach() - epsilon * grad_sign

            x_imgs = torch.clamp(
                x_imgs,
                min=imgs - limits,
                max=imgs + limits,
            )
            x_imgs = torch.clamp(x_imgs, 0, 1)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_imgs = x_imgs

            if not silent:
                self.logger.info(
                    f"[Epoch {epoch+1:4}/{self.config.third_party.defense.epochs:4}]loss: {loss:.5f}({pert_diff_loss.item():.5f}, {identity_diff_loss.item():.5f}, {context_diff_loss.item():.5f})"
                )

        return best_imgs

    def _get_full_swap_results(
        self, imgs_A: Tensor, imgs_B: Tensor, pert_imgs_A: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        imgs_A_identity = self._get_imgs_identity(imgs_A)
        imgs_A_src_swap = self.target(None, imgs_B, imgs_A_identity, None, True)

        pert_imgs_A_identity = self._get_imgs_identity(pert_imgs_A)
        pert_imgs_A_src_swap = self.target(
            None, imgs_B, pert_imgs_A_identity, None, True
        )

        imgs_B_identity = self._get_imgs_identity(imgs_B)
        imgs_A_tgt_swap = self.target(None, imgs_A, imgs_B_identity, None, True)

        pert_imgs_A_tgt_swap = self.target(
            None, pert_imgs_A, imgs_B_identity, None, True
        )

        return (
            imgs_A_src_swap,
            pert_imgs_A_src_swap,
            imgs_A_tgt_swap,
            pert_imgs_A_tgt_swap,
        )

    def _get_single_robustness_operate_metric(
        self,
        imgs_B: Tensor,
        operate_imgs_A: Tensor,
        cloak_imgs: Tensor,
        data: dict,
    ) -> None:
        torch.set_grad_enabled(True)
        operate_x_imgs = self._perturb_imgs(operate_imgs_A, cloak_imgs, silent=True)
        torch.set_grad_enabled(False)
        (
            imgs_A_src_swap,
            pert_imgs_A_src_swap,
            imgs_A_tgt_swap,
            pert_imgs_A_tgt_swap,
        ) = self._get_full_swap_results(operate_imgs_A, imgs_B, operate_x_imgs)

        (
            pert_utilities,
            pert_as_src_swap_utilities,
            pert_as_tgt_swap_utilities,
            source_effectivenesses,
            target_effectivenesses,
        ) = metric.get_defense_metric(
            self.utility,
            self.effectiveness,
            operate_imgs_A,
            imgs_B,
            operate_x_imgs,
            cloak_imgs,
            imgs_A_src_swap,
            pert_imgs_A_src_swap,
            imgs_A_tgt_swap,
            pert_imgs_A_tgt_swap,
        )

        metric.merge_metric(
            self.effectiveness,
            data,
            pert_utilities,
            pert_as_src_swap_utilities,
            pert_as_tgt_swap_utilities,
            source_effectivenesses,
            target_effectivenesses,
        )

    def _pick_on_cpu_then_to_gpu(
        self, big_cpu: Tensor, idx_cpu: Tensor, device="cuda"
    ) -> Tensor:
        with torch.no_grad():
            small_cpu = torch.index_select(big_cpu, 0, idx_cpu)
            small_gpu = small_cpu.to(device, non_blocking=True)
            return small_gpu

    def _free_gpu(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
