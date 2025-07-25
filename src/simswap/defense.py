import metric
from simswap.base import Base
from dataset import SampleDataset, MetricDataset
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
            else:
                total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs, silent=True)

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
            torch.cuda.empty_cache()

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
            ) = self.robustness.get_gauss_noise_metrics(
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

            torch.cuda.empty_cache()
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
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs, silent=True)
            x_imgs = self.robustness.webp_compress(x_imgs, 80)
            x_imgs = self._perturb_imgs(x_imgs, cloak_imgs, silent=True)
            x_imgs = self.robustness.webp_compress(x_imgs, 80)
            x_imgs = self._perturb_imgs(x_imgs, cloak_imgs, silent=True)

            pert_utilities = self.utility.calculate_utility(imgs_A, x_imgs)
            metric.merge_single_dict(data["utility"], pert_utilities)

            (
                noise_source_effectivenesses,
                noise_target_effectivenesses,
            ) = self.robustness.get_gauss_noise_metrics(
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

            torch.cuda.empty_cache()
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
