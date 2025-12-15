from src import metric
from src.faceshifter.base import Base
from src.dataset import MetricDataset
from src.evaluate import ScoreCalculator
from src.utils import save_tensor_imgs

import torch
import textwrap
import torch.nn.functional as F
from torch import Tensor, tensor
from torch.utils.data import DataLoader
from pathlib import Path


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.score_calculator = ScoreCalculator(logger, config)

        notes_path = Path(self.config.notes_path)
        notes_path.touch(exist_ok=True)

    def metric(
        self,
    ) -> None:
        metrics = metric.get_metric_data_template(self.effectiveness)

        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        batch_size = self.config.third_party.defense.batch_size
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            cloak_imgs = self.cloak.find_best_cloaks(self._denormalize(imgs_A))
            x_imgs = self._perturb_imgs(self._denormalize(imgs_A), cloak_imgs)

            torch.set_grad_enabled(False)

            imgs_A_list = list(torch.chunk(imgs_A, chunks=batch_size, dim=0))
            imgs_B_list = list(torch.chunk(imgs_B, chunks=batch_size, dim=0))
            cloak_imgs_list = list(torch.chunk(cloak_imgs, chunks=batch_size, dim=0))
            x_imgs_list = list(
                torch.chunk(self._normalize(x_imgs), chunks=batch_size, dim=0)
            )

            imgs_A_src_swap = []
            cloak_imgs_src_swap = []
            pert_imgs_A_src_swap = []
            imgs_A_tgt_swap = []
            pert_imgs_A_tgt_swap = []
            valid_indexes = []
            for i in range(batch_size):
                try:
                    result = self.swapface(imgs_A_list[i], imgs_B_list[i]).cuda()
                    cloak_result = self.swapface(
                        cloak_imgs_list[i], imgs_B_list[i]
                    ).cuda()
                    pert_result = self.swapface(x_imgs_list[i], imgs_B_list[i]).cuda()
                    reverse_result = self.swapface(
                        imgs_B_list[i], imgs_A_list[i]
                    ).cuda()
                    reverse_pert_result = self.swapface(
                        imgs_B_list[i], x_imgs_list[i]
                    ).cuda()

                    imgs_A_src_swap.append(result)
                    cloak_imgs_src_swap.append(cloak_result)
                    pert_imgs_A_src_swap.append(pert_result)
                    imgs_A_tgt_swap.append(reverse_result)
                    pert_imgs_A_tgt_swap.append(reverse_pert_result)

                    valid_indexes.append(i)
                    total_count += 1
                except Exception as e:
                    for imgs_list in [
                        imgs_A_src_swap,
                        cloak_imgs_src_swap,
                        pert_imgs_A_src_swap,
                        imgs_A_tgt_swap,
                        pert_imgs_A_tgt_swap,
                    ]:
                        if len(imgs_list) < i + 1:
                            imgs_list.append(torch.zeros_like(imgs_A[0]).cuda())
                    continue

            images_idx = torch.as_tensor(valid_indexes, device=imgs_A.device)
            imgs_A = self._denormalize(imgs_A[images_idx])
            imgs_B = self._denormalize(imgs_B[images_idx])
            x_imgs = x_imgs[images_idx]
            cloak_imgs = cloak_imgs[images_idx]
            results = torch.cat(
                [imgs_A_src_swap[i] for i in valid_indexes], dim=0
            ).float()
            cloak_results = torch.cat(
                [cloak_imgs_src_swap[i] for i in valid_indexes], dim=0
            ).float()
            pert_results = torch.cat(
                [pert_imgs_A_src_swap[i] for i in valid_indexes], dim=0
            ).float()
            reverse_results = torch.cat(
                [imgs_A_tgt_swap[i] for i in valid_indexes], dim=0
            ).float()
            reverse_pert_results = torch.cat(
                [pert_imgs_A_tgt_swap[i] for i in valid_indexes], dim=0
            ).float()

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
                results,
                pert_results,
                reverse_results,
                reverse_pert_results,
            )

            metric.merge_metric(
                self.effectiveness,
                metrics,
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
                    "cloak_swap",
                    "pert_swap",
                    "rev\nswap",
                    "rev\npert_swap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    x_imgs,
                    cloak_imgs,
                    results,
                    cloak_results,
                    pert_results,
                    reverse_results,
                    reverse_pert_results,
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            del imgs_A, imgs_B, x_imgs, cloak_imgs
            self._free_gpu()

            scores = self.score_calculator.calculate_score(
                source_effectivenesses, target_effectivenesses, metrics
            )

            iter_log_str = textwrap.dedent(
                f"""
            utility(mse, psnr, ssim, lpips), effectiveness {tuple(source_effectivenesses.keys())} identity {tuple(next(iter(source_effectivenesses.values())).keys())} context {tuple(next(iter(target_effectivenesses.values())).keys())}
            pert utility: {metric.generate_iter_utility_log(pert_utilities)}
            pert as swap source utility: {metric.generate_iter_utility_log(pert_as_src_swap_utilities)}
            pert as swap target utility: {metric.generate_iter_utility_log(pert_as_tgt_swap_utilities)}
            pert as swap source effectiveness: {metric.generate_iter_effectiveness_log(source_effectivenesses)}
            pert as swap target effectiveness: {metric.generate_iter_effectiveness_log(target_effectivenesses)}
            scores: {metric.generate_iter_score_log(scores)}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            {metric.generate_summary_utility_log(metrics, 'pert_utility', idx)}
            {metric.generate_summary_utility_log(metrics, 'src_pert_swap_utility', idx)}
            {metric.generate_summary_utility_log(metrics, 'tgt_pert_swap_utility', idx)}
            {metric.generate_summary_effectiveness_log(metrics, 'src_pert_swap_effectiveness')}
            {metric.generate_summary_effectiveness_log(metrics, 'tgt_pert_swap_effectiveness')}
            scores: {metric.generate_summary_score_log(scores)}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

    def _perturb_imgs(self, imgs: Tensor, cloak_imgs: Tensor) -> Tensor:
        def l2_per_image(x: Tensor, y: Tensor) -> Tensor:
            return ((x - y) ** 2).view(x.size(0), -1).mean(dim=1)

        def get_imgs_identity(imgs: Tensor) -> Tensor:
            return self.arcface(
                F.interpolate(
                    imgs[:, :, 19:237, 19:237],
                    (112, 112),
                    mode="bilinear",
                    align_corners=True,
                )
            )

        x_imgs = imgs.clone().detach() + torch.randn_like(imgs) * 1e-5

        with torch.no_grad():
            self_identity = get_imgs_identity(imgs)
            cloak_identity = get_imgs_identity(cloak_imgs)
            imgs_latent_code = self.G.encoder(imgs)

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

        B = imgs.size(0)
        best_imgs = imgs.clone()
        best_loss = torch.full((B,), float("inf"), device=imgs.device)

        for epoch in range(self.config.third_party.defense.epochs):
            x_imgs = x_imgs.clone().detach().requires_grad_(True)

            pert_diff_loss = (
                self.config.third_party.defense.weight.perturb
                * l2_per_image(x_imgs, imgs.detach())
            )

            x_identity = get_imgs_identity(x_imgs)
            identity_diff = torch.clamp(
                l2_per_image(x_identity, self_identity),
                0,
                self.config.third_party.defense.limit.identity,
            )
            identity_diff_loss = (
                -self.config.third_party.defense.weight.identity * identity_diff
            )

            cloak_diff_loss = (
                self.config.third_party.defense.weight.cloak
                * l2_per_image(x_identity, cloak_identity)
            )

            x_latent_code = self.G.encoder(x_imgs)
            context_diff = torch.clamp(
                l2_per_image(x_latent_code, imgs_latent_code),
                0,
                self.config.third_party.defense.limit.context,
            )
            context_diff_loss = (
                -self.config.third_party.defense.weight.context * context_diff
            )

            loss_per_img = (
                pert_diff_loss
                + identity_diff_loss
                + cloak_diff_loss
                + context_diff_loss
            )
            loss = loss_per_img.mean()
            loss.backward()

            if x_imgs.grad is not None:
                grad_sign = x_imgs.grad.sign().detach()
            else:
                grad_sign = torch.zeros_like(x_imgs)

            x_imgs = x_imgs.detach() - epsilon * grad_sign
            x_imgs = torch.clamp(
                x_imgs,
                min=imgs - limits,
                max=imgs + limits,
            )
            x_imgs = torch.clamp(x_imgs, 0, 1)

            loss_per_img_detached = loss_per_img.detach()
            improved = loss_per_img_detached < best_loss
            best_loss[improved] = loss_per_img_detached[improved]
            best_imgs[improved] = x_imgs[improved].detach()

            if (
                not self.config.third_party.defense.silent_perturb
                and (epoch + 1) % self.config.third_party.defense.log_interval == 0
                or epoch + 1 == self.config.third_party.defense.epochs
            ):
                self.logger.info(
                    f"[Epoch {epoch+1:4}/{self.config.third_party.defense.epochs:4}] "
                    f"loss: {loss.item():.5f}("
                    f"{pert_diff_loss.mean().item():.5f}, "
                    f"{identity_diff_loss.mean().item():.5f}, "
                    f"{cloak_diff_loss.mean().item():.5f}, "
                    f"{context_diff_loss.mean().item():.5f})"
                )

        return best_imgs

    def _free_gpu(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
