from src import metric
from src.hififace.base import Base
from src.dataset import MetricDataset
from src.utils import save_tensor_imgs
from src.evaluate import ScoreCalculator

import torch
import textwrap
from torch import tensor, Tensor
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F


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
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            total_count += len(imgs_A)

            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs)
            torch.set_grad_enabled(False)

            imgs_A_src_swap = self.net(imgs_A, imgs_B)
            pert_imgs_A_src_swap = self.net(x_imgs, imgs_B)
            imgs_A_tgt_swap = self.net(imgs_B, imgs_A)
            pert_imgs_A_tgt_swap = self.net(imgs_B, x_imgs)
            cloak_result_imgs = self.net(cloak_imgs, imgs_B)

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
                    "imgs_b",
                    "pert_imgs",
                    "cloak_imgs",
                    "swap",
                    "pert_swap",
                    "cloak_swap",
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
                    cloak_result_imgs,
                    imgs_A_tgt_swap,
                    pert_imgs_A_tgt_swap,
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            del imgs_A, imgs_B, x_imgs, cloak_imgs
            del (
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
                cloak_result_imgs,
            )
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

        def get_identity(imgs: Tensor) -> Tensor:
            return F.normalize(
                self.net.generator.id_extractor.f_id(
                    F.interpolate((imgs - 0.5) / 0.5, size=112, mode="bilinear")
                ),
                dim=-1,
                p=2,
            )

        x_imgs = imgs.clone().detach() + torch.randn_like(imgs) * 1e-5

        with torch.no_grad():
            self_3d = self.net.generator.id_extractor.f_3d(imgs)[:, :80]
            cloak_3d = self.net.generator.id_extractor.f_3d(cloak_imgs)[:, :80]
            context_3d = self.net.generator.id_extractor.f_3d(imgs)[:, 80:]
            self_identity = get_identity(imgs)
            cloak_identity = get_identity(cloak_imgs)
            middle_feat, final_feat = self.net.generator.encoder(imgs)

        epsilon = (
            self.config.third_party.defense.epsilon
            * (torch.max(imgs) - torch.min(imgs))
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

            # identity 3d loss
            x_3d = self.net.generator.id_extractor.f_3d(x_imgs)[:, :80]
            identity_3d_diff = torch.clamp(
                l2_per_image(x_3d, self_3d),
                0,
                self.config.third_party.defense.limit.identity_3d,
            )
            x_3d_diff_loss = (
                -self.config.third_party.defense.weight.identity_3d * identity_3d_diff
            )

            cloak_3d_diff_loss = (
                self.config.third_party.defense.weight.cloak_3d
                * l2_per_image(x_3d, cloak_3d.detach())
            )

            # identity id loss
            x_identity = get_identity(x_imgs)
            identity_id_diff = torch.clamp(
                l2_per_image(x_identity, self_identity),
                0,
                self.config.third_party.defense.limit.identity,
            )
            x_id_diff_loss = (
                -self.config.third_party.defense.weight.identity_id * identity_id_diff
            )

            cloak_id_diff_loss = (
                self.config.third_party.defense.weight.cloak_id
                * l2_per_image(x_identity, cloak_identity.detach())
            )

            # context loss
            context_middle_loss = torch.tensor(0.0, device=x_imgs.device)
            context_final_loss = torch.tensor(0.0, device=x_imgs.device)
            context_3d_loss = torch.tensor(0.0, device=x_imgs.device)
            if (
                self.config.third_party.defense.weight.context_middle > 0
                and self.config.third_party.defense.weight.context_final > 0
                and self.config.third_party.defense.weight.context_3d > 0
            ):
                x_middle_feat, x_final_feat = self.net.generator.encoder(x_imgs)
                context_middle_loss = (
                    -self.config.third_party.defense.weight.context_middle
                    * l2_per_image(x_middle_feat, middle_feat.detach())
                )
                context_final_loss = (
                    -self.config.third_party.defense.weight.context_final
                    * torch.clamp(
                        l2_per_image(x_final_feat, final_feat.detach()),
                        min=0,
                        max=self.config.third_party.defense.limit.context_final,
                    )
                )

                x_context_3d = self.net.generator.id_extractor.f_3d(x_imgs)[:, 80:]
                context_3d_loss = (
                    -self.config.third_party.defense.weight.context_3d
                    * l2_per_image(x_context_3d, context_3d.detach())
                )

            loss_per_img = (
                pert_diff_loss
                + x_3d_diff_loss
                + cloak_3d_diff_loss
                + x_id_diff_loss
                + cloak_id_diff_loss
                + context_middle_loss
                + context_final_loss
                + context_3d_loss
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
                or (epoch + 1) == self.config.third_party.defense.epochs
            ):
                self.logger.info(
                    f"[Epoch {epoch+1:4}/{self.config.third_party.defense.epochs:4}] "
                    f"loss: {loss.item():.3f}("
                    f"{pert_diff_loss.mean().item():.3f}, "
                    f"{x_3d_diff_loss.mean().item():.3f}, "
                    f"{cloak_3d_diff_loss.mean().item():.3f}, "
                    f"{x_id_diff_loss.mean().item():.3f}, "
                    f"{cloak_id_diff_loss.mean().item():.3f}, "
                    f"{context_middle_loss.mean().item():.3f}, "
                    f"{context_final_loss.mean().item():.3f}, "
                    f"{context_3d_loss.mean().item():.3f})"
                )

        return best_imgs

    def _free_gpu(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
