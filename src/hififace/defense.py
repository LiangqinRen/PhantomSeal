import metric
from hififace.base import Base
from dataset import MetricDataset
from utils import save_tensor_imgs

import torch
from torch import tensor, nn
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

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
            total_count += len(imgs_A)

            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs, silent=True)

            with torch.no_grad():
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
                True,
            )

            del imgs_A, imgs_B, x_imgs, cloak_imgs
            del (
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
                cloak_result_imgs,
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

    def _perturb_imgs(
        self, imgs: tensor, cloak_imgs: tensor, silent: bool = False
    ) -> tensor:
        l2_loss = nn.MSELoss().cuda()
        cloak_3d = self.net.generator.id_extractor.f_3d(cloak_imgs)[:, :80]
        context_3d = self.net.generator.id_extractor.f_3d(imgs)[:, 80:]
        cloak_identity = F.normalize(
            self.net.generator.id_extractor.f_id(
                F.interpolate((cloak_imgs - 0.5) / 0.5, size=112, mode="bilinear")
            ),
            dim=-1,
            p=2,
        )
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

        x_imgs = imgs.clone().detach() + 1e-5
        best_imgs, best_loss = None, float("inf")
        for epoch in range(self.config.third_party.defense.epochs):
            x_imgs.requires_grad = True

            pert_loss = self.config.third_party.defense.weight.perturb * l2_loss(
                x_imgs, imgs.detach()
            )

            x_3d = self.net.generator.id_extractor.f_3d(x_imgs)[:, :80]
            identity_3d_loss = (
                self.config.third_party.defense.weight.identity_3d
                * l2_loss(x_3d, cloak_3d.detach())
            )

            x_identity = F.normalize(
                self.net.generator.id_extractor.f_id(
                    F.interpolate((x_imgs - 0.5) / 0.5, size=112, mode="bilinear")
                ),
                dim=-1,
                p=2,
            )
            identity_id_loss = (
                self.config.third_party.defense.weight.identity_id
                * l2_loss(x_identity, cloak_identity.detach())
            )

            x_middle_feat, x_final_feat = self.net.generator.encoder(x_imgs)
            context_middle_loss = (
                -self.config.third_party.defense.weight.context_middle
                * l2_loss(x_middle_feat, middle_feat.detach())
            )
            context_final_loss = (
                -self.config.third_party.defense.weight.context_final
                * torch.clamp(
                    l2_loss(x_final_feat, final_feat.detach()),
                    min=0,
                    max=self.config.third_party.defense.limit.context_final,
                )
            )

            x_context_3d = self.net.generator.id_extractor.f_3d(x_imgs)[:, 80:]
            context_3d_loss = (
                -self.config.third_party.defense.weight.context_3d
                * l2_loss(x_context_3d, context_3d.detach())
            )

            loss = (
                pert_loss
                + identity_3d_loss
                + identity_id_loss
                + context_middle_loss
                + context_final_loss
                + context_3d_loss
            )
            loss.backward(retain_graph=False)

            x_imgs = (
                x_imgs.clone().detach() - epsilon * x_imgs.grad.sign().clone().detach()
            )

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
                    f"[Epoch {epoch+1:4}/{self.config.third_party.defense.epochs}]loss: {loss:.5f}({pert_loss.item():.5f}, {identity_3d_loss.item():.5f}, {identity_id_loss.item():.5f}, {context_middle_loss.item():.5f}, {context_final_loss.item():.5f}, {context_3d_loss.item():.5f})"
                )

        return best_imgs
