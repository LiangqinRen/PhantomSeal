import metric
from faceshifter.base import Base
from dataset import MetricDataset
from utils import save_tensor_imgs

import torch
import torch.nn.functional as F
from torch import tensor, nn
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision.utils import save_image


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def metric(
        self,
    ) -> None:
        metric_data = metric.get_metric_data_template(self.effectiveness)

        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        batch_size = self.config.third_party.defense.batch_size
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            cloak_imgs = self.cloak.find_best_cloaks(self._denormalize(imgs_A))
            x_imgs = self._perturb_imgs(
                self._denormalize(imgs_A), cloak_imgs, silent=True
            )

            x_imgs = self._normalize(x_imgs)
            imgs_A = list(torch.chunk(imgs_A, chunks=batch_size, dim=0))
            imgs_B = list(torch.chunk(imgs_B, chunks=batch_size, dim=0))
            cloak_imgs = list(torch.chunk(cloak_imgs, chunks=batch_size, dim=0))
            x_imgs = list(torch.chunk(x_imgs, chunks=batch_size, dim=0))

            imgs_A_src_swap = []
            cloak_imgs_src_swap = []
            pert_imgs_A_src_swap = []
            imgs_A_tgt_swap = []
            pert_imgs_A_tgt_swap = []
            for i in range(batch_size):
                try:
                    result = self.swapface(imgs_A[i], imgs_B[i])
                    cloak_result = self.swapface(cloak_imgs[i], imgs_B[i])
                    pert_result = self.swapface(x_imgs[i], imgs_B[i])
                    reverse_result = self.swapface(imgs_B[i], imgs_A[i])
                    reverse_pert_result = self.swapface(imgs_B[i], x_imgs[i])

                    imgs_A_src_swap.append(result)
                    cloak_imgs_src_swap.append(cloak_result)
                    pert_imgs_A_src_swap.append(pert_result)
                    imgs_A_tgt_swap.append(reverse_result)
                    pert_imgs_A_tgt_swap.append(reverse_pert_result)

                    imgs_A[i] = self._denormalize(imgs_A[i])
                    imgs_B[i] = self._denormalize(imgs_B[i])
                    x_imgs[i] = self._denormalize(x_imgs[i])

                    (
                        pert_utilities,
                        pert_as_src_swap_utilities,
                        pert_as_tgt_swap_utilities,
                        source_effectivenesses,
                        target_effectivenesses,
                    ) = metric.get_defense_metric(
                        self.utility,
                        self.effectiveness,
                        imgs_A[i].float(),
                        imgs_B[i].float(),
                        x_imgs[i].float(),
                        cloak_imgs[i].float(),
                        result.cuda().float(),
                        pert_result.cuda().float(),
                        reverse_result.cuda().float(),
                        reverse_pert_result.cuda().float(),
                    )

                    metric.merge_metric(
                        self.effectiveness,
                        metric_data,
                        pert_utilities,
                        pert_as_src_swap_utilities,
                        pert_as_tgt_swap_utilities,
                        source_effectivenesses,
                        target_effectivenesses,
                    )
                    del (
                        result,
                        pert_result,
                        reverse_result,
                        reverse_pert_result,
                    )
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
                            imgs_list.append(torch.zeros_like(imgs_A[0].cuda()))
                    continue

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_b",
                    "pert_imgs",
                    "cloak_imgs",
                    "swap",
                    "cloak_swap",
                    "pert_swap",
                    "rev\nswap",
                    "rev\npert_swap",
                ],
                [
                    torch.cat(imgs_A, dim=0).cuda(),
                    torch.cat(imgs_B, dim=0).cuda(),
                    torch.cat(x_imgs, dim=0).cuda(),
                    torch.cat(cloak_imgs, dim=0).cuda(),
                    torch.cat(imgs_A_src_swap, dim=0).cuda(),
                    torch.cat(cloak_imgs_src_swap, dim=0).cuda(),
                    torch.cat(pert_imgs_A_src_swap, dim=0).cuda(),
                    torch.cat(imgs_A_tgt_swap, dim=0).cuda(),
                    torch.cat(pert_imgs_A_tgt_swap, dim=0).cuda(),
                ],
                only_save_summary=True,
            )

            del imgs_A, imgs_B, x_imgs, cloak_imgs
            torch.cuda.empty_cache()

            self.logger.info(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            {metric.generate_summary_utility_log(metric_data, 'pert_utility', total_count)}
            {metric.generate_summary_utility_log(metric_data, 'src_pert_swap_utility', total_count)}
            {metric.generate_summary_utility_log(metric_data, 'tgt_pert_swap_utility', total_count)}
            {metric.generate_summary_effectiveness_log(metric_data, 'src_pert_swap_effectiveness')}
            {metric.generate_summary_effectiveness_log(metric_data, 'tgt_pert_swap_effectiveness')}
            """
            )

    def _perturb_imgs(
        self, imgs: tensor, cloak_imgs: tensor, silent: bool = False
    ) -> tensor:
        l2_loss = nn.MSELoss().cuda()
        x_imgs = (imgs + 1e-5).detach().requires_grad_(True)
        cloak_identity = self.arcface(
            F.interpolate(
                cloak_imgs[:, :, 19:237, 19:237],
                (112, 112),
                mode="bilinear",
                align_corners=True,
            )
        )
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

        best_imgs, best_loss = None, float("inf")
        for epoch in range(self.config.third_party.defense.epochs):
            x_imgs = x_imgs.detach().requires_grad_(True)

            pert_diff_loss = self.config.third_party.defense.weight.perturb * l2_loss(
                x_imgs, imgs.detach()
            )

            x_identity = self.arcface(
                F.interpolate(
                    x_imgs[:, :, 19:237, 19:237],
                    (112, 112),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            identity_diff_loss = (
                self.config.third_party.defense.weight.identity
                * l2_loss(x_identity, cloak_identity.detach())
            )

            x_latent_code = self.G.encoder(x_imgs)
            context_diff_loss = (
                self.config.third_party.defense.weight.context
                * -torch.clamp(
                    sum(
                        l2_loss(x1, x2.detach())
                        for x1, x2 in zip(x_latent_code[6:7], imgs_latent_code[6:7])
                    ),
                    0,
                    self.config.third_party.defense.limit.context,
                )
            )

            loss = pert_diff_loss + identity_diff_loss + context_diff_loss
            loss.backward()

            with torch.no_grad():
                x_imgs -= epsilon * x_imgs.grad.sign()
                x_imgs.clamp_(min=imgs - limits, max=imgs + limits)
                x_imgs.clamp_(min=0, max=1)

            epsilon *= 0.995

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_imgs = x_imgs

            if not silent:
                self.logger.info(
                    f"[Epoch {epoch+1:4}/{self.config.third_party.defense.epochs:4}]loss: {loss:.5f}({pert_diff_loss.item():.5f}, {identity_diff_loss.item():.5f}, {context_diff_loss.item():.5f})"
                )

        return best_imgs
