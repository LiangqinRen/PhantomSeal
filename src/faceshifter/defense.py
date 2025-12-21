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
            pert_imgs = self._perturb_imgs(self._denormalize(imgs_A), cloak_imgs)

            torch.set_grad_enabled(False)

            imgs_A_list = list(torch.chunk(imgs_A, chunks=batch_size, dim=0))
            imgs_B_list = list(torch.chunk(imgs_B, chunks=batch_size, dim=0))
            pert_imgs_list = list(
                torch.chunk(self._normalize(pert_imgs), chunks=batch_size, dim=0)
            )

            source_swap_list = []
            pert_source_swap_list = []
            valid_indexes = []
            for i in range(batch_size):
                try:
                    source_swap = self.swapface(imgs_A_list[i], imgs_B_list[i]).cuda()
                    pert_source_swap = self.swapface(
                        pert_imgs_list[i], imgs_B_list[i]
                    ).cuda()

                    source_swap_list.append(source_swap)
                    pert_source_swap_list.append(pert_source_swap)

                    valid_indexes.append(i)
                    total_count += 1
                except Exception as e:
                    for imgs_list in [
                        source_swap_list,
                        pert_source_swap_list,
                    ]:
                        if len(imgs_list) < i + 1:
                            imgs_list.append(torch.zeros_like(imgs_A[0]).cuda())
                    continue

            images_idx = torch.as_tensor(valid_indexes, device=imgs_A.device)
            imgs_A = self._denormalize(imgs_A[images_idx])
            imgs_B = self._denormalize(imgs_B[images_idx])
            pert_imgs = pert_imgs[images_idx]
            cloak_imgs = cloak_imgs[images_idx]

            source_swap = torch.cat(
                [source_swap_list[i] for i in valid_indexes], dim=0
            ).float()
            pert_source_swap = torch.cat(
                [pert_source_swap_list[i] for i in valid_indexes], dim=0
            ).float()

            (
                utility,
                source_utility,
                _,
                source_effectiveness,
                _,
            ) = metric.get_defense_metric(
                self.utility,
                self.effectiveness,
                imgs_A,
                imgs_B,
                pert_imgs,
                cloak_imgs,
                source_swap,
                pert_source_swap,
                None,
                None,
            )

            metric.merge_metric(
                self.effectiveness,
                metrics,
                utility,
                source_utility,
                None,
                source_effectiveness,
                None,
            )

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "perturb\nimgs",
                    "cloak\nimgs",
                    "source\nswap",
                    "perturb\nsource\nswap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    pert_imgs,
                    cloak_imgs,
                    source_swap,
                    pert_source_swap,
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            del imgs_A, imgs_B, pert_imgs, cloak_imgs
            self._free_gpu()

            scores = self.score_calculator.calculate_score(
                source_effectiveness, None, metrics
            )

            iter_log_str = textwrap.dedent(
                f"""
            utility (mse, psnr, ssim, lpips), effectiveness ({', '.join(self.effectiveness.candi_funcs.keys())}), identity ({', '.join(next(iter(source_effectiveness.values())).keys())})
            utility: {metric.generate_iter_utility_log(utility)}
            source utility: {metric.generate_iter_utility_log(source_utility)}
            source effectiveness: {metric.generate_iter_effectiveness_log(source_effectiveness)}
            scores: {metric.generate_iter_score_log(scores)}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            utility: {metric.generate_summary_utility_log(metrics, 'utility', idx)}
            source utility: {metric.generate_summary_utility_log(metrics, 'pert_source_utility', idx)}
            source effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'pert_source_effectiveness')}
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

            loss_per_img = pert_diff_loss + identity_diff_loss + cloak_diff_loss
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
                    f"{cloak_diff_loss.mean().item():.5f})"
                )

        return best_imgs

    def _free_gpu(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
