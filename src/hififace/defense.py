from src import metric
from src.hififace.base import Base
from src.dataset import MetricDataset
from src.common_utils import save_tensor_imgs
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

        notes_path = Path(self.config.notes_path)
        notes_path.touch(exist_ok=True)

        self.score_calculator = ScoreCalculator(logger, config)

    @torch.no_grad()
    def swap(self) -> None:
        dataset = MetricDataset(self.config)
        swap_batch_size = self.config.third_party.dataset.swap_batch_size
        dataloader = DataLoader(dataset, batch_size=swap_batch_size, shuffle=False)
        metrics = self._get_swap_success_metric_data_template(self.effectiveness)
        total_count = 0

        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)
            source_swap = self.swap_face(imgs_A, imgs_B)
            target_swap = self.swap_face(imgs_B, imgs_A)

            source_effectiveness = self.effectiveness.calculate_effectiveness(
                imgs_A, None, source_swap, None, None
            )
            target_effectiveness = self.effectiveness.calculate_effectiveness(
                imgs_B, None, target_swap, None, None
            )
            self._merge_swap_success_metric(
                metrics, source_effectiveness, target_effectiveness
            )

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "source_swap",
                    "target_swap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    source_swap,
                    target_swap,
                ],
                only_save_summary=True,
            )

            iter_log_str = textwrap.dedent(
                f"""
            𝒯_identity effectiveness {metric.generate_iter_effectiveness_label(source_effectiveness)}: {metric.generate_iter_effectiveness_log(source_effectiveness, include_labels=False)}
            𝒯_context effectiveness {metric.generate_iter_effectiveness_label(target_effectiveness)}: {metric.generate_iter_effectiveness_log(target_effectiveness, include_labels=False)}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            𝒯_identity effectiveness {metric.generate_summary_effectiveness_label(metrics, 'source_effectiveness')}: {metric.generate_summary_effectiveness_log(metrics, 'source_effectiveness', include_labels=False)}
            𝒯_context effectiveness {metric.generate_summary_effectiveness_label(metrics, 'target_effectiveness')}: {metric.generate_summary_effectiveness_log(metrics, 'target_effectiveness', include_labels=False)}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

            del imgs_A, imgs_B, source_swap, target_swap
            self._free_gpu()

    @staticmethod
    def _get_swap_success_metric_data_template(effectiveness) -> dict:
        data = {
            "source_effectiveness": {},
            "target_effectiveness": {},
        }

        for function in effectiveness.candi_funcs.keys():
            data["source_effectiveness"][function] = {"swap": (0, 0)}
            data["target_effectiveness"][function] = {"swap": (0, 0)}

        return data

    @staticmethod
    def _merge_swap_success_metric(
        metrics: dict, source_effectiveness: dict, target_effectiveness: dict
    ) -> None:
        for effec in source_effectiveness.keys():
            source_prev = metrics["source_effectiveness"][effec]["swap"]
            source_cur = source_effectiveness[effec]["swap"]
            metrics["source_effectiveness"][effec]["swap"] = (
                source_prev[0] + source_cur[0],
                source_prev[1] + source_cur[1],
            )

            target_prev = metrics["target_effectiveness"][effec]["swap"]
            target_cur = target_effectiveness[effec]["swap"]
            metrics["target_effectiveness"][effec]["swap"] = (
                target_prev[0] + target_cur[0],
                target_prev[1] + target_cur[1],
            )

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
            pert_imgs = self._perturb_imgs(imgs_A, cloak_imgs)
            torch.set_grad_enabled(False)

            source_swap = self.swap_face(imgs_A, imgs_B)
            pert_source_swap = self.swap_face(pert_imgs, imgs_B)

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
                    "perturb_imgs",
                    "cloak_imgs",
                    "source_swap",
                    "perturb_source_swap",
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
            protection utility: {metric.generate_iter_utility_log(utility)}
            𝒯_identity utility: {metric.generate_iter_utility_log(source_utility)}
            𝒯_identity effectiveness {metric.generate_iter_effectiveness_label(source_effectiveness)}: {metric.generate_iter_effectiveness_log(source_effectiveness, include_labels=False)}
            scores: {metric.generate_iter_score_log(scores)}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            protection utility: {metric.generate_summary_utility_log(metrics, 'utility', idx)}
            𝒯_identity utility: {metric.generate_summary_utility_log(metrics, 'pert_source_utility', idx)}
            𝒯_identity effectiveness {metric.generate_summary_effectiveness_label(metrics, 'pert_source_effectiveness')}: {metric.generate_summary_effectiveness_log(metrics, 'pert_source_effectiveness', include_labels=False)}
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
            self_identity = get_identity(imgs)
            cloak_identity = get_identity(cloak_imgs)

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
                self.config.third_party.defense.limit.identity_id,
            )
            x_id_diff_loss = (
                -self.config.third_party.defense.weight.identity_id * identity_id_diff
            )

            cloak_id_diff_loss = (
                self.config.third_party.defense.weight.cloak_id
                * l2_per_image(x_identity, cloak_identity.detach())
            )

            loss_per_img = (
                pert_diff_loss
                + x_3d_diff_loss
                + cloak_3d_diff_loss
                + x_id_diff_loss
                + cloak_id_diff_loss
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
                    f"{cloak_id_diff_loss.mean().item():.3f}) "
                )

        return best_imgs

    def _free_gpu(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
