from src import metric
from src.cross.base import Base
from src.dataset import FFHQDataset
from src.evaluate import ScoreCalculator
from src.common_utils import save_tensor_imgs

import torch
import textwrap
from torch import tensor, Tensor
from torch.utils.data import DataLoader
from pathlib import Path


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        notes_path = Path(self.config.notes_path)
        notes_path.touch(exist_ok=True)

        self.score_calculator = ScoreCalculator(logger, config)

        self.face_ids = [1, 2, 3, 4, 5, 10, 11, 12, 13]
        self.target_nonface_id = 0

    def metric(self) -> None:
        metrics = metric.get_metric_data_template(self.effectiveness)

        dataset = FFHQDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            total_count += len(imgs_A)
            torch.set_grad_enabled(True)
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            pert_imgs = self._perturb_imgs(imgs_A, cloak_imgs)
            torch.set_grad_enabled(False)

            source_swap = target_swap = pert_source_swap = pert_target_swap = (
                torch.ones_like(pert_imgs)
            )

            if self.config.evaluate.effectiveness.ASRp:
                pert_source_swap = self._face_swap_per_image(pert_imgs, imgs_B)
                pert_target_swap = self._face_swap_per_image(imgs_B, pert_imgs)

            (
                utility,
                source_utility,
                target_utility,
                source_effectiveness,
                target_effectiveness,
            ) = metric.get_defense_metric(
                self.utility,
                self.effectiveness,
                imgs_A,
                imgs_B,
                pert_imgs,
                cloak_imgs,
                None,
                pert_source_swap,
                None,
                pert_target_swap,
            )

            metric.merge_metric(
                self.effectiveness,
                metrics,
                utility,
                source_utility,
                target_utility,
                source_effectiveness,
                target_effectiveness,
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
                    "target\nswap",
                    "perturb\ntarget\nswap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    pert_imgs,
                    cloak_imgs,
                    source_swap,
                    pert_source_swap,
                    target_swap,
                    pert_target_swap,
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )
            del imgs_A, imgs_B, pert_imgs, cloak_imgs
            del (
                source_swap,
                target_swap,
                pert_source_swap,
                pert_target_swap,
            )
            self._free_gpu()

            scores = self.score_calculator.calculate_score(
                source_effectiveness, target_effectiveness, metrics
            )

            iter_log_str = textwrap.dedent(
                f"""
            utility (mse, psnr, ssim, lpips), effectiveness ({', '.join(self.effectiveness.candi_funcs.keys())}), identity ({', '.join(next(iter(source_effectiveness.values())).keys())}), context ({', '.join(next(iter(target_effectiveness.values())).keys())})
            utility: {metric.generate_iter_utility_log(utility)}
            source utility: {metric.generate_iter_utility_log(source_utility)}
            target utility: {metric.generate_iter_utility_log(target_utility)}
            source effectiveness: {metric.generate_iter_effectiveness_log(source_effectiveness)}
            target effectiveness: {metric.generate_iter_effectiveness_log(target_effectiveness)}
            scores: {metric.generate_iter_score_log(scores)}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            utility: {metric.generate_summary_utility_log(metrics, 'utility', idx)}
            source utility: {metric.generate_summary_utility_log(metrics, 'pert_source_utility', idx)}
            target utility: {metric.generate_summary_utility_log(metrics, 'pert_target_utility', idx)}
            source effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'pert_source_effectiveness')}
            target effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'pert_target_effectiveness')}
            scores: {metric.generate_summary_score_log(scores)}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

    def _face_swap_per_image(self, imgs_A: Tensor, imgs_B: Tensor) -> Tensor:
        assert imgs_A.shape[0] == imgs_B.shape[0]

        source_swap = []
        imgs_A = imgs_A.cpu()
        imgs_B = imgs_B.cpu()

        for i in range(imgs_A.size(0)):
            a = imgs_A[i : i + 1].contiguous().to(self.device, non_blocking=True)
            b = imgs_B[i : i + 1].contiguous().to(self.device, non_blocking=True)
            with torch.no_grad():
                out = self.swap_face(a, b)

            source_swap.append(out.detach().cpu())

            del a, b, out

        return torch.cat(source_swap, dim=0).cuda()

    def _perturb_imgs(self, imgs: Tensor, cloak_imgs: Tensor) -> Tensor:
        def l2_per_image(x: Tensor, y: Tensor) -> Tensor:
            return ((x - y) ** 2).view(x.size(0), -1).mean(dim=1)

        x_imgs = imgs.clone().detach() + torch.randn_like(imgs) * 1e-5

        with torch.no_grad():
            simswap_self_identity = self.get_simswap_identity(imgs)
            simswap_cloak_identity = self.get_simswap_identity(cloak_imgs)

            faceshifter_self_identity = self.get_faceshifter_identity(imgs)
            faceshifter_cloak_identity = self.get_faceshifter_identity(cloak_imgs)

            hififace_self_3d = self.net.generator.id_extractor.f_3d(imgs)[:, :80]
            hififace_self_id = self.get_hififace_identity(imgs)
            hififace_cloak_3d = self.net.generator.id_extractor.f_3d(cloak_imgs)[:, :80]
            hififace_cloak_id = self.get_hififace_identity(cloak_imgs)

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

            # simswap loss
            x_simswap_identity = self.get_simswap_identity(x_imgs)

            simswap_identity_diff = torch.clamp(
                l2_per_image(x_simswap_identity, simswap_self_identity),
                0,
                self.config.third_party.defense.limit.simswap,
            )
            simswap_identity_diff_loss = (
                -self.config.third_party.defense.weight.simswap_id
                * simswap_identity_diff
            )

            simswap_cloak_diff_loss = (
                self.config.third_party.defense.weight.simswap_cloak
                * l2_per_image(x_simswap_identity, simswap_cloak_identity)
            )

            # hififace loss
            x_hififace_3d = self.net.generator.id_extractor.f_3d(x_imgs)[:, :80]
            x_hififace_id = self.get_hififace_identity(x_imgs)

            hififace_3d_diff = torch.clamp(
                l2_per_image(x_hififace_3d, hififace_self_3d),
                0,
                self.config.third_party.defense.limit.hififace_3d,
            )
            hififace_3d_diff_loss = (
                -self.config.third_party.defense.weight.hififace_self_3d
                * hififace_3d_diff
            )
            hififace_3d_cloak_loss = (
                self.config.third_party.defense.weight.hififace_cloak_3d
                * l2_per_image(x_hififace_3d, hififace_cloak_3d)
            )

            hififace_id_diff = torch.clamp(
                l2_per_image(x_hififace_id, hififace_self_id),
                0,
                self.config.third_party.defense.limit.hififace_id,
            )
            hififace_id_diff_loss = (
                -self.config.third_party.defense.weight.hififace_self_id
                * hififace_id_diff
            )
            hififace_id_cloak_loss = (
                self.config.third_party.defense.weight.hififace_cloak_id
                * l2_per_image(x_hififace_id, hififace_cloak_id)
            )

            # faceshifter loss
            x_faceshifter_identity = self.get_faceshifter_identity(x_imgs)

            faceshifter_identity_diff = torch.clamp(
                l2_per_image(x_faceshifter_identity, faceshifter_self_identity),
                0,
                self.config.third_party.defense.limit.faceshifter,
            )
            faceshifter_identity_diff_loss = (
                -self.config.third_party.defense.weight.faceshifter_id
                * faceshifter_identity_diff
            )

            faceshifter_cloak_diff_loss = (
                self.config.third_party.defense.weight.faceshifter_cloak
                * l2_per_image(x_faceshifter_identity, faceshifter_cloak_identity)
            )

            loss_per_img = (
                pert_diff_loss
                + simswap_identity_diff_loss
                + simswap_cloak_diff_loss
                + hififace_3d_diff_loss
                + hififace_3d_cloak_loss
                + hififace_id_diff_loss
                + hififace_id_cloak_loss
                + faceshifter_identity_diff_loss
                + faceshifter_cloak_diff_loss
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
                    f"{simswap_identity_diff_loss.mean().item():.3f}, "
                    f"{simswap_cloak_diff_loss.mean().item():.3f}, "
                    f"{hififace_3d_diff_loss.mean().item():.3f}, "
                    f"{hififace_3d_cloak_loss.mean().item():.3f}, "
                    f"{hififace_id_diff_loss.mean().item():.3f}, "
                    f"{hififace_id_cloak_loss.mean().item():.3f}, "
                    f"{faceshifter_identity_diff_loss.mean().item():.3f}, "
                    f"{faceshifter_cloak_diff_loss.mean().item():.3f})"
                )

        return best_imgs

    def _free_gpu(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
