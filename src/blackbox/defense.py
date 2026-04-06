from src import metric
from src.blackbox.base import Base
from src.dataset import FFHQMetric
from src.evaluate import ScoreCalculator
from src.common_utils import save_tensor_imgs

import torch
import textwrap
import torch.nn.functional as F
from torch import tensor, Tensor
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms


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

        dataset_config = self.config.third_party.dataset
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (dataset_config.image_size, dataset_config.image_size)
                ),
                transforms.ToTensor(),
            ]
        )
        dataset = FFHQMetric(
            Path(dataset_config.metric_dir), dataset_config.metric_pairs, transform
        )
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

            if self.config.evaluate.effectiveness.ASRo:
                with torch.no_grad():
                    source_swap = self.swap_face(imgs_A, imgs_B)
                    target_swap = self.swap_face(imgs_B, imgs_A)
            if self.config.evaluate.effectiveness.ASRp:
                with torch.no_grad():
                    pert_source_swap = self.swap_face(pert_imgs, imgs_B)
                    pert_target_swap = self.swap_face(imgs_B, pert_imgs)

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
                source_swap,
                pert_source_swap,
                target_swap,
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

    def swap_face(self, imgs_A: Tensor, imgs_B: Tensor) -> Tensor:
        if self.config.third_party.defense.target == "diffface":
            assert imgs_A.shape[0] == imgs_B.shape[0]

            source_swap = []
            imgs_A = imgs_A.cpu()
            imgs_B = imgs_B.cpu()

            for i in range(imgs_A.size(0)):
                a = imgs_A[i : i + 1].contiguous().to(self.device, non_blocking=True)
                b = imgs_B[i : i + 1].contiguous().to(self.device, non_blocking=True)
                out = self.defense_target.swap_face(a, b)

                source_swap.append(out.detach().cpu())

                del a, b, out

            return torch.cat(source_swap, dim=0).cuda()
        elif self.config.third_party.defense.target == "uniface":
            pass
        elif self.config.third_party.defense.target == "infoswap":
            imgs_A = F.interpolate(
                imgs_A, size=(512, 512), mode="bilinear", align_corners=False
            )
            imgs_B = F.interpolate(
                imgs_B, size=(512, 512), mode="bilinear", align_corners=False
            )
        elif self.config.third_party.defense.target == "e4s":
            imgs_A = F.interpolate(
                imgs_A, size=(1024, 1024), mode="bilinear", align_corners=False
            )
            imgs_B = F.interpolate(
                imgs_B, size=(1024, 1024), mode="bilinear", align_corners=False
            )
        else:
            raise ValueError(
                f"Unsupported defense target: {self.config.third_party.defense.target}"
            )

        imgs_A = imgs_A * 2 - 1
        imgs_B = imgs_B * 2 - 1

        out = self.defense_target.swap_face(imgs_A, imgs_B)
        out = ((out + 1) / 2).clamp(0, 1)
        out = F.interpolate(out, size=(256, 256), mode="bilinear", align_corners=False)

        return out

    def _perturb_imgs(self, imgs: Tensor, cloak_imgs: Tensor) -> Tensor:
        def l2_per_image(x: Tensor, y: Tensor) -> Tensor:
            return ((x - y) ** 2).view(x.size(0), -1).mean(dim=1)

        def get_relative_progress(
            current_value: Tensor,
            initial_value: Tensor,
            target_value: Tensor,
        ) -> Tensor:
            denom = torch.clamp(target_value - initial_value, min=dynamic_weight_epsilon)
            progress = (current_value - initial_value) / denom
            return torch.clamp(progress, min=0.0, max=1.0)

        def update_progress_stats(
            progress_ema: Tensor,
            progress_sq_ema: Tensor,
            progress: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            detached_progress = progress.detach()
            progress_ema = (
                dynamic_weight_progress_ema_decay * progress_ema
                + (1 - dynamic_weight_progress_ema_decay) * detached_progress
            )
            progress_sq_ema = (
                dynamic_weight_progress_ema_decay * progress_sq_ema
                + (1 - dynamic_weight_progress_ema_decay) * (detached_progress**2)
            )
            progress_var = torch.clamp(
                progress_sq_ema - progress_ema**2,
                min=0.0,
            )

            return progress_ema, progress_sq_ema, progress_var

        def get_dynamic_weights(
            progresses: list[Tensor],
            progress_vars: list[Tensor],
        ) -> list[Tensor]:
            variance_terms = [
                torch.sqrt(torch.clamp(progress_var, min=0.0))
                for progress_var in progress_vars
            ]
            variance_term_sum = torch.stack(variance_terms, dim=0).sum(dim=0)
            normalized_variance_terms = [
                variance_term / torch.clamp(
                    variance_term_sum,
                    min=dynamic_weight_epsilon,
                )
                for variance_term in variance_terms
            ]
            task_count = len(progresses)
            raw_weights = [
                torch.clamp(1.0 - progress.detach(), min=dynamic_weight_epsilon)
                + dynamic_weight_variance_weight * task_count * normalized_variance_term
                for progress, normalized_variance_term in zip(
                    progresses, normalized_variance_terms
                )
            ]
            raw_weight_sum = torch.stack(raw_weights, dim=0).sum(dim=0)

            return [
                task_count
                * raw_weight
                / torch.clamp(raw_weight_sum, min=dynamic_weight_epsilon)
                for raw_weight in raw_weights
            ]

        def get_utility_mse(x: Tensor, y: Tensor) -> Tensor:
            return l2_per_image(x, y) * (255.0**2)

        def get_perturb_weight_scale(utility_mse: Tensor) -> Tensor:
            lower_bound = self.config.third_party.defense.utility_mse_lower_bound
            upper_bound = self.config.third_party.defense.utility_mse_upper_bound
            smoothness = self.config.third_party.defense.utility_mse_smoothness

            lower_gate = torch.sigmoid((utility_mse - lower_bound) / smoothness)
            upper_gate = torch.sigmoid((utility_mse - upper_bound) / smoothness)

            return torch.clamp(
                lower_gate + upper_gate,
                min=0.0,
                max=2.0,
            )

        x_imgs = imgs.clone().detach() + torch.randn_like(imgs) * 1e-5
        B = imgs.size(0)
        perturb_mse_ema = get_utility_mse(x_imgs, imgs.detach()).detach()
        perturb_mse_ema_decay = self.config.third_party.defense.utility_mse_ema_decay
        dynamic_weight_epsilon = self.config.third_party.defense.dynamic_weight_epsilon
        dynamic_weight_progress_ema_decay = (
            self.config.third_party.defense.dynamic_weight_progress_ema_decay
        )
        dynamic_weight_variance_weight = (
            self.config.third_party.defense.dynamic_weight_variance_weight
        )

        with torch.no_grad():
            simswap_self_identity = self.get_simswap_identity(imgs)
            simswap_cloak_identity = self.get_simswap_identity(cloak_imgs)

            faceshifter_self_identity = self.get_faceshifter_identity(imgs)
            faceshifter_cloak_identity = self.get_faceshifter_identity(cloak_imgs)

            hififace_self_3d = self.net.generator.id_extractor.f_3d(imgs)[:, :80]
            hififace_self_id = self.get_hififace_identity(imgs)
            hififace_cloak_3d = self.net.generator.id_extractor.f_3d(cloak_imgs)[:, :80]
            hififace_cloak_id = self.get_hififace_identity(cloak_imgs)

            simswap_initial_push = l2_per_image(simswap_self_identity, simswap_self_identity)
            simswap_target_push = l2_per_image(
                simswap_self_identity, simswap_cloak_identity
            )
            simswap_initial_pull = simswap_target_push

            hififace_3d_initial_push = l2_per_image(hififace_self_3d, hififace_self_3d)
            hififace_3d_target_push = l2_per_image(hififace_self_3d, hififace_cloak_3d)
            hififace_3d_initial_pull = hififace_3d_target_push

            hififace_id_initial_push = l2_per_image(hififace_self_id, hififace_self_id)
            hififace_id_target_push = l2_per_image(hififace_self_id, hififace_cloak_id)
            hififace_id_initial_pull = hififace_id_target_push

            faceshifter_initial_push = l2_per_image(
                faceshifter_self_identity, faceshifter_self_identity
            )
            faceshifter_target_push = l2_per_image(
                faceshifter_self_identity, faceshifter_cloak_identity
            )
            faceshifter_initial_pull = faceshifter_target_push

        push_progress_ema = {
            "simswap": torch.zeros(B, device=imgs.device),
            "hififace_3d": torch.zeros(B, device=imgs.device),
            "hififace_id": torch.zeros(B, device=imgs.device),
            "faceshifter": torch.zeros(B, device=imgs.device),
        }
        push_progress_sq_ema = {
            name: torch.zeros_like(value) for name, value in push_progress_ema.items()
        }
        pull_progress_ema = {
            "simswap": torch.zeros(B, device=imgs.device),
            "hififace_3d": torch.zeros(B, device=imgs.device),
            "hififace_id": torch.zeros(B, device=imgs.device),
            "faceshifter": torch.zeros(B, device=imgs.device),
        }
        pull_progress_sq_ema = {
            name: torch.zeros_like(value) for name, value in pull_progress_ema.items()
        }

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

        best_imgs = imgs.clone()
        best_loss = torch.full((B,), float("inf"), device=imgs.device)

        for epoch in range(self.config.third_party.defense.epochs):
            x_imgs = x_imgs.clone().detach().requires_grad_(True)

            perturb_l2 = l2_per_image(x_imgs, imgs.detach())
            perturb_mse = get_utility_mse(x_imgs, imgs.detach())
            perturb_mse_ema = (
                perturb_mse_ema_decay * perturb_mse_ema
                + (1 - perturb_mse_ema_decay) * perturb_mse.detach()
            )
            perturb_weight_scale = get_perturb_weight_scale(perturb_mse_ema)
            pert_diff_loss = (
                self.config.third_party.defense.weight.perturb
                * perturb_weight_scale
                * perturb_l2
            )

            # simswap loss
            x_simswap_identity = self.get_simswap_identity(x_imgs)

            simswap_identity_raw_diff = l2_per_image(
                x_simswap_identity, simswap_self_identity
            )
            simswap_identity_diff = torch.clamp(
                simswap_identity_raw_diff,
                0,
                self.config.third_party.defense.limit.simswap,
            )
            simswap_identity_diff_loss = (
                -self.config.third_party.defense.weight.simswap_id
                * simswap_identity_diff
            )

            simswap_cloak_diff = l2_per_image(x_simswap_identity, simswap_cloak_identity)

            # hififace loss
            x_hififace_3d = self.net.generator.id_extractor.f_3d(x_imgs)[:, :80]
            x_hififace_id = self.get_hififace_identity(x_imgs)

            hififace_3d_raw_diff = l2_per_image(x_hififace_3d, hififace_self_3d)
            hififace_3d_diff = torch.clamp(
                hififace_3d_raw_diff,
                0,
                self.config.third_party.defense.limit.hififace_3d,
            )
            hififace_3d_diff_loss = (
                -self.config.third_party.defense.weight.hififace_self_3d
                * hififace_3d_diff
            )
            hififace_3d_cloak_diff = l2_per_image(x_hififace_3d, hififace_cloak_3d)

            hififace_id_raw_diff = l2_per_image(x_hififace_id, hififace_self_id)
            hififace_id_diff = torch.clamp(
                hififace_id_raw_diff,
                0,
                self.config.third_party.defense.limit.hififace_id,
            )
            hififace_id_diff_loss = (
                -self.config.third_party.defense.weight.hififace_self_id
                * hififace_id_diff
            )
            hififace_id_cloak_diff = l2_per_image(x_hififace_id, hififace_cloak_id)

            # faceshifter loss
            x_faceshifter_identity = self.get_faceshifter_identity(x_imgs)

            faceshifter_identity_raw_diff = l2_per_image(
                x_faceshifter_identity, faceshifter_self_identity
            )
            faceshifter_identity_diff = torch.clamp(
                faceshifter_identity_raw_diff,
                0,
                self.config.third_party.defense.limit.faceshifter,
            )
            faceshifter_identity_diff_loss = (
                -self.config.third_party.defense.weight.faceshifter_id
                * faceshifter_identity_diff
            )

            faceshifter_cloak_diff = l2_per_image(
                x_faceshifter_identity, faceshifter_cloak_identity
            )

            simswap_push_progress = get_relative_progress(
                simswap_identity_raw_diff,
                simswap_initial_push,
                simswap_target_push,
            )
            hififace_3d_push_progress = get_relative_progress(
                hififace_3d_raw_diff,
                hififace_3d_initial_push,
                hififace_3d_target_push,
            )
            hififace_id_push_progress = get_relative_progress(
                hififace_id_raw_diff,
                hififace_id_initial_push,
                hififace_id_target_push,
            )
            faceshifter_push_progress = get_relative_progress(
                faceshifter_identity_raw_diff,
                faceshifter_initial_push,
                faceshifter_target_push,
            )
            (
                push_progress_ema["simswap"],
                push_progress_sq_ema["simswap"],
                simswap_push_progress_var,
            ) = update_progress_stats(
                push_progress_ema["simswap"],
                push_progress_sq_ema["simswap"],
                simswap_push_progress,
            )
            (
                push_progress_ema["hififace_3d"],
                push_progress_sq_ema["hififace_3d"],
                hififace_3d_push_progress_var,
            ) = update_progress_stats(
                push_progress_ema["hififace_3d"],
                push_progress_sq_ema["hififace_3d"],
                hififace_3d_push_progress,
            )
            (
                push_progress_ema["hififace_id"],
                push_progress_sq_ema["hififace_id"],
                hififace_id_push_progress_var,
            ) = update_progress_stats(
                push_progress_ema["hififace_id"],
                push_progress_sq_ema["hififace_id"],
                hififace_id_push_progress,
            )
            (
                push_progress_ema["faceshifter"],
                push_progress_sq_ema["faceshifter"],
                faceshifter_push_progress_var,
            ) = update_progress_stats(
                push_progress_ema["faceshifter"],
                push_progress_sq_ema["faceshifter"],
                faceshifter_push_progress,
            )
            (
                simswap_push_weight_scale,
                hififace_3d_push_weight_scale,
                hififace_id_push_weight_scale,
                faceshifter_push_weight_scale,
            ) = get_dynamic_weights(
                [
                    simswap_push_progress,
                    hififace_3d_push_progress,
                    hififace_id_push_progress,
                    faceshifter_push_progress,
                ],
                [
                    simswap_push_progress_var,
                    hififace_3d_push_progress_var,
                    hififace_id_push_progress_var,
                    faceshifter_push_progress_var,
                ],
            )

            simswap_pull_progress = 1.0 - get_relative_progress(
                simswap_cloak_diff,
                simswap_initial_push,
                simswap_initial_pull,
            )
            hififace_3d_pull_progress = 1.0 - get_relative_progress(
                hififace_3d_cloak_diff,
                hififace_3d_initial_push,
                hififace_3d_initial_pull,
            )
            hififace_id_pull_progress = 1.0 - get_relative_progress(
                hififace_id_cloak_diff,
                hififace_id_initial_push,
                hififace_id_initial_pull,
            )
            faceshifter_pull_progress = 1.0 - get_relative_progress(
                faceshifter_cloak_diff,
                faceshifter_initial_push,
                faceshifter_initial_pull,
            )
            (
                pull_progress_ema["simswap"],
                pull_progress_sq_ema["simswap"],
                simswap_pull_progress_var,
            ) = update_progress_stats(
                pull_progress_ema["simswap"],
                pull_progress_sq_ema["simswap"],
                simswap_pull_progress,
            )
            (
                pull_progress_ema["hififace_3d"],
                pull_progress_sq_ema["hififace_3d"],
                hififace_3d_pull_progress_var,
            ) = update_progress_stats(
                pull_progress_ema["hififace_3d"],
                pull_progress_sq_ema["hififace_3d"],
                hififace_3d_pull_progress,
            )
            (
                pull_progress_ema["hififace_id"],
                pull_progress_sq_ema["hififace_id"],
                hififace_id_pull_progress_var,
            ) = update_progress_stats(
                pull_progress_ema["hififace_id"],
                pull_progress_sq_ema["hififace_id"],
                hififace_id_pull_progress,
            )
            (
                pull_progress_ema["faceshifter"],
                pull_progress_sq_ema["faceshifter"],
                faceshifter_pull_progress_var,
            ) = update_progress_stats(
                pull_progress_ema["faceshifter"],
                pull_progress_sq_ema["faceshifter"],
                faceshifter_pull_progress,
            )
            (
                simswap_cloak_weight_scale,
                hififace_3d_cloak_weight_scale,
                hififace_id_cloak_weight_scale,
                faceshifter_cloak_weight_scale,
            ) = get_dynamic_weights(
                [
                    simswap_pull_progress,
                    hififace_3d_pull_progress,
                    hififace_id_pull_progress,
                    faceshifter_pull_progress,
                ],
                [
                    simswap_pull_progress_var,
                    hififace_3d_pull_progress_var,
                    hififace_id_pull_progress_var,
                    faceshifter_pull_progress_var,
                ],
            )

            simswap_identity_diff_loss = (
                simswap_push_weight_scale * simswap_identity_diff_loss
            )
            simswap_cloak_diff_loss = (
                self.config.third_party.defense.weight.simswap_cloak
                * simswap_cloak_weight_scale
                * simswap_cloak_diff
            )
            hififace_3d_diff_loss = (
                hififace_3d_push_weight_scale * hififace_3d_diff_loss
            )
            hififace_3d_cloak_loss = (
                self.config.third_party.defense.weight.hififace_cloak_3d
                * hififace_3d_cloak_weight_scale
                * hififace_3d_cloak_diff
            )
            hififace_id_diff_loss = (
                hififace_id_push_weight_scale * hififace_id_diff_loss
            )
            hififace_id_cloak_loss = (
                self.config.third_party.defense.weight.hififace_cloak_id
                * hififace_id_cloak_weight_scale
                * hififace_id_cloak_diff
            )
            faceshifter_identity_diff_loss = (
                faceshifter_push_weight_scale * faceshifter_identity_diff_loss
            )
            faceshifter_cloak_diff_loss = (
                self.config.third_party.defense.weight.faceshifter_cloak
                * faceshifter_cloak_weight_scale
                * faceshifter_cloak_diff
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
                    f"{faceshifter_cloak_diff_loss.mean().item():.3f}) "
                    f"perturb_mse: {perturb_mse.mean().item():.3f}, "
                    f"perturb_mse_ema: {perturb_mse_ema.mean().item():.3f}, "
                    f"perturb_weight_scale: {perturb_weight_scale.mean().item():.3f}, "
                    f"push_progress: ("
                    f"{simswap_push_progress.mean().item():.3f}, "
                    f"{hififace_3d_push_progress.mean().item():.3f}, "
                    f"{hififace_id_push_progress.mean().item():.3f}, "
                    f"{faceshifter_push_progress.mean().item():.3f}), "
                    f"push_progress_var: ("
                    f"{simswap_push_progress_var.mean().item():.5f}, "
                    f"{hififace_3d_push_progress_var.mean().item():.5f}, "
                    f"{hififace_id_push_progress_var.mean().item():.5f}, "
                    f"{faceshifter_push_progress_var.mean().item():.5f}), "
                    f"push_weight_scale: ("
                    f"{simswap_push_weight_scale.mean().item():.3f}, "
                    f"{hififace_3d_push_weight_scale.mean().item():.3f}, "
                    f"{hififace_id_push_weight_scale.mean().item():.3f}, "
                    f"{faceshifter_push_weight_scale.mean().item():.3f}), "
                    f"pull_progress: ("
                    f"{simswap_pull_progress.mean().item():.3f}, "
                    f"{hififace_3d_pull_progress.mean().item():.3f}, "
                    f"{hififace_id_pull_progress.mean().item():.3f}, "
                    f"{faceshifter_pull_progress.mean().item():.3f}), "
                    f"pull_progress_var: ("
                    f"{simswap_pull_progress_var.mean().item():.5f}, "
                    f"{hififace_3d_pull_progress_var.mean().item():.5f}, "
                    f"{hififace_id_pull_progress_var.mean().item():.5f}, "
                    f"{faceshifter_pull_progress_var.mean().item():.5f}), "
                    f"pull_weight_scale: ("
                    f"{simswap_cloak_weight_scale.mean().item():.3f}, "
                    f"{hififace_3d_cloak_weight_scale.mean().item():.3f}, "
                    f"{hififace_id_cloak_weight_scale.mean().item():.3f}, "
                    f"{faceshifter_cloak_weight_scale.mean().item():.3f})"
                )

        return best_imgs

    def _free_gpu(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
