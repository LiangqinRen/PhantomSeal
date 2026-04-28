from src import metric
from src.blackbox.base import Base
from src.dataset import FFHQMetric
from src.evaluate import ScoreCalculator
from src.common_utils import save_tensor_imgs

import torch
import textwrap
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
        target_names = self.get_eval_target_names()
        metrics_by_target = {
            target_name: metric.get_metric_data_template(self.effectiveness)
            for target_name in target_names
        }
        utility_total = (0.0, 0.0, 0.0, 0.0)

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
            enable_tsr = (
                self.protection_method == "phantomseal"
                and self.config.evaluate.effectiveness.TSR
            )

            if self.protection_method == "phantomseal":
                cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
                pert_imgs = self._perturb_imgs(imgs_A, cloak_imgs)
                cloak_label = "cloak_imgs"
            elif self.protection_method == "nullswap":
                pert_imgs = self._perturb_imgs(imgs_A, None)
                cloak_imgs = pert_imgs
                cloak_label = "nullswap_imgs"
            else:
                raise ValueError(
                    f"Unsupported blackbox protection method: {self.protection_method}"
                )
            torch.set_grad_enabled(False)
            utility = self.utility.calculate_utility(imgs_A, pert_imgs)

            summary_image_labels = [
                "imgs_A",
                "imgs_B",
                "perturb_imgs",
                cloak_label,
            ]
            summary_image_tensors = [
                imgs_A,
                imgs_B,
                pert_imgs,
                cloak_imgs,
            ]
            per_target_results = {}

            for target_name in target_names:
                source_swap = pert_source_swap = torch.ones_like(pert_imgs)

                if self.config.evaluate.effectiveness.ASRo:
                    source_swap = self._face_swap_per_image(imgs_A, imgs_B, target_name)
                if (
                    self.config.evaluate.effectiveness.ASRp
                    or self.config.evaluate.effectiveness.TSR
                ):
                    pert_source_swap = self._face_swap_per_image(
                        pert_imgs, imgs_B, target_name
                    )

                source_effectiveness = self.effectiveness.calculate_effectiveness(
                    imgs_A,
                    pert_imgs,
                    source_swap if self.config.evaluate.effectiveness.ASRo else None,
                    pert_source_swap,
                    cloak_imgs if enable_tsr else None,
                )

                metric.merge_metric(
                    self.effectiveness,
                    metrics_by_target[target_name],
                    None,
                    None,
                    None,
                    source_effectiveness,
                    None,
                )
                per_target_results[target_name] = (
                    source_effectiveness,
                )
                summary_image_labels.extend(
                    [
                        f"{target_name}_source_swap",
                        f"{target_name}_perturb_source_swap",
                    ]
                )
                summary_image_tensors.extend(
                    [
                        source_swap,
                        pert_source_swap,
                    ]
                )

            save_tensor_imgs(
                self.image_dir,
                idx,
                summary_image_labels,
                summary_image_tensors,
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )
            utility_total = tuple(
                x + y
                for x, y in zip(
                    utility_total,
                    (
                        utility["mse"],
                        utility["psnr"],
                        utility["ssim"],
                        utility["lpips"],
                    ),
                )
            )
            del imgs_A, imgs_B, pert_imgs, cloak_imgs
            del summary_image_labels, summary_image_tensors
            self._free_gpu()

            iter_parts = [
                f"Batch {idx:4}/{len(dataloader):4}",
                "utility: "
                f"mse_255={utility['mse']:.3f}, "
                f"psnr={utility['psnr']:.3f}, "
                f"ssim={utility['ssim']:.3f}, "
                f"lpips={utility['lpips']:.3f}",
            ]
            summary_parts = [
                f"Total {total_count} pairs",
                "utility: "
                f"mse_255={utility_total[0] / idx:.3f}, "
                f"psnr={utility_total[1] / idx:.3f}, "
                f"ssim={utility_total[2] / idx:.3f}, "
                f"lpips={utility_total[3] / idx:.3f}",
            ]
            if idx == 1:
                effect_labels = []
                if self.config.evaluate.effectiveness.ASRo:
                    effect_labels.append("swap")
                if self.config.evaluate.effectiveness.ASRp:
                    effect_labels.append("pert_swap")
                if enable_tsr:
                    effect_labels.append("cloak")
                effect_tools = []
                if self.config.evaluate.facenet_512.enable:
                    effect_tools.append(
                        f"facenet_512(threshold={self.config.evaluate.facenet_512.threshold:.2f})"
                    )
                if self.config.evaluate.face_recognition.enable:
                    effect_tools.append("face_recognition")
                if self.config.evaluate.facepp.enable:
                    effect_tools.append("face++")
                if self.config.evaluate.aws.enable:
                    effect_tools.append("aws_rekognition")
                if effect_labels:
                    effect_explain = (
                        f"protection: {self.protection_method}, "
                        "effectiveness tuple order: "
                        f"({', '.join(effect_labels)}), "
                        f"tool: {', '.join(effect_tools) if effect_tools else 'none'}"
                    )
                else:
                    effect_explain = (
                        f"protection: {self.protection_method}, "
                        "effectiveness tuple order: (), "
                        f"tool: {', '.join(effect_tools) if effect_tools else 'none'}"
                    )
                iter_parts.append(effect_explain)
                summary_parts.append(effect_explain)
            for target_name in target_names:
                (source_effectiveness,) = per_target_results[target_name]
                scores = self.score_calculator.calculate_score(
                    source_effectiveness,
                    None,
                    metrics_by_target[target_name],
                )
                iter_parts.append(
                    f"{target_name}: {metric.generate_iter_effectiveness_log(source_effectiveness)} score {metric.generate_iter_score_log(scores)}"
                )
                summary_parts.append(
                    f"{target_name}: {metric.generate_summary_effectiveness_log(metrics_by_target[target_name], 'pert_source_effectiveness')} score {metric.generate_summary_score_log(scores)}"
                )

            iter_log_str = "\n    ".join(iter_parts)
            summary_log_str = "\n    ".join(summary_parts)
            self.logger.info(f"\n    {iter_log_str}")
            self.logger.info(f"\n    {summary_log_str}")

    def _face_swap_per_image(self, imgs_A: Tensor, imgs_B: Tensor, target_name: str) -> Tensor:
        assert imgs_A.shape[0] == imgs_B.shape[0]

        source_swap = []
        imgs_A = imgs_A.cpu()
        imgs_B = imgs_B.cpu()

        for i in range(imgs_A.size(0)):
            a = imgs_A[i : i + 1].contiguous().to(self.device, non_blocking=True)
            b = imgs_B[i : i + 1].contiguous().to(self.device, non_blocking=True)
            with torch.no_grad():
                out = self.swap_face(a, b, target_name)

            source_swap.append(out.detach().cpu())

            del a, b, out

        return torch.cat(source_swap, dim=0).cuda()

    def _perturb_imgs(self, imgs: Tensor, cloak_imgs: Tensor | None) -> Tensor:
        if self.protection_method == "nullswap":
            return self.protect_with_nullswap(imgs)

        if self.protection_method != "phantomseal":
            raise ValueError(
                f"Unsupported blackbox protection method: {self.protection_method}"
            )
        if cloak_imgs is None:
            raise ValueError("PhantomSeal protection requires cloak images")

        def l2_per_image(x: Tensor, y: Tensor) -> Tensor:
            return ((x - y) ** 2).view(x.size(0), -1).mean(dim=1)

        def weighted_l2_per_image(x: Tensor, y: Tensor, weight_map: Tensor) -> Tensor:
            return (((x - y) ** 2) * weight_map).view(x.size(0), -1).mean(dim=1)

        def total_variation_per_image(x: Tensor, y: Tensor) -> Tensor:
            delta = x - y
            tv_h = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
            tv_w = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
            return tv_h + tv_w

        def get_relative_progress(
            current_value: Tensor,
            initial_value: Tensor,
            target_value: Tensor,
        ) -> Tensor:
            denom = torch.clamp(target_value - initial_value, min=dynamic_weight_epsilon)
            progress = (current_value - initial_value) / denom
            return torch.clamp(progress, min=0.0, max=1.0)

        def get_reverse_relative_progress(
            current_value: Tensor,
            initial_value: Tensor,
            target_value: Tensor,
        ) -> Tensor:
            denom = torch.clamp(initial_value - target_value, min=dynamic_weight_epsilon)
            progress = (initial_value - current_value) / denom
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
            progress_var = torch.clamp(progress_sq_ema - progress_ema**2, min=0.0)

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
                variance_term
                / torch.clamp(variance_term_sum, min=dynamic_weight_epsilon)
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
            normalized_weights = [
                task_count
                * raw_weight
                / torch.clamp(raw_weight_sum, min=dynamic_weight_epsilon)
                for raw_weight in raw_weights
            ]
            return [
                dynamic_weight_strength * weight
                for weight in normalized_weights
            ]

        def mean_item(value: Tensor | float) -> float:
            if isinstance(value, Tensor):
                return float(value.mean().item())
            return float(value)

        def format_loss_log(
            epoch_label: int,
            total_loss_value: Tensor,
            perturb_raw_value: Tensor,
            perturb_loss_value: Tensor,
            background_raw_value: Tensor,
            background_loss_value: Tensor,
            structure_raw_value: Tensor,
            structure_loss_value: Tensor,
            tv_raw_value: Tensor,
            tv_loss_value: Tensor,
            arcface_push_raw_value: Tensor,
            arcface_push_clamped_value: Tensor,
            arcface_push_weight_scale_value: Tensor,
            arcface_push_loss_value: Tensor,
            arcface_cloak_raw_value: Tensor,
            arcface_cloak_target_value: Tensor,
            arcface_cloak_excess_value: Tensor,
            arcface_cloak_weight_scale_value: Tensor,
            arcface_cloak_loss_value: Tensor,
            facenet_push_raw_value: Tensor,
            facenet_push_clamped_value: Tensor,
            facenet_push_weight_scale_value: Tensor,
            facenet_push_loss_value: Tensor,
            facenet_cloak_raw_value: Tensor,
            facenet_cloak_target_value: Tensor,
            facenet_cloak_excess_value: Tensor,
            facenet_cloak_weight_scale_value: Tensor,
            facenet_cloak_loss_value: Tensor,
        ) -> str:
            return (
                f"[Epoch {epoch_label:4}/{self.config.third_party.defense.epochs:4}] "
                f"total_loss: {mean_item(total_loss_value):.3f}("
                f"{mean_item(perturb_loss_value):.3f}, "
                f"{mean_item(background_loss_value):.3f}, "
                f"{mean_item(structure_loss_value):.3f}, "
                f"{mean_item(tv_loss_value):.3f}, "
                f"{mean_item(arcface_push_loss_value):.3f}, "
                f"{mean_item(arcface_cloak_loss_value):.3f}, "
                f"{mean_item(facenet_push_loss_value):.3f}, "
                f"{mean_item(facenet_cloak_loss_value):.3f})\n"
                f"    perturb: {mean_item(perturb_loss_value):.3f} "
                f"({mean_item(perturb_raw_value):.6f} * "
                f"{float(self.config.third_party.defense.weight.perturb):.3f})\n"
                f"    background: {mean_item(background_loss_value):.3f} "
                f"({mean_item(background_raw_value):.6f} * "
                f"{float(self.config.third_party.defense.weight.background):.3f})\n"
                f"    structure: {mean_item(structure_loss_value):.3f} "
                f"({mean_item(structure_raw_value):.6f} * "
                f"{float(self.config.third_party.defense.weight.structure):.3f})\n"
                f"    tv: {mean_item(tv_loss_value):.3f} "
                f"({mean_item(tv_raw_value):.6f} * "
                f"{float(self.config.third_party.defense.weight.tv):.3f})\n"
                f"    arcface_push: {mean_item(arcface_push_loss_value):.3f} "
                f"(min({mean_item(arcface_push_raw_value):.6f}, "
                f"limit {float(self.config.third_party.defense.limit.arcface):.6f})="
                f"{mean_item(arcface_push_clamped_value):.6f} * "
                f"-{float(self.config.third_party.defense.weight.arcface_push):.3f} * "
                f"dyn {mean_item(arcface_push_weight_scale_value):.3f})\n"
                f"    arcface_cloak: {mean_item(arcface_cloak_loss_value):.3f} "
                f"(max({mean_item(arcface_cloak_raw_value):.6f} - "
                f"target {mean_item(arcface_cloak_target_value):.6f}, 0)="
                f"{mean_item(arcface_cloak_excess_value):.6f} * "
                f"{float(self.config.third_party.defense.weight.arcface_cloak):.3f} * "
                f"dyn {mean_item(arcface_cloak_weight_scale_value):.3f})\n"
                f"    facenet_push: {mean_item(facenet_push_loss_value):.3f} "
                f"(min({mean_item(facenet_push_raw_value):.6f}, "
                f"limit {float(self.config.third_party.defense.limit.facenet):.6f})="
                f"{mean_item(facenet_push_clamped_value):.6f} * "
                f"-{float(self.config.third_party.defense.weight.facenet_push):.3f} * "
                f"dyn {mean_item(facenet_push_weight_scale_value):.3f})\n"
                f"    facenet_cloak: {mean_item(facenet_cloak_loss_value):.3f} "
                f"(max({mean_item(facenet_cloak_raw_value):.6f} - "
                f"target {mean_item(facenet_cloak_target_value):.6f}, 0)="
                f"{mean_item(facenet_cloak_excess_value):.6f} * "
                f"{float(self.config.third_party.defense.weight.facenet_cloak):.3f} * "
                f"dyn {mean_item(facenet_cloak_weight_scale_value):.3f})\n"
                f"    structure_mask: {structure_mask.mean().item():.3f}"
            )

        x_imgs = imgs.clone().detach() + torch.randn_like(imgs) * 1e-5

        with torch.no_grad():
            clean_arcface = self.get_arcface_embedding(imgs)
            cloak_arcface = self.get_arcface_embedding(cloak_imgs)
            clean_facenet = self.get_facenet_embedding(imgs)
            cloak_facenet = self.get_facenet_embedding(cloak_imgs)
            arcface_initial_push = l2_per_image(clean_arcface, clean_arcface)
            arcface_target_push = l2_per_image(clean_arcface, cloak_arcface)
            facenet_initial_push = l2_per_image(clean_facenet, clean_facenet)
            facenet_target_push = l2_per_image(clean_facenet, cloak_facenet)
            cloak_target_ratio = float(
                getattr(self.config.third_party.defense, "cloak_target_ratio", 0.4)
            )
            if not 0.0 <= cloak_target_ratio <= 1.0:
                raise ValueError(
                    "cross defense cloak_target_ratio must be in [0, 1]"
                )
            arcface_target_cloak_diff = arcface_target_push * cloak_target_ratio
            facenet_target_cloak_diff = facenet_target_push * cloak_target_ratio
            clean_parse_logits = self.get_face_parse_logits(imgs)
            face_mask = self.get_face_region_mask(clean_parse_logits)
            feature_mask = self.get_feature_region_mask(clean_parse_logits)
            structure_mask = torch.clamp(face_mask + feature_mask, 0.0, 1.0)
            face_only_mask = torch.clamp(face_mask - feature_mask, 0.0, 1.0)
            non_face_mask = 1.0 - face_mask

        region_outside = float(
            getattr(self.config.third_party.defense, "region_weight_outside", 3.0)
        )
        region_face = float(
            getattr(self.config.third_party.defense, "region_weight_face", 1.5)
        )
        region_feature = float(
            getattr(self.config.third_party.defense, "region_weight_feature", 0.75)
        )
        perturb_weight_map = (
            region_outside * non_face_mask
            + region_face * face_only_mask
            + region_feature * feature_mask
        )
        dynamic_weight_epsilon = float(
            getattr(self.config.third_party.defense, "dynamic_weight_epsilon", 1.0e-6)
        )
        dynamic_weight_progress_ema_decay = float(
            getattr(
                self.config.third_party.defense,
                "dynamic_weight_progress_ema_decay",
                0.9,
            )
        )
        dynamic_weight_variance_weight = float(
            getattr(
                self.config.third_party.defense,
                "dynamic_weight_variance_weight",
                0.5,
            )
        )
        dynamic_weight_strength = float(
            getattr(
                self.config.third_party.defense,
                "dynamic_weight_strength",
                1.0,
            )
        )
        if dynamic_weight_strength < 0:
            raise ValueError(
                "cross defense dynamic_weight_strength must be non-negative"
            )

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
        push_progress_ema = {
            "arcface": torch.zeros(B, device=imgs.device),
            "facenet": torch.zeros(B, device=imgs.device),
        }
        push_progress_sq_ema = {
            name: torch.zeros_like(value) for name, value in push_progress_ema.items()
        }
        pull_progress_ema = {
            "arcface": torch.zeros(B, device=imgs.device),
            "facenet": torch.zeros(B, device=imgs.device),
        }
        pull_progress_sq_ema = {
            name: torch.zeros_like(value) for name, value in pull_progress_ema.items()
        }

        if not self.config.third_party.defense.silent_perturb:
            with torch.no_grad():
                baseline_imgs = imgs.detach()
                baseline_pert_diff_loss = (
                    self.config.third_party.defense.weight.perturb
                    * weighted_l2_per_image(
                        baseline_imgs, imgs.detach(), perturb_weight_map
                    )
                )
                baseline_pert_diff_raw = weighted_l2_per_image(
                    baseline_imgs, imgs.detach(), perturb_weight_map
                )
                baseline_background_loss = (
                    self.config.third_party.defense.weight.background
                    * l2_per_image(
                        baseline_imgs * non_face_mask,
                        imgs.detach() * non_face_mask,
                    )
                )
                baseline_background_raw = l2_per_image(
                    baseline_imgs * non_face_mask,
                    imgs.detach() * non_face_mask,
                )
                baseline_structure_raw = weighted_l2_per_image(
                    baseline_imgs, imgs.detach(), structure_mask
                )
                baseline_structure_loss = (
                    self.config.third_party.defense.weight.structure
                    * baseline_structure_raw
                )
                baseline_tv_raw = total_variation_per_image(
                    baseline_imgs, imgs.detach()
                )
                baseline_tv_loss = (
                    self.config.third_party.defense.weight.tv * baseline_tv_raw
                )

                baseline_arcface = self.get_arcface_embedding(baseline_imgs)
                baseline_arcface_push_raw = l2_per_image(
                    baseline_arcface, clean_arcface
                )
                baseline_arcface_push = torch.clamp(
                    baseline_arcface_push_raw,
                    0,
                    self.config.third_party.defense.limit.arcface,
                )
                baseline_arcface_cloak_diff = l2_per_image(
                    baseline_arcface, cloak_arcface
                )
                baseline_arcface_cloak_excess = torch.clamp(
                    baseline_arcface_cloak_diff - arcface_target_cloak_diff,
                    min=0.0,
                )

                baseline_facenet = self.get_facenet_embedding(baseline_imgs)
                baseline_facenet_push_raw = l2_per_image(
                    baseline_facenet, clean_facenet
                )
                baseline_facenet_push = torch.clamp(
                    baseline_facenet_push_raw,
                    0,
                    self.config.third_party.defense.limit.facenet,
                )
                baseline_facenet_cloak_diff = l2_per_image(
                    baseline_facenet, cloak_facenet
                )
                baseline_facenet_cloak_excess = torch.clamp(
                    baseline_facenet_cloak_diff - facenet_target_cloak_diff,
                    min=0.0,
                )

                baseline_arcface_push_weight_scale = torch.ones(
                    B, device=imgs.device
                )
                baseline_facenet_push_weight_scale = torch.ones(
                    B, device=imgs.device
                )
                baseline_arcface_cloak_weight_scale = torch.ones(
                    B, device=imgs.device
                )
                baseline_facenet_cloak_weight_scale = torch.ones(
                    B, device=imgs.device
                )

                baseline_arcface_push_loss = (
                    -self.config.third_party.defense.weight.arcface_push
                    * baseline_arcface_push_weight_scale
                    * baseline_arcface_push
                )
                baseline_arcface_cloak_loss = (
                    self.config.third_party.defense.weight.arcface_cloak
                    * baseline_arcface_cloak_weight_scale
                    * baseline_arcface_cloak_excess
                )
                baseline_facenet_push_loss = (
                    -self.config.third_party.defense.weight.facenet_push
                    * baseline_facenet_push_weight_scale
                    * baseline_facenet_push
                )
                baseline_facenet_cloak_loss = (
                    self.config.third_party.defense.weight.facenet_cloak
                    * baseline_facenet_cloak_weight_scale
                    * baseline_facenet_cloak_excess
                )
                baseline_loss = (
                    baseline_pert_diff_loss
                    + baseline_background_loss
                    + baseline_structure_loss
                    + baseline_tv_loss
                    + baseline_arcface_push_loss
                    + baseline_arcface_cloak_loss
                    + baseline_facenet_push_loss
                    + baseline_facenet_cloak_loss
                ).mean()
                self.logger.info(
                    format_loss_log(
                        0,
                        baseline_loss,
                        baseline_pert_diff_raw,
                        baseline_pert_diff_loss,
                        baseline_background_raw,
                        baseline_background_loss,
                        baseline_structure_raw,
                        baseline_structure_loss,
                        baseline_tv_raw,
                        baseline_tv_loss,
                        baseline_arcface_push_raw,
                        baseline_arcface_push,
                        baseline_arcface_push_weight_scale,
                        baseline_arcface_push_loss,
                        baseline_arcface_cloak_diff,
                        arcface_target_cloak_diff,
                        baseline_arcface_cloak_excess,
                        baseline_arcface_cloak_weight_scale,
                        baseline_arcface_cloak_loss,
                        baseline_facenet_push_raw,
                        baseline_facenet_push,
                        baseline_facenet_push_weight_scale,
                        baseline_facenet_push_loss,
                        baseline_facenet_cloak_diff,
                        facenet_target_cloak_diff,
                        baseline_facenet_cloak_excess,
                        baseline_facenet_cloak_weight_scale,
                        baseline_facenet_cloak_loss,
                    )
                )

        for epoch in range(self.config.third_party.defense.epochs):
            x_imgs = x_imgs.clone().detach().requires_grad_(True)

            pert_diff_loss = (
                self.config.third_party.defense.weight.perturb
                * weighted_l2_per_image(x_imgs, imgs.detach(), perturb_weight_map)
            )
            pert_diff_raw = weighted_l2_per_image(
                x_imgs, imgs.detach(), perturb_weight_map
            )
            background_loss = (
                self.config.third_party.defense.weight.background
                * l2_per_image(x_imgs * non_face_mask, imgs.detach() * non_face_mask)
            )
            background_raw = l2_per_image(
                x_imgs * non_face_mask, imgs.detach() * non_face_mask
            )
            structure_raw = weighted_l2_per_image(
                x_imgs, imgs.detach(), structure_mask
            )
            structure_loss = (
                self.config.third_party.defense.weight.structure * structure_raw
            )
            tv_raw = total_variation_per_image(x_imgs, imgs.detach())
            tv_loss = self.config.third_party.defense.weight.tv * tv_raw

            x_arcface = self.get_arcface_embedding(x_imgs)
            arcface_push_raw = l2_per_image(x_arcface, clean_arcface)
            arcface_push = torch.clamp(
                arcface_push_raw,
                0,
                self.config.third_party.defense.limit.arcface,
            )
            arcface_cloak_diff = l2_per_image(
                x_arcface, cloak_arcface
            )
            arcface_cloak_excess = torch.clamp(
                arcface_cloak_diff - arcface_target_cloak_diff,
                min=0.0,
            )

            arcface_push_progress = get_relative_progress(
                arcface_push_raw,
                arcface_initial_push,
                arcface_target_push,
            )
            (
                push_progress_ema["arcface"],
                push_progress_sq_ema["arcface"],
                arcface_push_progress_var,
            ) = update_progress_stats(
                push_progress_ema["arcface"],
                push_progress_sq_ema["arcface"],
                arcface_push_progress,
            )
            arcface_pull_progress = get_reverse_relative_progress(
                arcface_cloak_diff,
                arcface_target_push,
                arcface_target_cloak_diff,
            )
            (
                pull_progress_ema["arcface"],
                pull_progress_sq_ema["arcface"],
                arcface_pull_progress_var,
            ) = update_progress_stats(
                pull_progress_ema["arcface"],
                pull_progress_sq_ema["arcface"],
                arcface_pull_progress,
            )

            x_facenet = self.get_facenet_embedding(x_imgs)
            facenet_push_raw = l2_per_image(x_facenet, clean_facenet)
            facenet_push = torch.clamp(
                facenet_push_raw,
                0,
                self.config.third_party.defense.limit.facenet,
            )
            facenet_cloak_diff = l2_per_image(x_facenet, cloak_facenet)
            facenet_cloak_excess = torch.clamp(
                facenet_cloak_diff - facenet_target_cloak_diff,
                min=0.0,
            )

            facenet_push_progress = get_relative_progress(
                facenet_push_raw,
                facenet_initial_push,
                facenet_target_push,
            )
            (
                push_progress_ema["facenet"],
                push_progress_sq_ema["facenet"],
                facenet_push_progress_var,
            ) = update_progress_stats(
                push_progress_ema["facenet"],
                push_progress_sq_ema["facenet"],
                facenet_push_progress,
            )
            facenet_pull_progress = get_reverse_relative_progress(
                facenet_cloak_diff,
                facenet_target_push,
                facenet_target_cloak_diff,
            )
            (
                pull_progress_ema["facenet"],
                pull_progress_sq_ema["facenet"],
                facenet_pull_progress_var,
            ) = update_progress_stats(
                pull_progress_ema["facenet"],
                pull_progress_sq_ema["facenet"],
                facenet_pull_progress,
            )
            arcface_push_weight_scale, facenet_push_weight_scale = get_dynamic_weights(
                [arcface_push_progress, facenet_push_progress],
                [arcface_push_progress_var, facenet_push_progress_var],
            )
            arcface_cloak_weight_scale, facenet_cloak_weight_scale = get_dynamic_weights(
                [arcface_pull_progress, facenet_pull_progress],
                [arcface_pull_progress_var, facenet_pull_progress_var],
            )
            arcface_push_loss = (
                -self.config.third_party.defense.weight.arcface_push
                * arcface_push_weight_scale
                * arcface_push
            )
            arcface_cloak_loss = (
                self.config.third_party.defense.weight.arcface_cloak
                * arcface_cloak_weight_scale
                * arcface_cloak_excess
            )
            facenet_push_loss = (
                -self.config.third_party.defense.weight.facenet_push
                * facenet_push_weight_scale
                * facenet_push
            )
            facenet_cloak_loss = (
                self.config.third_party.defense.weight.facenet_cloak
                * facenet_cloak_weight_scale
                * facenet_cloak_excess
            )

            loss_per_img = (
                pert_diff_loss
                + background_loss
                + structure_loss
                + tv_loss
                + arcface_push_loss
                + arcface_cloak_loss
                + facenet_push_loss
                + facenet_cloak_loss
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
                    format_loss_log(
                        epoch + 1,
                        loss,
                        pert_diff_raw,
                        pert_diff_loss,
                        background_raw,
                        background_loss,
                        structure_raw,
                        structure_loss,
                        tv_raw,
                        tv_loss,
                        arcface_push_raw,
                        arcface_push,
                        arcface_push_weight_scale,
                        arcface_push_loss,
                        arcface_cloak_diff,
                        arcface_target_cloak_diff,
                        arcface_cloak_excess,
                        arcface_cloak_weight_scale,
                        arcface_cloak_loss,
                        facenet_push_raw,
                        facenet_push,
                        facenet_push_weight_scale,
                        facenet_push_loss,
                        facenet_cloak_diff,
                        facenet_target_cloak_diff,
                        facenet_cloak_excess,
                        facenet_cloak_weight_scale,
                        facenet_cloak_loss,
                    )
                )

        return best_imgs

    def _free_gpu(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
