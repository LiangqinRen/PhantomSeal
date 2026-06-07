from src import metric
from src.unify.base import Base
from src.dataset import FFHQMetric, MetricDataset
from src.evaluate import DistanceCloakSelector, ScoreCalculator
from src.common_utils import save_tensor_imgs
from src.diffface.defense import Defense as DiffFaceDefense

import torch
import copy
import textwrap
from torch import tensor, Tensor
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torchvision import transforms


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        notes_path = Path(self.config.notes_path)
        notes_path.touch(exist_ok=True)

    def extend(self) -> None:
        metrics = self._get_extend_metric_template()
        extend_config = self.config.third_party.extend
        diffface_config = self._load_diffface_config()
        diffface_dataset = diffface_config.dataset
        metric_dir = extend_config.get("metric_dir", diffface_dataset.metric_dir)
        metric_pairs = extend_config.get("metric_pairs", diffface_dataset.metric_pairs)
        image_size = extend_config.get("image_size", diffface_dataset.image_size)
        batch_size = self.config.third_party.defense.batch_size

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        dataset = FFHQMetric(Path(metric_dir), metric_pairs, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        ffhq_config = copy.deepcopy(self.config)
        with open_dict(ffhq_config):
            ffhq_config.third_party.dataset.cloak_dir = extend_config.get(
                "cloak_dir", diffface_dataset.cloak_dir
            )
            ffhq_config.third_party.dataset.cloak_mix = extend_config.get(
                "cloak_mix", diffface_dataset.cloak_mix
            )
            ffhq_config.third_party.dataset.cloak_count = extend_config.get(
                "cloak_count", diffface_dataset.cloak_count
            )
            ffhq_config.third_party.dataset.cloak_min_distance = extend_config.get(
                "cloak_min_distance", diffface_dataset.cloak_min_distance
            )
            ffhq_config.third_party.dataset.cloak_distance = extend_config.get(
                "cloak_distance", diffface_dataset.cloak_distance
            )
            ffhq_config.third_party.dataset.use_224 = False
        ffhq_cloak = DistanceCloakSelector(self.logger, ffhq_config, self.effectiveness)
        diffface = self._build_diffface_defense(diffface_config)

        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            cloak_imgs = ffhq_cloak.find_best_cloaks(imgs_A)
            if cloak_imgs.shape[-2:] != imgs_A.shape[-2:]:
                cloak_imgs = F.interpolate(
                    cloak_imgs,
                    size=imgs_A.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            pert_imgs = self._perturb_imgs(imgs_A, cloak_imgs)
            torch.set_grad_enabled(False)

            diffface_swap = diffface._face_swap_per_image(imgs_A, imgs_B)
            diffface_pert_swap = diffface._face_swap_per_image(pert_imgs, imgs_B)

            total_count += len(imgs_A)
            utility = self.utility.calculate_utility(imgs_A, pert_imgs)
            diffface_effectiveness = self.effectiveness.calculate_effectiveness(
                imgs_A,
                None,
                diffface_swap,
                diffface_pert_swap,
                cloak_imgs,
            )
            self._merge_extend_metric(metrics, utility, diffface_effectiveness)

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "perturb_imgs",
                    "cloak_imgs",
                    "diffface_swap",
                    "diffface_perturb_swap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    pert_imgs,
                    cloak_imgs,
                    diffface_swap,
                    diffface_pert_swap,
                ],
                image_name="extend_diffface",
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            del imgs_A, imgs_B, cloak_imgs, pert_imgs, diffface_swap, diffface_pert_swap
            self._free_gpu()

            diffface_iter_scores = self._calculate_iter_score(diffface_effectiveness)
            summary_scores = self._calculate_summary_score(metrics, ["diffface"])

            iter_log_str = textwrap.dedent(
                f"""
            protection utility: {metric.generate_iter_utility_log(utility)}
            diffface 𝒯_identity effectiveness {metric.generate_iter_effectiveness_label(diffface_effectiveness)}: {metric.generate_iter_effectiveness_log(diffface_effectiveness, include_labels=False)}
            scores: {metric.generate_iter_score_log(diffface_iter_scores)}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            protection utility: {metric.generate_summary_utility_log(metrics, 'pert_utility', idx)}
            diffface 𝒯_identity effectiveness {metric.generate_summary_effectiveness_label(metrics, 'diffface')}: {metric.generate_summary_effectiveness_log(metrics, 'diffface', include_labels=False)}
            scores: {self._generate_summary_score_log(summary_scores['diffface'])}
            """
            )
            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

    def _load_diffface_config(self):
        return OmegaConf.load(
            Path(self.config.root_dir) / "config/third_party/diffface.yaml"
        )

    def _build_diffface_defense(self, diffface_third_party_config) -> DiffFaceDefense:
        diffface_config = copy.deepcopy(self.config)
        with open_dict(diffface_config):
            diffface_config.third_party = copy.deepcopy(diffface_third_party_config)
            diffface_config.third_party.defense = copy.deepcopy(
                self.config.third_party.defense
            )
            diffface_config.third_party.defense.batch_size = 1
        return DiffFaceDefense(self.logger, diffface_config)

    def metric(
        self,
    ) -> None:
        metrics = self._get_unify_metric_template()

        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            pert_imgs = self._perturb_imgs(imgs_A, cloak_imgs)
            torch.set_grad_enabled(False)

            simswap_swap, hififace_swap, faceshifter_swap = self._get_full_swap_results(
                imgs_A, imgs_B
            )
            simswap_pert_swap, hififace_pert_swap, faceshifter_pert_swap = (
                self._get_full_swap_results(pert_imgs, imgs_B)
            )

            # filter valid images
            faceshifter_swap_mask = (
                faceshifter_swap.view(faceshifter_swap.size(0), -1).abs().sum(dim=1)
                > 0
            ) & (
                faceshifter_pert_swap.view(faceshifter_pert_swap.size(0), -1)
                .abs()
                .sum(dim=1)
                > 0
            )

            imgs_A = imgs_A[faceshifter_swap_mask]
            imgs_B = imgs_B[faceshifter_swap_mask]
            cloak_imgs = cloak_imgs[faceshifter_swap_mask]
            pert_imgs = pert_imgs[faceshifter_swap_mask]
            simswap_swap = simswap_swap[faceshifter_swap_mask]
            hififace_swap = hififace_swap[faceshifter_swap_mask]
            faceshifter_swap = faceshifter_swap[faceshifter_swap_mask]
            simswap_pert_swap = simswap_pert_swap[faceshifter_swap_mask]
            hififace_pert_swap = hififace_pert_swap[faceshifter_swap_mask]
            faceshifter_pert_swap = faceshifter_pert_swap[faceshifter_swap_mask]

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "perturb_imgs",
                    "cloak_imgs",
                    "simswap_swap",
                    "hififace_swap",
                    "faceshifter_swap",
                    "simswap_perturb_swap",
                    "hififace_perturb_swap",
                    "faceshifter_perturb_swap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    pert_imgs,
                    cloak_imgs,
                    simswap_swap,
                    hififace_swap,
                    faceshifter_swap,
                    simswap_pert_swap,
                    hififace_pert_swap,
                    faceshifter_pert_swap,
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            total_count += len(faceshifter_pert_swap)

            (
                utility,
                simswap_effectiveness,
                faceshifter_effectiveness,
                hififace_effectiveness,
            ) = self._calculate_cross_model_metric(
                imgs_A,
                pert_imgs,
                cloak_imgs,
                simswap_swap,
                simswap_pert_swap,
                hififace_swap,
                hififace_pert_swap,
                faceshifter_swap,
                faceshifter_pert_swap,
            )

            self._merge_unify_metric(
                metrics,
                utility,
                simswap_effectiveness,
                faceshifter_effectiveness,
                hififace_effectiveness,
            )

            del imgs_A, imgs_B, pert_imgs, cloak_imgs
            del simswap_swap, hififace_swap, faceshifter_swap
            del simswap_pert_swap, hififace_pert_swap, faceshifter_pert_swap
            self._free_gpu()

            simswap_iter_scores = self._calculate_iter_score(simswap_effectiveness)
            faceshifter_iter_scores = self._calculate_iter_score(
                faceshifter_effectiveness
            )
            hififace_iter_scores = self._calculate_iter_score(hififace_effectiveness)
            summary_scores = self._calculate_summary_score(metrics)

            iter_log_str = textwrap.dedent(
                f"""
            protection utility: {metric.generate_iter_utility_log(utility)}
            simswap 𝒯_identity effectiveness {metric.generate_iter_effectiveness_label(simswap_effectiveness)}: {metric.generate_iter_effectiveness_log(simswap_effectiveness, include_labels=False)}
            faceshifter 𝒯_identity effectiveness {metric.generate_iter_effectiveness_label(faceshifter_effectiveness)}: {metric.generate_iter_effectiveness_log(faceshifter_effectiveness, include_labels=False)}
            hififace 𝒯_identity effectiveness {metric.generate_iter_effectiveness_label(hififace_effectiveness)}: {metric.generate_iter_effectiveness_log(hififace_effectiveness, include_labels=False)}
            scores: {metric.generate_iter_score_log(simswap_iter_scores)} {metric.generate_iter_score_log(faceshifter_iter_scores)} {metric.generate_iter_score_log(hififace_iter_scores)}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            protection utility: {metric.generate_summary_utility_log(metrics, 'pert_utility', idx)}
            simswap 𝒯_identity effectiveness {metric.generate_summary_effectiveness_label(metrics, 'simswap')}: {metric.generate_summary_effectiveness_log(metrics, 'simswap', include_labels=False)}
            faceshifter 𝒯_identity effectiveness {metric.generate_summary_effectiveness_label(metrics, 'faceshifter')}: {metric.generate_summary_effectiveness_log(metrics, 'faceshifter', include_labels=False)}
            hififace 𝒯_identity effectiveness {metric.generate_summary_effectiveness_label(metrics, 'hififace')}: {metric.generate_summary_effectiveness_log(metrics, 'hififace', include_labels=False)}
            scores: {self._generate_summary_score_log(summary_scores['simswap'])} {self._generate_summary_score_log(summary_scores['faceshifter'])} {self._generate_summary_score_log(summary_scores['hififace'])}
            """
            )
            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

    def _perturb_imgs(self, imgs: Tensor, cloak_imgs: Tensor) -> Tensor:
        def l2_per_image(x: Tensor, y: Tensor) -> Tensor:
            return ((x - y) ** 2).view(x.size(0), -1).mean(dim=1)

        x_imgs = imgs.clone().detach() + torch.randn_like(imgs) * 1e-5

        with torch.no_grad():
            simswap_self_identity = self.get_simswap_identity(imgs)
            simswap_cloak_identity = self.get_simswap_identity(cloak_imgs)

            faceshifter_self_identity = self.get_faceshifter_identity(imgs)
            faceshifter_cloak_identity = self.get_faceshifter_identity(cloak_imgs)

            hififace_self_3d = self.net.generator.id_extractor.f_3d(
                self._model_256_input(imgs)
            )[:, :80]
            hififace_self_id = self.get_hififace_identity(imgs)
            hififace_cloak_3d = self.net.generator.id_extractor.f_3d(
                self._model_256_input(cloak_imgs)
            )[:, :80]
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
            x_hififace_3d = self.net.generator.id_extractor.f_3d(
                self._model_256_input(x_imgs)
            )[:, :80]
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
                and (
                    (epoch + 1) % self.config.third_party.defense.log_interval == 0
                    or (epoch + 1) == self.config.third_party.defense.epochs
                )
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

    def _get_full_swap_results(
        self, pert_imgs_A: Tensor, imgs_B: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        simswap_pert_swap = self._simswap_swapface(pert_imgs_A, imgs_B)
        hififace_pert_swap = self._hififace_swapface(pert_imgs_A, imgs_B)
        faceshifter_pert_swap = self._faceshifter_swapface(pert_imgs_A, imgs_B)

        return simswap_pert_swap, hififace_pert_swap, faceshifter_pert_swap

    def _calculate_cross_model_metric(
        self,
        imgs_A: Tensor,
        x_imgs: Tensor,
        cloak_imgs: Tensor,
        simswap_swap: Tensor,
        simswap_pert_swap: Tensor,
        hififace_swap: Tensor,
        hififace_pert_swap: Tensor,
        faceshifter_swap: Tensor,
        faceshifter_pert_swap: Tensor,
    ) -> tuple[dict, dict, dict, dict]:
        utility = self.utility.calculate_utility(imgs_A, x_imgs)
        simswap_effectiveness = self.effectiveness.calculate_effectiveness(
            imgs_A,
            x_imgs,
            simswap_swap,
            simswap_pert_swap,
            cloak_imgs,
        )
        faceshifter_effectiveness = self.effectiveness.calculate_effectiveness(
            imgs_A,
            x_imgs,
            faceshifter_swap,
            faceshifter_pert_swap,
            cloak_imgs,
        )
        hififace_effectiveness = self.effectiveness.calculate_effectiveness(
            imgs_A,
            x_imgs,
            hififace_swap,
            hififace_pert_swap,
            cloak_imgs,
        )

        return (
            utility,
            simswap_effectiveness,
            faceshifter_effectiveness,
            hififace_effectiveness,
        )

    def _get_extend_metric_template(self) -> dict:
        data = {
            "pert_utility": (0, 0, 0, 0),
            "diffface": {},
        }

        for effec in self.effectiveness.candi_funcs.keys():
            data["diffface"][effec] = {}
            if self.config.evaluate.effectiveness.perturb:
                data["diffface"][effec]["pert"] = (0, 0)
            if self.config.evaluate.effectiveness.ASRo:
                data["diffface"][effec]["swap"] = (0, 0)
            if self.config.evaluate.effectiveness.ASRp:
                data["diffface"][effec]["pert_swap"] = (0, 0)
            if self.config.evaluate.effectiveness.TSR:
                data["diffface"][effec]["cloak"] = (0, 0)

        return data

    def _merge_extend_metric(
        self,
        data: dict,
        utility: dict,
        diffface_effectiveness: dict,
    ) -> None:
        data["pert_utility"] = tuple(
            x + y
            for x, y in zip(
                data["pert_utility"],
                (
                    utility["mse"],
                    utility["psnr"],
                    utility["ssim"],
                    utility["lpips"],
                ),
            )
        )

        for effec, values in diffface_effectiveness.items():
            for key, value in values.items():
                prev = data["diffface"][effec][key]
                data["diffface"][effec][key] = (
                    prev[0] + value[0],
                    prev[1] + value[1],
                )

    def _get_unify_metric_template(self) -> dict:
        data = {
            "pert_utility": (0, 0, 0, 0),
            "simswap": {},
            "faceshifter": {},
            "hififace": {},
        }

        for effec in self.effectiveness.candi_funcs.keys():
            for model in ["simswap", "faceshifter", "hififace"]:
                data[model][effec] = {}
                if self.config.evaluate.effectiveness.perturb:
                    data[model][effec]["pert"] = (0, 0)
                if self.config.evaluate.effectiveness.ASRo:
                    data[model][effec]["swap"] = (0, 0)
                if self.config.evaluate.effectiveness.ASRp:
                    data[model][effec]["pert_swap"] = (0, 0)
                if self.config.evaluate.effectiveness.TSR:
                    data[model][effec]["cloak"] = (0, 0)
        return data

    def _merge_unify_metric(
        self,
        data: dict,
        utility: dict,
        simswap_effectiveness: dict,
        faceshifter_effectiveness: dict,
        hififace_effectiveness: dict,
    ) -> None:
        data["pert_utility"] = tuple(
            x + y
            for x, y in zip(
                data["pert_utility"],
                (
                    utility["mse"],
                    utility["psnr"],
                    utility["ssim"],
                    utility["lpips"],
                ),
            )
        )

        for model, effectiveness in [
            ("simswap", simswap_effectiveness),
            ("faceshifter", faceshifter_effectiveness),
            ("hififace", hififace_effectiveness),
        ]:
            for effec, values in effectiveness.items():
                for key, value in values.items():
                    prev = data[model][effec][key]
                    data[model][effec][key] = (
                        prev[0] + value[0],
                        prev[1] + value[1],
                    )

    def _free_gpu(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _calculate_iter_score(
        self,
        iter_source_metric: dict,
    ) -> dict:
        scores = {key: {"iter": 0} for key in iter_source_metric.keys()}
        identity_weight = self.config.evaluate.score.identity
        trace_weight = self.config.evaluate.score.trace

        for key in scores.keys():
            iter_source_swap = (
                iter_source_metric[key]["pert_swap"][0]
                / iter_source_metric[key]["pert_swap"][1]
            )
            iter_trace = (
                iter_source_metric[key]["cloak"][0]
                / iter_source_metric[key]["cloak"][1]
            )

            scores[key]["iter"] = (
                identity_weight * (1 - iter_source_swap) + trace_weight * iter_trace
            )

        return scores

    def _calculate_summary_score(
        self, metric: dict, models: list[str] = ["simswap", "faceshifter", "hififace"]
    ) -> dict:
        scores = {key: {k: 0 for k in metric[key].keys()} for key in models}
        identity_weight = self.config.evaluate.score.identity
        trace_weight = self.config.evaluate.score.trace

        for model in models:
            for key in scores[model].keys():
                source_swap = (
                    metric[model][key]["pert_swap"][0]
                    / metric[model][key]["pert_swap"][1]
                )
                trace = metric[model][key]["cloak"][0] / metric[model][key]["cloak"][1]

                scores[model][key] = (
                    identity_weight * (1 - source_swap) + trace_weight * trace
                )

        return scores

    def _generate_summary_score_log(self, scores: dict) -> str:
        vals = (f"{scores[effec]:.3f}" for effec in scores)
        return f"({', '.join(vals)})"
