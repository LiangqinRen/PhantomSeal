from src.diffface.base import Base
from src.dataset import FFHQDataset
from src.utils import save_tensor_imgs
from src.evaluate import ScoreCalculator
import src.metric as metric
from src.utils import check_tensor_info

import torch
import textwrap
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch import Tensor


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
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs)
            torch.set_grad_enabled(False)

            results = rev_results = pert_src_results = pert_tgt_results = (
                torch.ones_like(x_imgs)
            )

            if self.config.evaluate.effectiveness.ASRo:
                results = self._face_swap_per_image(imgs_A, imgs_B)
                rev_results = self._face_swap_per_image(imgs_B, imgs_A)

            if self.config.evaluate.effectiveness.ASRp:
                pert_src_results = self._face_swap_per_image(x_imgs, imgs_B)
                pert_tgt_results = self._face_swap_per_image(imgs_B, x_imgs)

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
                pert_src_results,
                rev_results,
                pert_tgt_results,
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
                    "swap",
                    "rev_swap",
                    "cloak_imgs",
                    "pert",
                    "pert_src\nswap",
                    "pert_tgt\nswap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    results,
                    rev_results,
                    cloak_imgs,
                    x_imgs,
                    pert_src_results,
                    pert_tgt_results,
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )
            del imgs_A, imgs_B, x_imgs, cloak_imgs
            del (
                results,
                rev_results,
                pert_src_results,
                pert_tgt_results,
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

    def _face_swap_per_image(self, imgs_A: Tensor, imgs_B: Tensor) -> Tensor:
        assert imgs_A.shape[0] == imgs_B.shape[0]

        results = []
        imgs_A = imgs_A.cpu()
        imgs_B = imgs_B.cpu()

        for i in range(imgs_A.size(0)):
            a = imgs_A[i : i + 1].contiguous().to(self.device, non_blocking=True)
            b = imgs_B[i : i + 1].contiguous().to(self.device, non_blocking=True)
            with torch.no_grad():
                out = self.swap_face(a, b)

            results.append(out.detach().cpu())

            del a, b, out

        return torch.cat(results, dim=0).cuda()

    def _perturb_imgs(self, imgs: Tensor, cloak_imgs: Tensor) -> Tensor:
        def l2_per_image(x: Tensor, y: Tensor) -> Tensor:
            return ((x - y) ** 2).view(x.size(0), -1).mean(dim=1)

        x_imgs = imgs.clone().detach() + torch.randn_like(imgs) * 1e-5

        with torch.no_grad():
            self_identity = self._get_imgs_identity(self._normalize(x_imgs))
            cloak_identity = self._get_imgs_identity(self._normalize(cloak_imgs))

        epsilon = (
            self.config.third_party.defense.epsilon
            * (torch.max(x_imgs) - torch.min(x_imgs))
            / 2
        )
        limits = torch.tensor(
            [
                self.config.third_party.defense.limit.R,
                self.config.third_party.defense.limit.G,
                self.config.third_party.defense.limit.B,
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

        B = imgs.size(0)
        best_imgs = imgs.clone()
        best_loss = torch.full((B,), float("inf"), device=imgs.device)

        for epoch in range(self.config.third_party.defense.epochs):
            x_imgs = x_imgs.clone().detach().requires_grad_(True)
            # perturb
            pert_diff_loss = (
                self.config.third_party.defense.weight.perturb
                * l2_per_image(x_imgs, imgs.detach())
            )

            # identity
            x_identity = self._get_imgs_identity(self._normalize(x_imgs))
            identity_diff = (
                self.config.third_party.defense.weight.identity
                * l2_per_image(x_identity, self_identity)
            )
            identity_diff_loss = -torch.clamp(
                identity_diff,
                0,
                self.config.third_party.defense.limit.identity,
            )
            # cloak
            cloak_diff_loss = (
                self.config.third_party.defense.weight.cloak
                * l2_per_image(x_identity, cloak_identity)
            )

            # context
            src_logits = self.netSeg(self.spNorm(self._normalize(x_imgs)))[0]
            probs = torch.softmax(src_logits, dim=1)
            face_logits_mean = src_logits[:, self.face_ids].mean(dim=(1, 2, 3))
            tgt_logits_mean = src_logits[:, self.target_nonface_id].mean(dim=(1, 2))

            seg_margin_loss_per_img = F.relu(face_logits_mean - tgt_logits_mean + 0.1)

            face_prob_mean_per_img = probs[:, self.face_ids].mean(dim=(1, 2, 3))

            context_loss = self.config.third_party.defense.weight.context * (
                seg_margin_loss_per_img + 0.5 * face_prob_mean_per_img
            )

            loss_per_img = (
                pert_diff_loss + identity_diff_loss + cloak_diff_loss + context_loss
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
            ):
                self.logger.info(
                    f"[Epoch {epoch+1:4}/{self.config.third_party.defense.epochs:4}] "
                    f"loss: {loss.item():.5f}("
                    f"{pert_diff_loss.mean().item():.5f}, "
                    f"{identity_diff_loss.mean().item():.5f}, "
                    f"{cloak_diff_loss.mean().item():.5f}, "
                    f"{context_loss.mean().item():.5f})"
                )

        return best_imgs

    def _free_gpu(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
