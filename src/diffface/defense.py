from src.diffface.base import Base
from src.dataset import FFHQDataset
from src.utils import save_tensor_imgs
from src.evaluate import Utility, Effectiveness, Cloak
import src.metric as metric

import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torchvision import transforms


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.utility = Utility(logger, config)
        self.effectiveness = Effectiveness(logger, config)
        self.cloak = Cloak(logger, config, self.effectiveness)

        self.face_ids = [1, 2, 3, 4, 5, 10, 11, 12, 13]
        self.target_nonface_id = 10

    def metric(self) -> None:
        data = metric.get_metric_data_template(self.effectiveness)

        dataset = FFHQDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs, silent=False)

            results = self._face_swap_per_image(imgs_A, imgs_B)
            rev_results = self._face_swap_per_image(imgs_B, imgs_A)
            # cloak_results = self._face_swap_per_image(cloak_imgs, imgs_B)
            pert_src_results = self._face_swap_per_image(x_imgs, imgs_B)
            pert_tgt_results = self._face_swap_per_image(imgs_B, x_imgs)
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "swap",
                    "rev_swap",
                    "cloak_imgs",
                    # "cloak_swap",
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
                    # cloak_results,
                    x_imgs,
                    pert_src_results,
                    pert_tgt_results,
                ],
                only_save_summary=True,
            )

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
                data,
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            )

            del imgs_A, imgs_B, x_imgs, cloak_imgs
            del (
                results,
                rev_results,
                # cloak_results,
                pert_src_results,
                pert_tgt_results,
            )
            self._free_gpu()

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

    def validate(self) -> None:
        pass

    def _face_swap_per_image(self, imgs_A: Tensor, imgs_B: Tensor) -> Tensor:
        assert imgs_A.shape[0] == imgs_B.shape[0]

        results = []
        imgs_A = imgs_A.cpu()
        imgs_B = imgs_B.cpu()
        torch.cuda.empty_cache()

        for i in range(imgs_A.size(0)):
            a = imgs_A[i : i + 1].contiguous().to(self.device, non_blocking=True)
            b = imgs_B[i : i + 1].contiguous().to(self.device, non_blocking=True)
            with torch.no_grad():
                out = self.swap_face(a, b)

            results.append(out.detach().cpu())

            del a, b, out
            torch.cuda.empty_cache()

        return torch.cat(results, dim=0).cuda()

    def _perturb_imgs(self, imgs: Tensor, cloak_imgs: Tensor, silent=True) -> Tensor:
        l2_loss = nn.MSELoss()
        cloak_identity = self._get_imgs_identity(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                cloak_imgs
            )
        ).detach()
        limits = torch.tensor(
            [
                self.config.third_party.defense.limit.R,
                self.config.third_party.defense.limit.G,
                self.config.third_party.defense.limit.B,
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

        x0 = imgs.detach()
        steps = int(self.config.third_party.defense.epochs)
        alpha = limits / steps
        delta = torch.empty_like(x0).uniform_(-1.0, 1.0) * limits
        delta = torch.max(torch.min(delta, 1 - x0), -x0)
        m = torch.zeros_like(x0)

        best_imgs, best_loss = x0.clone(), float("inf")
        for epoch in range(steps):
            x_imgs = (x0 + delta).clamp(0, 1).detach().requires_grad_(True)

            pert_diff_loss = self.config.third_party.defense.weight.perturb * l2_loss(
                x_imgs, imgs.detach()
            )

            x_identity = self._get_imgs_identity(
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                    x_imgs
                )
            )
            identity_diff_loss = (
                self.config.third_party.defense.weight.identity
                * l2_loss(x_identity, cloak_identity)
            )

            src_logits = self.netSeg(
                self.spNorm(
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                        x_imgs
                    )
                )
            )[0]
            probs = torch.softmax(src_logits, dim=1)
            face_logits_mean = src_logits[:, self.face_ids].mean(dim=(1, 2, 3))
            tgt_logits_mean = src_logits[:, self.target_nonface_id].mean(dim=(1, 2))
            seg_margin_loss = F.relu(face_logits_mean - tgt_logits_mean + 0.1).mean()
            face_prob_mean = probs[:, self.face_ids].mean()
            context_loss = self.config.third_party.defense.weight.context * (
                seg_margin_loss + 0.5 * face_prob_mean
            )

            loss = pert_diff_loss + identity_diff_loss + context_loss
            loss.backward()

            g = x_imgs.grad
            m = 0.9 * m + g / (g.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8)
            grad_sign = m.sign()

            delta = delta - alpha * grad_sign
            delta = torch.clamp(delta, -limits, limits)
            delta = torch.max(torch.min(delta, 1 - x0), -x0)

            x_curr = (x0 + delta).clamp(0, 1).detach()

            with torch.no_grad():
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_imgs = x_curr.clone()

            if not silent:
                self.logger.info(
                    f"[Epoch {epoch+1:4}/{self.config.third_party.defense.epochs}]loss: {loss:.5f}({pert_diff_loss.item():.5f}, {identity_diff_loss.item():.5f}, {context_loss.item():.5f})"
                )

        return best_imgs

    def _free_gpu(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
