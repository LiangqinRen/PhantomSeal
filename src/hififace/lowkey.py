from src import metric
from src.hififace.defense import Defense
from src.dataset import MetricDataset
from src.common_utils import save_tensor_imgs

import textwrap
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from pathlib import Path


class GaussianBlur(nn.Module):
    def __init__(
        self, sigma: float = 3.0, kernel_size: int = 7, pad_mode: str = "reflect"
    ):
        super().__init__()
        assert kernel_size % 2 == 1 and kernel_size > 0
        self.sigma = float(sigma)
        self.kernel_size = int(kernel_size)
        self.pad_mode = pad_mode

        coords = torch.arange(self.kernel_size, dtype=torch.float32)
        coords = coords - (self.kernel_size - 1) / 2
        gaussian_1d = torch.exp(-(coords**2) / (2 * self.sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        gaussian_2d = gaussian_2d / gaussian_2d.sum()

        self.register_buffer(
            "kernel2d",
            gaussian_2d.view(1, 1, self.kernel_size, self.kernel_size),
            persistent=False,
        )

    def forward(self, imgs: Tensor) -> Tensor:
        assert imgs.dim() == 4
        _, channels, _, _ = imgs.shape
        pad = self.kernel_size // 2

        kernel = self.kernel2d.to(dtype=imgs.dtype, device=imgs.device)
        kernel = kernel.expand(channels, 1, self.kernel_size, self.kernel_size)

        if self.pad_mode == "constant":
            return F.conv2d(imgs, kernel, padding=pad, groups=channels)

        padded_imgs = F.pad(imgs, (pad, pad, pad, pad), mode=self.pad_mode)
        return F.conv2d(padded_imgs, kernel, padding=0, groups=channels)


class Lowkey(Defense):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.blur = GaussianBlur(sigma=3.0, kernel_size=7, pad_mode="reflect").cuda()

    def metric(self) -> None:
        metrics = metric.get_metric_data_template(self.effectiveness)

        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.lowkey.batch_size, shuffle=True
        )
        total_count = 0

        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            torch.set_grad_enabled(True)
            pert_imgs = self._perturb_imgs_lowkey(
                imgs_A,
                silent=self.config.third_party.lowkey.silent_perturb,
            )
            torch.set_grad_enabled(False)

            source_swap = self.swap_face(imgs_A, imgs_B)
            pert_source_swap = self.swap_face(pert_imgs, imgs_B)
            utility = self.utility.calculate_utility(imgs_A, pert_imgs)
            source_utility = self.utility.calculate_utility(source_swap, pert_source_swap)
            source_effectiveness = self.effectiveness.calculate_effectiveness(
                imgs_A,
                pert_imgs,
                source_swap,
                pert_source_swap,
                cloak_imgs,
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

            scores = self.score_calculator.calculate_score(
                source_effectiveness,
                None,
                metrics,
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
                only_save_summary=True,
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

            del imgs_A, imgs_B, pert_imgs
            del source_swap, pert_source_swap
            self._free_gpu()

    def _perturb_imgs_lowkey(self, imgs: Tensor, silent: bool = False) -> Tensor:
        def l2_per_image(x: Tensor, y: Tensor) -> Tensor:
            return ((x - y) ** 2).view(x.size(0), -1).mean(dim=1)

        def norm_per_image(x: Tensor) -> Tensor:
            return x.view(x.size(0), -1).norm(p=2, dim=1).clamp_min(1e-12)

        def get_id_identity(inputs: Tensor) -> Tensor:
            return F.normalize(
                self.net.generator.id_extractor.f_id(
                    F.interpolate((inputs - 0.5) / 0.5, size=112, mode="bilinear")
                ),
                dim=-1,
                p=2,
            )

        x_imgs = imgs.clone().detach() + torch.randn_like(imgs) * 1.0e-5
        lowkey_config = self.config.third_party.lowkey
        epsilon = (
            lowkey_config.epsilon * (torch.max(x_imgs) - torch.min(x_imgs)) / 2
        )

        with torch.no_grad():
            clean_3d_identity = self.net.generator.id_extractor.f_3d(imgs)[:, :80]
            clean_id_identity = get_id_identity(imgs)

        best_imgs = imgs.clone().detach()
        best_loss = torch.full((imgs.size(0),), float("inf"), device=imgs.device)

        if not silent:
            with torch.no_grad():
                baseline_perturb_loss = (
                    lowkey_config.weight.perturb
                    * self.utility.lpips_distance(imgs, imgs.detach()).view(-1)
                )
                baseline_blur_3d_identity = self.net.generator.id_extractor.f_3d(
                    self.blur(imgs)
                )[:, :80]
                baseline_blur_id_identity = get_id_identity(self.blur(imgs))
                baseline_identity_3d_loss = (
                    lowkey_config.weight.identity_3d
                    * -l2_per_image(
                        baseline_blur_3d_identity, clean_3d_identity.detach()
                    )
                    / norm_per_image(clean_3d_identity.detach())
                )
                baseline_identity_id_loss = (
                    lowkey_config.weight.identity_id
                    * -l2_per_image(
                        baseline_blur_id_identity, clean_id_identity.detach()
                    )
                    / norm_per_image(clean_id_identity.detach())
                )
                baseline_loss = (
                    baseline_perturb_loss
                    + baseline_identity_3d_loss
                    + baseline_identity_id_loss
                ).mean()
                self.logger.info(
                    f"[Epoch {0:4}/{lowkey_config.epochs:4}] "
                    f"loss: {baseline_loss.item():.5f}"
                    f"({baseline_perturb_loss.mean().item():.5f}, "
                    f"{baseline_identity_3d_loss.mean().item():.5f}, "
                    f"{baseline_identity_id_loss.mean().item():.5f})"
                )

        for epoch in range(lowkey_config.epochs):
            x_imgs = x_imgs.clone().detach().requires_grad_(True)

            perturb_loss = (
                lowkey_config.weight.perturb
                * self.utility.lpips_distance(x_imgs, imgs.detach()).view(-1)
            )

            x_3d_identity = self.net.generator.id_extractor.f_3d(x_imgs)[:, :80]
            blur_3d_identity = self.net.generator.id_extractor.f_3d(
                self.blur(x_imgs)
            )[:, :80]
            identity_3d_loss = (
                lowkey_config.weight.identity_3d
                * -(
                    l2_per_image(x_3d_identity, clean_3d_identity.detach())
                    + l2_per_image(blur_3d_identity, clean_3d_identity.detach())
                )
                / norm_per_image(clean_3d_identity.detach())
            )

            x_id_identity = get_id_identity(x_imgs)
            blur_id_identity = get_id_identity(self.blur(x_imgs))
            identity_id_loss = (
                lowkey_config.weight.identity_id
                * -(
                    l2_per_image(x_id_identity, clean_id_identity.detach())
                    + l2_per_image(blur_id_identity, clean_id_identity.detach())
                )
                / norm_per_image(clean_id_identity.detach())
            )

            loss_per_img = perturb_loss + identity_3d_loss + identity_id_loss
            loss = loss_per_img.mean()
            loss.backward()

            if x_imgs.grad is not None:
                grad_sign = x_imgs.grad.sign().detach()
            else:
                grad_sign = torch.zeros_like(x_imgs)

            x_imgs = x_imgs.detach() - epsilon * grad_sign
            x_imgs = torch.clamp(x_imgs, 0.0, 1.0)

            improved = loss_per_img.detach() < best_loss
            best_loss[improved] = loss_per_img.detach()[improved]
            best_imgs[improved] = x_imgs[improved].detach()

            if (not silent) and (
                (epoch + 1) % lowkey_config.log_interval == 0
                or (epoch + 1) == lowkey_config.epochs
            ):
                self.logger.info(
                    f"[Epoch {epoch + 1:4}/{lowkey_config.epochs:4}] "
                    f"loss: {loss.item():.5f}"
                    f"({perturb_loss.mean().item():.5f}, "
                    f"{identity_3d_loss.mean().item():.5f}, "
                    f"{identity_id_loss.mean().item():.5f})"
                )

        return best_imgs
