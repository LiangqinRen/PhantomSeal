from src import metric
from src.common_utils import save_tensor_imgs
from src.dataset import MetricDataset
from src.faceshifter.defense import Defense

import textwrap
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import Tensor, nn
from torch.utils.data import DataLoader


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
        batch_size = self.config.third_party.lowkey.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_count = 0

        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()

            cloak_imgs = self.cloak.find_best_cloaks(self._denormalize(imgs_A))
            torch.set_grad_enabled(True)
            pert_imgs = self._perturb_imgs_lowkey(
                self._denormalize(imgs_A),
                silent=self.config.third_party.lowkey.silent_perturb,
            )
            torch.set_grad_enabled(False)

            imgs_A_list = list(torch.chunk(imgs_A, chunks=imgs_A.size(0), dim=0))
            imgs_B_list = list(torch.chunk(imgs_B, chunks=imgs_B.size(0), dim=0))
            pert_imgs_list = list(
                torch.chunk(self._normalize(pert_imgs), chunks=pert_imgs.size(0), dim=0)
            )

            source_swap_list = []
            pert_source_swap_list = []
            valid_indexes = []
            for sample_idx in range(imgs_A.size(0)):
                try:
                    source_swap = self.swap_face_whitebox(
                        imgs_A_list[sample_idx], imgs_B_list[sample_idx]
                    ).cuda()
                    pert_source_swap = self.swap_face_whitebox(
                        pert_imgs_list[sample_idx], imgs_B_list[sample_idx]
                    ).cuda()
                except Exception as exc:
                    self.logger.warning(
                        "FaceShifter low-key whitebox swap skipped on batch %d pair %d: %s",
                        idx,
                        sample_idx + 1,
                        exc,
                    )
                    continue

                source_swap_list.append(source_swap)
                pert_source_swap_list.append(pert_source_swap)
                valid_indexes.append(sample_idx)
                total_count += 1

            if not valid_indexes:
                self.logger.warning(
                    "FaceShifter low-key batch %d skipped because all whitebox swaps failed.",
                    idx,
                )
                self._free_gpu()
                continue

            images_idx = torch.as_tensor(valid_indexes, device=imgs_A.device)
            imgs_A_valid = self._denormalize(imgs_A[images_idx])
            imgs_B_valid = self._denormalize(imgs_B[images_idx])
            pert_imgs_valid = pert_imgs[images_idx]
            cloak_imgs_valid = cloak_imgs[images_idx]

            source_swap = torch.cat(source_swap_list, dim=0).float()
            pert_source_swap = torch.cat(pert_source_swap_list, dim=0).float()

            utility = self.utility.calculate_utility(imgs_A_valid, pert_imgs_valid)
            source_utility = self.utility.calculate_utility(source_swap, pert_source_swap)
            source_effectiveness = self.effectiveness.calculate_effectiveness(
                imgs_A_valid,
                pert_imgs_valid,
                source_swap,
                pert_source_swap,
                cloak_imgs_valid,
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
                    imgs_A_valid,
                    imgs_B_valid,
                    pert_imgs_valid,
                    cloak_imgs_valid,
                    source_swap,
                    pert_source_swap,
                ],
                only_save_summary=True,
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
                Batch {idx:4}/{len(dataloader):4}, {total_count} valid pairs of pictures
                utility: {metric.generate_summary_utility_log(metrics, 'utility', idx)}
                source utility: {metric.generate_summary_utility_log(metrics, 'pert_source_utility', idx)}
                source effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'pert_source_effectiveness')}
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

        def get_imgs_identity(inputs: Tensor) -> Tensor:
            normalized_inputs = self._normalize(inputs)
            return self.arcface(
                F.interpolate(
                    normalized_inputs[:, :, 19:237, 19:237],
                    (112, 112),
                    mode="bilinear",
                    align_corners=True,
                )
            )

        x_imgs = imgs.clone().detach() + torch.randn_like(imgs) * 1.0e-5
        clean_identity = get_imgs_identity(imgs)
        lowkey_config = self.config.third_party.lowkey
        epsilon = (
            lowkey_config.epsilon * (torch.max(x_imgs) - torch.min(x_imgs)) / 2
        )

        best_imgs = imgs.clone().detach()
        best_loss = torch.full((imgs.size(0),), float("inf"), device=imgs.device)

        if not silent:
            with torch.no_grad():
                baseline_perturb_loss = (
                    lowkey_config.weight.perturb
                    * self.utility.lpips_distance(imgs, imgs.detach()).view(-1)
                )
                baseline_blur_identity = get_imgs_identity(self.blur(imgs))
                baseline_identity_loss = (
                    lowkey_config.weight.identity
                    * -(l2_per_image(baseline_blur_identity, clean_identity.detach()))
                    / norm_per_image(clean_identity.detach())
                )
                baseline_loss = (baseline_perturb_loss + baseline_identity_loss).mean()
                self.logger.info(
                    f"[Epoch {0:4}/{lowkey_config.epochs:4}] "
                    f"loss: {baseline_loss.item():.5f}"
                    f"({baseline_perturb_loss.mean().item():.5f}, "
                    f"{baseline_identity_loss.mean().item():.5f})"
                )

        for epoch in range(lowkey_config.epochs):
            x_imgs = x_imgs.clone().detach().requires_grad_(True)

            perturb_loss = (
                lowkey_config.weight.perturb
                * self.utility.lpips_distance(x_imgs, imgs.detach()).view(-1)
            )

            x_identity = get_imgs_identity(x_imgs)
            blur_identity = get_imgs_identity(self.blur(x_imgs))
            identity_loss = (
                lowkey_config.weight.identity
                * -(
                    l2_per_image(x_identity, clean_identity.detach())
                    + l2_per_image(blur_identity, clean_identity.detach())
                )
                / norm_per_image(clean_identity.detach())
            )

            loss_per_img = perturb_loss + identity_loss
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
                    f"({perturb_loss.mean().item():.5f}, {identity_loss.mean().item():.5f})"
                )

        return best_imgs
