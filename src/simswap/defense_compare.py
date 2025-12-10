from src import metric
from src.simswap.base import Base
from src.dataset import MetricDataset
from src.utils import save_tensor_imgs


import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F


class GaussianBlur(nn.Module):
    def __init__(
        self, sigma: float = 3.0, kernel_size: int = 7, pad_mode: str = "reflect"
    ):
        super().__init__()
        assert kernel_size % 2 == 1 and kernel_size > 0
        self.sigma = float(sigma)
        self.kernel_size = int(kernel_size)
        self.pad_mode = pad_mode

        k = self.kernel_size
        coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
        g1d = torch.exp(-(coords**2) / (2 * self.sigma**2))
        g1d = g1d / g1d.sum()
        g2d = g1d[:, None] * g1d[None, :]
        g2d = g2d / g2d.sum()

        self.register_buffer("kernel2d", g2d.view(1, 1, k, k), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        N, C, H, W = x.shape
        k = self.kernel_size
        pad = k // 2

        weight = (
            self.kernel2d.to(dtype=x.dtype, device=x.device)
            .expand(C, 1, k, k)
            .contiguous()
        )

        if self.pad_mode == "constant":
            y = F.conv2d(x, weight, bias=None, stride=1, padding=pad, groups=C)
        else:
            x_pad = F.pad(x, (pad, pad, pad, pad), mode=self.pad_mode)
            y = F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, groups=C)
        return y


class Lowkey(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.blur = GaussianBlur(sigma=3.0, kernel_size=7, pad_mode="reflect").cuda()

    def metric(
        self,
    ) -> None:
        data = metric.get_metric_data_template(self.effectiveness)

        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.lowkey.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, silent=True)

            (
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            ) = self._get_full_swap_results(imgs_A, imgs_B, x_imgs)

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
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
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

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "pert_imgs",
                    "swap",
                    "pert_swap",
                    "rev\nswap",
                    "rev\npert_swap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    x_imgs,
                    imgs_A_src_swap,
                    pert_imgs_A_src_swap,
                    imgs_A_tgt_swap,
                    pert_imgs_A_tgt_swap,
                ],
                only_save_summary=True,
            )

            del imgs_A, imgs_B, x_imgs
            del (
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            )
            torch.cuda.empty_cache()

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

    def _perturb_imgs(self, imgs: Tensor, silent: bool = False) -> Tensor:
        l2_loss = nn.MSELoss().cuda()
        x_imgs = imgs.clone().detach() + torch.randn_like(imgs) * 1e-5
        identity = self._get_imgs_identity(imgs)
        epsilon = (
            self.config.third_party.lowkey.epsilon
            * (torch.max(x_imgs) - torch.min(x_imgs))
            / 2
        )

        best_imgs, best_loss = torch.ones_like(imgs), float("inf")
        for epoch in range(self.config.third_party.lowkey.epochs):
            x_imgs = x_imgs.clone().detach().requires_grad_(True)

            pert_diff_loss = (
                self.config.third_party.lowkey.weight.perturb
                * self.utility.lpips_distance(x_imgs, imgs.detach()).sum()
            )

            identity_diff_loss = (
                self.config.third_party.lowkey.weight.identity
                * -(
                    l2_loss(self._get_imgs_identity(x_imgs), identity.detach())
                    + l2_loss(
                        self._get_imgs_identity(self.blur(x_imgs)),
                        identity.detach(),
                    )
                )
                / torch.norm(identity.detach(), p=2)
            )

            loss = pert_diff_loss + identity_diff_loss
            loss.backward()

            if x_imgs.grad is not None:
                grad_sign = x_imgs.grad.sign().clone().detach()
            else:
                grad_sign = torch.zeros_like(x_imgs)

            x_imgs = x_imgs.clone().detach() - epsilon * grad_sign

            x_imgs = torch.clamp(x_imgs, 0, 1)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_imgs = x_imgs

            if not silent:
                self.logger.info(
                    f"[Epoch {epoch+1:4}/{self.config.third_party.lowkey.epochs:4}]loss: {loss:.5f}({pert_diff_loss.item():.5f}, {identity_diff_loss.item():.5f})"
                )

        return best_imgs

    def _get_full_swap_results(
        self, imgs_A: Tensor, imgs_B: Tensor, pert_imgs_A: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        imgs_A_identity = self._get_imgs_identity(imgs_A)
        imgs_A_src_swap = self.target(None, imgs_B, imgs_A_identity, None, True)

        pert_imgs_A_identity = self._get_imgs_identity(pert_imgs_A)
        pert_imgs_A_src_swap = self.target(
            None, imgs_B, pert_imgs_A_identity, None, True
        )

        imgs_B_identity = self._get_imgs_identity(imgs_B)
        imgs_A_tgt_swap = self.target(None, imgs_A, imgs_B_identity, None, True)

        pert_imgs_A_tgt_swap = self.target(
            None, pert_imgs_A, imgs_B_identity, None, True
        )

        return (
            imgs_A_src_swap,
            pert_imgs_A_src_swap,
            imgs_A_tgt_swap,
            pert_imgs_A_tgt_swap,
        )
