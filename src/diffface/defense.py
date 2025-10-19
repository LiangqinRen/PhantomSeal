from src.diffface.base import Base
from src.dataset import FFHQDataset
from src.utils import save_tensor_imgs
from src.evaluate import Utility, Effectiveness, Cloak

import torch
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

        self.segment_ids = [1, 2, 3, 4, 5, 10, 11, 12, 13]

    def metric(self) -> None:
        dataset = FFHQDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs, silent=False)

            results = self._face_swap_per_image(imgs_A, imgs_B)
            rev_results = self._face_swap_per_image(imgs_B, imgs_A)
            cloak_results = self._face_swap_per_image(cloak_imgs, imgs_B)
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
                    "cloak_swap",
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
                    cloak_results,
                    x_imgs,
                    pert_src_results,
                    pert_tgt_results,
                ],
                only_save_summary=True,
            )

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
        l1_loss = nn.L1Loss().cuda()
        l2_loss = nn.MSELoss().cuda()
        x_imgs = imgs.clone().detach() + torch.randn_like(imgs) * 1e-5
        cloak_identity = self.get_imgs_identity(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                cloak_imgs
            )
        )

        origin_mask = self.netSeg(
            self.spNorm(
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(imgs)
            )
        )[0]
        epsilon = (
            self.config.third_party.defense.epsilon
            * (torch.max(x_imgs) - torch.min(x_imgs))
            / 2
        )

        best_imgs, best_loss = torch.ones_like(imgs), float("inf")
        for epoch in range(self.config.third_party.defense.epochs):
            x_imgs = x_imgs.clone().detach().requires_grad_(True)

            pert_diff_loss = self.config.third_party.defense.weight.perturb * l2_loss(
                x_imgs, imgs.detach()
            )

            x_identity = self.get_imgs_identity(
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                    x_imgs
                )
            )
            identity_diff_loss = (
                self.config.third_party.defense.weight.identity
                * l2_loss(x_identity, cloak_identity.detach())
            )

            x_mask = self.netSeg(
                self.spNorm(
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                        x_imgs
                    )
                )
            )[0]
            seg_loss = torch.tensor(0).to(self.device).float()
            for id in self.segment_ids:
                seg_loss += l2_loss(
                    x_mask[:, id, :, :],
                    torch.zeros_like(
                        x_mask[:, id, :, :]
                    ),  # origin_mask[:, id, :, :].detach()
                )
            mask_diff_loss = self.config.third_party.defense.weight.context * seg_loss

            loss = pert_diff_loss + identity_diff_loss + mask_diff_loss
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
                    f"[Epoch {epoch+1:4}/{self.config.third_party.defense.epochs}]loss: {loss:.5f}({pert_diff_loss.item():.5f}, {identity_diff_loss.item():.5f}, {mask_diff_loss.item():.5f})"
                )

        return best_imgs
