import metric
from unify.base import Base
from dataset import MetricDataset
from utils import save_tensor_imgs


import torch
import torch.nn.functional as F
from torch import tensor, nn
from torch.utils.data import DataLoader
from pathlib import Path


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def metric(
        self,
    ) -> None:
        metric_data = self._get_unify_metric_template()

        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self._perturb_imgs(imgs_A, cloak_imgs, silent=True)

            (
                simswap_swap,
                simswap_cloak_swap,
                simswap_pert_swap,
                hififace_swap,
                hififace_cloak_swap,
                hififace_pert_swap,
                faceshifter_swap,
                faceshifter_cloak_swap,
                faceshifter_pert_swap,
            ) = self._get_full_swap_results(imgs_A, imgs_B, x_imgs, cloak_imgs)

            (
                faceshifter_swap,
                faceshifter_cloak_swap,
                faceshifter_pert_swap,
                (
                    imgs_A,
                    imgs_B,
                    x_imgs,
                    cloak_imgs,
                    simswap_swap,
                    simswap_cloak_swap,
                    simswap_pert_swap,
                    hififace_swap,
                    hififace_cloak_swap,
                    hififace_pert_swap,
                ),
            ) = self._align_valid_imgs(
                faceshifter_swap,
                faceshifter_cloak_swap,
                faceshifter_pert_swap,
                [
                    imgs_A,
                    imgs_B,
                    x_imgs,
                    cloak_imgs,
                    simswap_swap,
                    simswap_cloak_swap,
                    simswap_pert_swap,
                    hififace_swap,
                    hififace_cloak_swap,
                    hififace_pert_swap,
                ],
            )

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "pert_imgs",
                    "cloak_imgs",
                    "simswap\nswap",
                    "simswap\ncloak_swap",
                    "simswap\npert_swap",
                    "hififace\nswap",
                    "hififace\ncloak_swap",
                    "hififace\npert_swap",
                    "faceshifter\nswap",
                    "faceshifter\ncloak_swap",
                    "faceshifter\npert_swap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    x_imgs,
                    cloak_imgs,
                    simswap_swap,
                    simswap_cloak_swap,
                    simswap_pert_swap,
                    hififace_swap,
                    hififace_cloak_swap,
                    hififace_pert_swap,
                    faceshifter_swap,
                    faceshifter_cloak_swap,
                    faceshifter_pert_swap,
                ],
                only_save_summary=True,
            )

            total_count += len(faceshifter_swap)

            (
                pert_utilities,
                simswap_identity_effec,
                hififace_identity_effec,
                faceshifter_identity_effec,
            ) = self._calculate_cross_model_metric(
                imgs_A,
                x_imgs,
                cloak_imgs,
                simswap_swap,
                simswap_pert_swap,
                hififace_swap,
                hififace_pert_swap,
                faceshifter_swap,
                faceshifter_pert_swap,
            )

            self._merge_unify_metric(
                metric_data,
                pert_utilities,
                simswap_identity_effec,
                faceshifter_identity_effec,
                hififace_identity_effec,
            )

            del imgs_A, imgs_B, x_imgs, cloak_imgs
            del (
                simswap_swap,
                simswap_cloak_swap,
                simswap_pert_swap,
                hififace_swap,
                hififace_cloak_swap,
                hififace_pert_swap,
                faceshifter_swap,
                faceshifter_cloak_swap,
                faceshifter_pert_swap,
            )
            torch.cuda.empty_cache()

            self.logger.info(
                f"""
            utility(mse, psnr, ssim, lpips), effectiveness{self.effectiveness.candi_funcs.keys()} source(pert, swap, pert_swap, anchor) target(swap, pert_swap)
            pert utility: {metric.generate_iter_utility_log(pert_utilities)}
            simswap identity effectiveness: {metric.generate_iter_effectiveness_log(simswap_identity_effec)}
            faceshifter identity effectiveness: {metric.generate_iter_effectiveness_log(faceshifter_identity_effec)}
            hififace identity effectiveness: {metric.generate_iter_effectiveness_log(hififace_identity_effec)}
            """
            )

            self.logger.info(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            {metric.generate_summary_utility_log(metric_data, 'pert_utility', idx)}
            {metric.generate_summary_effectiveness_log(metric_data, 'simswap')}
            {metric.generate_summary_effectiveness_log(metric_data, 'faceshifter')}
            {metric.generate_summary_effectiveness_log(metric_data, 'hififace')}
            """
            )

    def _perturb_imgs(
        self, imgs: tensor, cloak_imgs: tensor, silent: bool = False
    ) -> tensor:
        l2_loss = nn.MSELoss().cuda()
        x_imgs = imgs.clone().detach()

        simswap_cloak_identity = self._get_imgs_identity(cloak_imgs)
        faceshifter_cloak_identity = self.arcface(
            F.interpolate(
                cloak_imgs[:, :, 19:237, 19:237],
                (112, 112),
                mode="bilinear",
                align_corners=True,
            )
        )
        hififace_cloak_3d = self.net.generator.id_extractor.f_3d(cloak_imgs)[:, :80]
        hififace_cloak_id = F.normalize(
            self.net.generator.id_extractor.f_id(
                F.interpolate((cloak_imgs - 0.5) / 0.5, size=112, mode="bilinear")
            ),
            dim=-1,
            p=2,
        )

        epsilon = (
            self.config.third_party.defense.epsilon
            * (torch.max(x_imgs) - torch.min(x_imgs))
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

        best_imgs, best_loss = None, float("inf")
        for epoch in range(self.config.third_party.defense.epochs):
            x_imgs.requires_grad = True

            pert_diff_loss = self.config.third_party.defense.weight.perturb * l2_loss(
                x_imgs, imgs.detach()
            )

            x_simswap_identity = self._get_imgs_identity(x_imgs)
            simswap_identity_loss = (
                self.config.third_party.defense.weight.simswap
                * l2_loss(x_simswap_identity, simswap_cloak_identity.detach())
            )

            x_hififace_3d = self.net.generator.id_extractor.f_3d(x_imgs)[:, :80]
            x_hififace_id = F.normalize(
                self.net.generator.id_extractor.f_id(
                    F.interpolate((x_imgs - 0.5) / 0.5, size=112, mode="bilinear")
                ),
                dim=-1,
                p=2,
            )
            hififace_3d_loss = (
                self.config.third_party.defense.weight.hififace_3d
                * l2_loss(x_hififace_3d, hififace_cloak_3d.detach())
            )
            hififace_id_loss = (
                self.config.third_party.defense.weight.hififace_id
                * l2_loss(x_hififace_id, hififace_cloak_id.detach())
            )

            x_faceshifter_identity = self.arcface(
                F.interpolate(
                    x_imgs[:, :, 19:237, 19:237],
                    (112, 112),
                    mode="bilinear",
                    align_corners=True,
                )
            )
            faceshifter_identity_loss = (
                self.config.third_party.defense.weight.faceshifter
                * l2_loss(x_faceshifter_identity, faceshifter_cloak_identity.detach())
            )

            loss = (
                pert_diff_loss
                + simswap_identity_loss
                + hififace_3d_loss
                + hififace_id_loss
                + faceshifter_identity_loss
            )
            loss.backward()

            x_imgs = (
                x_imgs.clone().detach() - epsilon * x_imgs.grad.sign().clone().detach()
            )

            x_imgs = torch.clamp(
                x_imgs,
                min=imgs - limits,
                max=imgs + limits,
            )
            x_imgs = torch.clamp(x_imgs, 0, 1)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_imgs = x_imgs

            if not silent:
                self.logger.info(
                    f"[Epoch {epoch+1:4}/{self.args.epochs:4}]loss: {loss:.5f}({pert_diff_loss.item():.5f}, {simswap_identity_loss.item():.5f}, {hififace_3d_loss.item():.5f},{hififace_id_loss.item():.5f}, {faceshifter_identity_loss.item():.5f})"
                )

        return best_imgs

    def _get_full_swap_results(
        self, imgs_A: tensor, imgs_B: tensor, pert_imgs_A: tensor, cloak_imgs: tensor
    ) -> tuple[tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor]:
        simswap_swap = self._simswap_swapface(imgs_A, imgs_B)
        simswap_pert_swap = self._simswap_swapface(pert_imgs_A, imgs_B)
        simswap_cloak_swap = self._simswap_swapface(cloak_imgs, imgs_B)

        hififace_swap = self._hififace_swapface(imgs_A, imgs_B)
        hififace_pert_swap = self._hififace_swapface(pert_imgs_A, imgs_B)
        hififace_cloak_swap = self._hififace_swapface(cloak_imgs, imgs_B)

        faceshifter_swap = self._faceshifter_swapface(imgs_A, imgs_B)
        faceshifter_pert_swap = self._faceshifter_swapface(pert_imgs_A, imgs_B)
        faceshifter_cloak_swap = self._faceshifter_swapface(cloak_imgs, imgs_B)

        return (
            simswap_swap,
            simswap_pert_swap,
            simswap_cloak_swap,
            hififace_swap,
            hififace_pert_swap,
            hififace_cloak_swap,
            faceshifter_swap,
            faceshifter_pert_swap,
            faceshifter_cloak_swap,
        )

    def _align_valid_imgs(
        self,
        faceshifter_swap: tensor,
        faceshifter_cloak_swap: tensor,
        faceshifter_pert_swap: tensor,
        others: list[tensor],
    ) -> tuple[tensor, tensor, list[tensor]]:
        faceshifter_swap_mask = (
            faceshifter_swap.view(faceshifter_swap.size(0), -1).abs().sum(dim=1) > 0
        )
        faceshifter_cloak_swap_mask = (
            faceshifter_cloak_swap.view(faceshifter_cloak_swap.size(0), -1)
            .abs()
            .sum(dim=1)
            > 0
        )
        faceshifter_pert_swap_mask = (
            faceshifter_pert_swap.view(faceshifter_pert_swap.size(0), -1)
            .abs()
            .sum(dim=1)
            > 0
        )

        valid_mask = (
            faceshifter_swap_mask
            & faceshifter_cloak_swap_mask
            & faceshifter_pert_swap_mask
        )

        filter_faceshifter_swap = faceshifter_swap[valid_mask]
        filter_faceshifter_cloak_swap = faceshifter_cloak_swap[valid_mask]
        filter_faceshifter_pert_swap = faceshifter_pert_swap[valid_mask]

        for i in range(len(others)):
            others[i] = others[i][valid_mask]

        return (
            filter_faceshifter_swap,
            filter_faceshifter_cloak_swap,
            filter_faceshifter_pert_swap,
            others,
        )

    def _calculate_cross_model_metric(
        self,
        imgs_A: tensor,
        x_imgs: tensor,
        cloak_imgs: tensor,
        simswap_swap: tensor,
        simswap_pert_swap: tensor,
        faceshifter_swap: tensor,
        faceshifter_pert_swap: tensor,
        hififace_swap: tensor,
        hififace_pert_swap: tensor,
    ) -> tuple[dict, dict, dict, dict, dict]:
        pert_utilities = self.utility.calculate_utility(imgs_A, x_imgs)
        simswap_identity_effec = self.effectiveness.calculate_effectiveness(
            imgs_A,
            x_imgs,
            simswap_swap,
            simswap_pert_swap,
            cloak_imgs,
        )
        faceshifter_identity_effec = self.effectiveness.calculate_effectiveness(
            imgs_A,
            x_imgs,
            faceshifter_swap,
            faceshifter_pert_swap,
            cloak_imgs,
        )
        hififace_identity_effec = self.effectiveness.calculate_effectiveness(
            imgs_A,
            x_imgs,
            hififace_swap,
            hififace_pert_swap,
            cloak_imgs,
        )

        return (
            pert_utilities,
            simswap_identity_effec,
            faceshifter_identity_effec,
            hififace_identity_effec,
        )

    def _get_unify_metric_template(self) -> dict:
        data = {
            "pert_utility": (0, 0, 0, 0),
            "simswap": {},
            "faceshifter": {},
            "hififace": {},
        }

        for effec in self.effectiveness.candi_funcs.keys():
            data["simswap"][effec] = {
                "pert": (0, 0),
                "swap": (0, 0),
                "pert_swap": (0, 0),
                "cloak": (0, 0),
            }
            data["faceshifter"][effec] = {
                "pert": (0, 0),
                "swap": (0, 0),
                "pert_swap": (0, 0),
                "cloak": (0, 0),
            }
            data["hififace"][effec] = {
                "pert": (0, 0),
                "swap": (0, 0),
                "pert_swap": (0, 0),
                "cloak": (0, 0),
            }
        return data

    def _merge_unify_metric(
        self,
        data: dict,
        pert_utilities: dict,
        simswap_identity_effec: dict,
        faceshifter_identity_effec: dict,
        hififace_identity_effec: dict,
    ) -> None:
        data["pert_utility"] = tuple(
            x + y
            for x, y in zip(
                data["pert_utility"],
                (
                    pert_utilities["mse"],
                    pert_utilities["psnr"],
                    pert_utilities["ssim"],
                    pert_utilities["lpips"],
                ),
            )
        )

        for effec in self.effectiveness.candi_funcs.keys():
            data["simswap"][effec] = {
                key1: (value1[0] + value2[0], value1[1] + value2[1])
                for (key1, value1), (key2, value2) in zip(
                    data["simswap"][effec].items(),
                    simswap_identity_effec[effec].items(),
                )
            }
            data["faceshifter"][effec] = {
                key1: (value1[0] + value2[0], value1[1] + value2[1])
                for (key1, value1), (key2, value2) in zip(
                    data["faceshifter"][effec].items(),
                    faceshifter_identity_effec[effec].items(),
                )
            }
            data["hififace"][effec] = {
                key1: (value1[0] + value2[0], value1[1] + value2[1])
                for (key1, value1), (key2, value2) in zip(
                    data["hififace"][effec].items(),
                    hififace_identity_effec[effec].items(),
                )
            }
