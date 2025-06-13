import metric

from simswap.base import Base
from dataset import SampleDataset, MetricDataset_224, MetricDataset_512
from utils import save_tensor_imgs


import textwrap
import torch
from torch import tensor, nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from pathlib import Path


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            OmegaConf.to_yaml(OmegaConf.to_container(config.third_party, resolve=True))
        )

        self.pgd_rgb_limits = {"R": 0.075, "G": 0.03, "B": 0.075}
        self.pgd_loss_weights = {"pert": 1000, "identity": 10000, "latent": 0.1}
        self.pgd_loss_limits = {"latent": 30}

    # def __perturb_imgs(
    #     self, imgs: tensor, cloak_imgs: tensor, silent: bool = False
    # ) -> tensor:
    #     l2_loss = nn.MSELoss().cuda()
    #     x_imgs = imgs.clone().detach()
    #     cloak_identity = self._get_imgs_identity(cloak_imgs)
    #     imgs_latent_code = self.target.netG.encoder(x_imgs)
    #     epsilon = (
    #         self.config.third_party.defense.epsilon
    #         * (torch.max(x_imgs) - torch.min(x_imgs))
    #         / 2
    #     )
    #     limits = (
    #         tensor(
    #             [
    #                 self.config.third_party.defense.limit.R,
    #                 self.config.third_party.defense.limit.G,
    #                 self.config.third_party.defense.limit.B,
    #             ]
    #         )
    #         .view(1, 3, 1, 1)
    #         .cuda()
    #     )

    #     best_imgs, best_loss = None, float("inf")
    #     for epoch in range(self.config.third_party.defense.epochs):
    #         x_imgs.requires_grad = True

    #         pert_diff_loss = self.config.third_party.defense.weight.perturb * l2_loss(
    #             x_imgs, imgs.detach()
    #         )

    #         x_identity = self._get_imgs_identity(x_imgs)
    #         identity_diff_loss = (
    #             self.config.third_party.defense.weight.identity
    #             * l2_loss(x_identity, cloak_identity.detach())
    #         )

    #         x_latent_code = self.target.netG.encoder(x_imgs)
    #         context_diff_loss = (
    #             self.config.third_party.defense.weight.context
    #             * -torch.clamp(
    #                 l2_loss(x_latent_code, imgs_latent_code.detach()),
    #                 0,
    #                 self.config.third_party.defense.limit.context,
    #             )
    #         )

    #         loss = pert_diff_loss + identity_diff_loss + context_diff_loss
    #         loss.backward()

    #         x_imgs = (
    #             x_imgs.clone().detach() - epsilon * x_imgs.grad.sign().clone().detach()
    #         )

    #         x_imgs = torch.clamp(
    #             x_imgs,
    #             min=imgs - limits,
    #             max=imgs + limits,
    #         )
    #         x_imgs = torch.clamp(x_imgs, 0, 1)

    #         if loss.item() < best_loss:
    #             best_loss = loss.item()
    #             best_imgs = x_imgs

    #         if not silent:
    #             self.logger.info(
    #                 f"[Epoch {epoch+1:4}/{self.config.third_party.defense.epochs:4}]loss: {loss:.5f}({pert_diff_loss.item():.5f}, {identity_diff_loss.item():.5f}, {context_diff_loss.item():.5f})"
    #             )

    #     return best_imgs

    def __perturb_imgs(
        self, imgs: tensor, cloak_imgs: tensor, silent: bool = False
    ) -> tuple[tensor, tensor]:
        l2_loss = nn.MSELoss().cuda()
        x_imgs = imgs.clone().detach()
        best_anchor_identity = self._get_imgs_identity(cloak_imgs)
        imgs_latent_code = self.target.netG.encoder(x_imgs)
        epsilon = 0.005 * (torch.max(x_imgs) - torch.min(x_imgs)) / 2
        limits = (
            torch.tensor(
                [
                    self.pgd_rgb_limits["R"],
                    self.pgd_rgb_limits["G"],
                    self.pgd_rgb_limits["B"],
                ]
            )
            .view(1, 3, 1, 1)
            .cuda()
        )

        best_imgs, best_loss = None, float("inf")
        for epoch in range(1000):
            x_imgs.requires_grad = True

            pert_diff_loss = l2_loss(x_imgs, imgs.detach())
            x_identity = self._get_imgs_identity(x_imgs)
            identity_diff_loss = l2_loss(x_identity, best_anchor_identity.detach())
            x_latent_code = self.target.netG.encoder(x_imgs)
            latent_code_diff_loss = -torch.clamp(
                l2_loss(x_latent_code, imgs_latent_code.detach()),
                0,
                self.pgd_loss_limits["latent"],
            )

            loss = (
                self.pgd_loss_weights["pert"] * pert_diff_loss
                + self.pgd_loss_weights["identity"] * identity_diff_loss
                + self.pgd_loss_weights["latent"] * latent_code_diff_loss
            )
            loss.backward(retain_graph=False)

            x_imgs = (
                x_imgs.clone().detach() - epsilon * x_imgs.grad.sign().clone().detach()
            )

            x_imgs = torch.clamp(
                x_imgs,
                min=imgs - limits,
                max=imgs + limits,
            )
            x_imgs = torch.clamp(x_imgs, 0, 1)

            if loss < best_loss:
                best_loss = loss
                best_imgs = x_imgs

            if not silent:
                self.logger.info(
                    f"[Epoch {epoch+1:4}/{1000:4}]loss: {loss:.5f}({self.pgd_loss_weights['pert'] * pert_diff_loss.item():.5f}, {self.pgd_loss_weights['identity'] * identity_diff_loss.item():.5f}, {self.pgd_loss_weights['latent'] * latent_code_diff_loss.item():.5f})"
                )

        return best_imgs

    def __get_full_swap_results(
        self, imgs_A: tensor, imgs_B: tensor, pert_imgs_A: tensor
    ) -> tuple[tensor, tensor, tensor, tensor]:
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

    def __calculate_pgd_metric(
        self,
        imgs_A: tensor,
        imgs_B: tensor,
        x_imgs: tensor,
        cloak_imgs: tensor,
        imgs_A_src_swap: tensor,
        pert_imgs_A_src_swap: tensor,
        imgs_A_tgt_swap: tensor,
        pert_imgs_A_tgt_swap: tensor,
    ) -> tuple[dict, dict, dict, dict, dict]:
        pert_utilities = self.utility.calculate_utility(imgs_A, x_imgs)
        pert_as_src_swap_utilities = self.utility.calculate_utility(
            imgs_A_src_swap, pert_imgs_A_src_swap
        )
        pert_as_tgt_swap_utilities = self.utility.calculate_utility(
            imgs_A_tgt_swap, pert_imgs_A_tgt_swap
        )
        source_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs_A,
            x_imgs,
            imgs_A_src_swap,
            pert_imgs_A_src_swap,
            cloak_imgs,
        )
        target_effectivenesses = self.effectiveness.calculate_effectiveness(
            imgs_B, None, imgs_A_tgt_swap, pert_imgs_A_tgt_swap, None
        )

        return (
            pert_utilities,
            pert_as_src_swap_utilities,
            pert_as_tgt_swap_utilities,
            source_effectivenesses,
            target_effectivenesses,
        )

    def __generate_iter_utility_log(self, utilities: dict) -> str:
        return f"""
        {tuple(f'{v:.5f}' for _,v in utilities.items())}
        """.strip()

    def __generate_iter_effectiveness_log(self, effectiveness: dict) -> str:
        content = ""
        for effec in effectiveness:
            content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in effectiveness[effec].items())} "

        return content

    def sample(self) -> None:
        self.target.cuda().eval()

        dataset = SampleDataset(self.config.third_party.dataset.sample_dir)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self.__perturb_imgs(imgs_A, cloak_imgs, silent=False)

            (
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            ) = self.__get_full_swap_results(imgs_A, imgs_B, x_imgs)

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_b",
                    "pert_imgs",
                    "cloak_imgs",
                    "swap",
                    "pert_swap",
                    "rev\nswap",
                    "rev\npert_swap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    x_imgs,
                    cloak_imgs,
                    imgs_A_src_swap,
                    pert_imgs_A_src_swap,
                    imgs_A_tgt_swap,
                    pert_imgs_A_tgt_swap,
                ],
                False,
            )

            (
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            ) = self.__calculate_pgd_metric(
                imgs_A,
                imgs_B,
                x_imgs,
                cloak_imgs,
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            )

            self.logger.info(
                textwrap.dedent(
                    f"""
            pert utility(mse, psnr, ssim, lpips): {self.__generate_iter_utility_log(pert_utilities)}
            pert source utility(mse, psnr, ssim, lpips): {self.__generate_iter_utility_log(pert_as_src_swap_utilities)}
            pert target utility(mse, psnr, ssim, lpips): {self.__generate_iter_utility_log(pert_as_tgt_swap_utilities)}
            effectiveness tools: {list(self.effectiveness.candi_funcs.keys())}, source(pert, swap, pert_swap, cloak), target(swap, pert_swap)
            pert as swap source effectiveness: {self.__generate_iter_effectiveness_log(source_effectivenesses)}
            pert as swap target effectiveness: {self.__generate_iter_effectiveness_log(target_effectivenesses)}
            """
                )
            )

    def __merge_metric(
        self,
        data: dict,
        pert_utilities: dict,
        pert_as_src_swap_utilities: dict,
        pert_as_tgt_swap_utilities: dict,
        source_effectivenesses: dict,
        target_effectivenesses: dict,
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
        data["src_pert_swap_utility"] = tuple(
            x + y
            for x, y in zip(
                data["src_pert_swap_utility"],
                (
                    pert_as_src_swap_utilities["mse"],
                    pert_as_src_swap_utilities["psnr"],
                    pert_as_src_swap_utilities["ssim"],
                    pert_as_src_swap_utilities["lpips"],
                ),
            )
        )
        data["tgt_pert_swap_utility"] = tuple(
            x + y
            for x, y in zip(
                data["tgt_pert_swap_utility"],
                (
                    pert_as_tgt_swap_utilities["mse"],
                    pert_as_tgt_swap_utilities["psnr"],
                    pert_as_tgt_swap_utilities["ssim"],
                    pert_as_tgt_swap_utilities["lpips"],
                ),
            )
        )

        for effec in self.effectiveness.candi_funcs.keys():
            data["src_pert_swap_effectiveness"][effec] = {
                key1: (value1[0] + value2[0], value1[1] + value2[1])
                for (key1, value1), (key2, value2) in zip(
                    data["src_pert_swap_effectiveness"][effec].items(),
                    source_effectivenesses[effec].items(),
                )
            }
            data["tgt_pert_swap_effectiveness"][effec] = {
                key1: (value1[0] + value2[0], value1[1] + value2[1])
                for (key1, value1), (key2, value2) in zip(
                    data["tgt_pert_swap_effectiveness"][effec].items(),
                    target_effectivenesses[effec].items(),
                )
            }

    def __generate_summary_utility_log(self, data: dict, item: str, batch: int) -> str:
        return f"""
        {tuple(f'{x / (batch):.5f}' for x in data[item])}
        """.strip()

    def __generate_summary_effectiveness_log(self, data: dict, item: str) -> str:
        content = ""
        for effec in data[item]:
            content += f"{tuple(f'{v[0]/v[1]*100:.3f}/{v[1]:.0f}' for _,v in data[item][effec].items())} "

        return content

    def metric(
        self,
    ) -> None:
        data = metric.get_metric_data_template(self.effectiveness)

        dataset = MetricDataset_224(self.config.third_party.dataset.metric_224_dir)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            total_count += len(imgs_A)

            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            x_imgs = self.__perturb_imgs(imgs_A, cloak_imgs, silent=False)

            (
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            ) = self.__get_full_swap_results(imgs_A, imgs_B, x_imgs)

            (
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            ) = self.__calculate_pgd_metric(
                imgs_A,
                imgs_B,
                x_imgs,
                cloak_imgs,
                imgs_A_src_swap,
                pert_imgs_A_src_swap,
                imgs_A_tgt_swap,
                pert_imgs_A_tgt_swap,
            )

            self.__merge_metric(
                data,
                pert_utilities,
                pert_as_src_swap_utilities,
                pert_as_tgt_swap_utilities,
                source_effectivenesses,
                target_effectivenesses,
            )
            print(
                imgs_A.shape,
                imgs_B.shape,
                x_imgs.shape,
                cloak_imgs.shape,
                imgs_A_src_swap.shape,
                pert_imgs_A_src_swap.shape,
                imgs_A_tgt_swap.shape,
                pert_imgs_A_tgt_swap.shape,
            )

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_b",
                    "pert_imgs",
                    "cloak_imgs",
                    "swap",
                    "pert_swap",
                    "rev\nswap",
                    "rev\npert_swap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    x_imgs,
                    cloak_imgs,
                    imgs_A_src_swap,
                    pert_imgs_A_src_swap,
                    imgs_A_tgt_swap,
                    pert_imgs_A_tgt_swap,
                ],
                True,
            )

            del imgs_A, imgs_B, x_imgs, cloak_imgs
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
            pert utility: {self.__generate_iter_utility_log(pert_utilities)}
            pert as swap source utility: {self.__generate_iter_utility_log(pert_as_src_swap_utilities)}
            pert as swap target utility: {self.__generate_iter_utility_log(pert_as_tgt_swap_utilities)}
            pert as swap source effectiveness: {self.__generate_iter_effectiveness_log(source_effectivenesses)}
            pert as swap target effectiveness: {self.__generate_iter_effectiveness_log(target_effectivenesses)}
            """
            )

            self.logger.info(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            {self.__generate_summary_utility_log(data, 'pert_utility', idx)}
            {self.__generate_summary_utility_log(data, 'src_pert_swap_utility', idx)}
            {self.__generate_summary_utility_log(data, 'tgt_pert_swap_utility', idx)}
            {self.__generate_summary_effectiveness_log(data, 'src_pert_swap_effectiveness')}
            {self.__generate_summary_effectiveness_log(data, 'tgt_pert_swap_effectiveness')}
            """
            )
            quit()
