from src import metric
from src.faceswap.models import Autoencoder
from src.evaluate import Utility, Effectiveness

import cv2
import torch
import os
import numpy as np
import math
from os.path import join
from torch import nn, Tensor
from torchvision.utils import save_image


class PGD:
    def __init__(self, logger, config, model):
        self.logger = logger
        self.config = config
        self.model = model

        self.utility = Utility(logger, config)
        self.effectiveness = Effectiveness(logger, config)

        self.IDs = [1, 2, 3, 4, 5]

    def __get_imgs_path(self, identity: int) -> list[str]:
        imgs_name = sorted(
            os.listdir(join(self.config.third_party.dataset.data_dir, f"{identity}_64"))
        )
        imgs_path = [
            join(self.config.third_party.dataset.data_dir, f"{identity}_64", name)
            for name in imgs_name
        ]

        return imgs_path

    def __load_imgs_bgr(self, imgs_path: list[str]) -> Tensor:
        imgs = []
        for path in imgs_path:
            img = cv2.imread(path) / 255.0
            imgs.append(img)

        stacked_imgs = np.stack(imgs, axis=0)
        imgs_tensor = torch.from_numpy(stacked_imgs.transpose((0, 3, 1, 2)))

        return imgs_tensor.float().cuda()

    def __save_imgs_worker(self, batch: int, imgs_rgb: Tensor, name: str) -> Tensor:
        dir = join(self.config.log_dir, "image", name)
        os.makedirs(dir, exist_ok=True)
        for i, img in enumerate(imgs_rgb, start=1):
            save_image(img, join(dir, f"{batch}_{i}.png"))

        return imgs_rgb

    def save_imgs(
        self,
        id_i: int,
        id_j: int,
        batch: int,
        source_imgs_rgb: Tensor,
        target_imgs_rgb: Tensor,
        imgs_swap_rgb: Tensor,
        pert_imgs_rgb: Tensor,
        pert_swap_rgb: Tensor,
    ) -> None:
        self.__save_imgs_worker(batch, source_imgs_rgb, f"{id_i}_{id_j}_source")
        self.__save_imgs_worker(batch, target_imgs_rgb, f"{id_i}_{id_j}_target")
        self.__save_imgs_worker(batch, imgs_swap_rgb, f"{id_i}_{id_j}_swap")
        self.__save_imgs_worker(batch, pert_imgs_rgb, f"{id_i}_{id_j}_pert")
        self.__save_imgs_worker(batch, pert_swap_rgb, f"{id_i}_{id_j}_pert_swap")

        summary = torch.cat(
            (
                source_imgs_rgb[:16],
                target_imgs_rgb[:16],
                imgs_swap_rgb[:16],
                pert_imgs_rgb[:16],
                pert_swap_rgb[:16],
            ),
            dim=0,
        )
        save_image(
            summary,
            join(self.config.log_dir, "image", f"{id_i}_{id_j}_summary_{batch}.png"),
            nrow=len(source_imgs_rgb[:16]),
        )

    def __get_source_imgs(self, identity: int, count: int) -> Tensor:
        imgs_path = [join(self.config.third_party.dataset.data_dir, f"{identity}.png")]
        imgs = self.__load_imgs_bgr(imgs_path)

        return imgs.repeat(count, 1, 1, 1)

    def __merge_metric(
        self,
        data: dict,
        pert_utility: dict,
        pert_swap_utility: dict,
        pert_effectiveness: dict,
    ) -> None:
        data["pert_utility"] = tuple(
            x + y
            for x, y in zip(
                data["pert_utility"],
                pert_utility.values(),
            )
        )
        data["pert_swap_utility"] = tuple(
            x + y
            for x, y in zip(
                data["pert_swap_utility"],
                pert_swap_utility.values(),
            )
        )
        for effec in self.effectiveness.candi_funcs.keys():
            data["pert_effectiveness"][effec] = {
                key1: (value1[0] + value2[0], value1[1] + value2[1])
                for (key1, value1), (key2, value2) in zip(
                    data["pert_effectiveness"][effec].items(),
                    pert_effectiveness[effec].items(),
                )
            }

    def __bgr_to_rgb(self, imgs: Tensor) -> Tensor:
        return imgs[:, [2, 1, 0], :, :]

    def run(self) -> None:
        limits = (
            torch.tensor(
                [
                    self.config.third_party.defense.limit.R,
                    self.config.third_party.defense.limit.G,
                    self.config.third_party.defense.limit.B,
                ]
            )
            .view(1, 3, 1, 1)
            .cuda()
        )
        self.model.eval()
        batch_size = self.config.third_party.defense.batch_size
        l2_loss = nn.MSELoss().cuda()
        for id_i in self.IDs:
            data = {
                "pert_utility": (0, 0, 0, 0),
                "pert_swap_utility": (0, 0, 0, 0),
                "pert_effectiveness": {},
            }
            for effec in self.effectiveness.candi_funcs.keys():
                data["pert_effectiveness"][effec] = {
                    "pert": (0, 0),
                    "swap": (0, 0),
                    "pert_swap": (0, 0),
                }
            imgs_path = self.__get_imgs_path(id_i)
            total_batch = math.ceil(len(imgs_path) / batch_size)
            batch_count = 0
            other_ids = [x for x in self.IDs if x != id_i]
            for id_j in other_ids:
                for batch in range(1, total_batch + 1):
                    batch_count += 1
                    iter_imgs_path = imgs_path[
                        (batch - 1) * batch_size : batch * batch_size
                    ]
                    imgs = self.__load_imgs_bgr(iter_imgs_path)
                    latent = self.model.encoder(imgs)
                    swap_imgs = self.model(imgs, id_j).detach()

                    x_imgs = imgs.clone().detach()
                    initial_noise = 1e-6 * torch.ones_like(x_imgs)
                    x_imgs = torch.clamp(x_imgs + initial_noise, 0, 1)

                    epsilon = (
                        self.config.third_party.defense.epsilon
                        * (torch.max(imgs) - torch.min(imgs))
                        / 2
                    )
                    best_imgs, best_loss = torch.ones_like(imgs), float("inf")
                    for epoch in range(1, self.config.third_party.defense.epochs + 1):
                        x_imgs = x_imgs.clone().detach().requires_grad_(True)

                        x_latent = self.model.encoder(x_imgs)
                        pert_diff_loss = l2_loss(x_imgs, imgs)
                        latent_diff_loss = -torch.clamp(
                            l2_loss(x_latent, latent),
                            0.0,
                            self.config.third_party.defense.limit.latent,
                        )

                        loss = (
                            self.config.third_party.defense.weight.perturb
                            * pert_diff_loss
                            + self.config.third_party.defense.weight.latent
                            * latent_diff_loss
                        )
                        if x_imgs.grad is not None:
                            x_imgs.grad.zero_()
                        loss.backward(retain_graph=True)

                        grad_sign = (
                            x_imgs.grad.sign()
                            if x_imgs.grad is not None
                            else torch.zeros_like(x_imgs)
                        )
                        x_imgs = x_imgs.clone().detach() - epsilon * grad_sign
                        x_imgs = torch.clamp(x_imgs, imgs - limits, imgs + limits)
                        x_imgs = torch.clamp(x_imgs, 0, 1)

                        if loss < best_loss:
                            best_loss = loss
                            best_imgs = x_imgs

                        if epoch % self.config.third_party.worker.log_interval == 0:
                            self.logger.info(
                                f"[{id_i}->{id_j}][{batch:3d}/{total_batch:3d}][{epoch:3d}/{self.config.third_party.defense.epochs:3d}]|loss: {loss:.5f}({pert_diff_loss.item() * self.config.third_party.defense.weight.perturb:.5f}, {latent_diff_loss.item() * self.config.third_party.defense.weight.latent:.5f})({pert_diff_loss.item():.5f}, {latent_diff_loss.item():.5f})"
                            )

                    swap_x = self.model(best_imgs, id_j).detach()
                    source_imgs = self.__get_source_imgs(id_j, len(imgs))

                    source_imgs = self.__bgr_to_rgb(source_imgs)
                    target_imgs = self.__bgr_to_rgb(imgs)
                    pert_imgs = self.__bgr_to_rgb(best_imgs)
                    swap_imgs = self.__bgr_to_rgb(swap_imgs)
                    pert_swap_imgs = self.__bgr_to_rgb(swap_x)

                    self.save_imgs(
                        id_i,
                        id_j,
                        batch,
                        source_imgs,
                        target_imgs,
                        swap_imgs,
                        pert_imgs,
                        pert_swap_imgs,
                    )

                    pert_utility = self.utility.calculate_utility(
                        pert_imgs, target_imgs
                    )
                    pert_swap_utility = self.utility.calculate_utility(
                        pert_swap_imgs, swap_imgs
                    )
                    effectiveness = self.effectiveness.calculate_effectiveness(
                        source_imgs, target_imgs, pert_imgs, swap_imgs, pert_swap_imgs
                    )
                    self.__merge_metric(
                        data, pert_utility, pert_swap_utility, effectiveness
                    )

                    self.logger.info(
                        f"""
                    {id_i}->{id_j} utility(mse, psnr, ssim, lpips), effectiveness{self.effectiveness.candi_funcs.keys()} (pert, swap, pert_swap)
                    pert utility: {metric.generate_iter_utility_log(pert_utility)}
                    pert swap utility: {metric.generate_iter_utility_log(pert_swap_utility)}
                    pert effectiveness: {metric.generate_iter_effectiveness_log(effectiveness)}
                    """
                    )

                    self.logger.info(
                        f"""
                    {id_i}->{id_j} Batch {batch:3}/{total_batch:3}, {batch_size * batch_count} pairs of pictures
                    {metric.generate_summary_utility_log(data, 'pert_utility', batch_count)}
                    {metric.generate_summary_utility_log(data, 'pert_swap_utility', batch_count)}
                    {metric.generate_summary_effectiveness_log(data, 'pert_effectiveness')}
                    """
                    )
                self.logger.info(
                    f"""
                    {id_i}->{id_j}
                    {metric.generate_summary_utility_log(data, 'pert_utility', batch_count)}
                    {metric.generate_summary_utility_log(data, 'pert_swap_utility', batch_count)}
                    {metric.generate_summary_effectiveness_log(data, 'pert_effectiveness')}
                    """
                )
            self.logger.info(
                f"""
                    {id_i}
                    {metric.generate_summary_utility_log(data, 'pert_utility', batch_count)}
                    {metric.generate_summary_utility_log(data, 'pert_swap_utility', batch_count)}
                    {metric.generate_summary_effectiveness_log(data, 'pert_effectiveness')}
                    """
            )


class Defense:
    def __init__(self, logger, config):
        super(Defense, self).__init__()
        self.logger = logger
        self.config = config

        self.IDs = [1, 2, 3, 4, 5]

        self.model = Autoencoder(len(self.IDs)).cuda()
        self.model.load_state_dict(
            torch.load(self.config.third_party.defense.model_path)
        )

    def metric(self):
        pgd = PGD(self.logger, self.config, self.model)
        pgd.run()
