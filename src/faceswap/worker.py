from faceswap.umeyama import umeyama
from faceswap.models import Autoencoder

import cv2
import os
import torch
import random
import numpy as np
from torch import nn, optim, Tensor
from os.path import join
from torchvision.utils import save_image
from pathlib import Path


class Worker:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

        self.IDs = [1, 2, 3, 4, 5]

    def extract(self):
        for input in self.IDs:
            data_dir = self.config.third_party.data_dir
            source_dir = join(data_dir, f"{input}_256")
            target_dir = join(data_dir, f"{input}_64")
            os.makedirs(target_dir, exist_ok=True)
            imgs_name = os.listdir(source_dir)
            for _, name in enumerate(imgs_name):
                img = cv2.imread(join(source_dir, name)) / 255.0

                # 🤣 magic!
                range_ = np.linspace(128 - 80, 128 + 80, 5)
                mapx = np.broadcast_to(range_, (5, 5))
                mapy = mapx.T

                mapx = mapx + np.random.normal(size=(5, 5), scale=5)
                mapy = mapy + np.random.normal(size=(5, 5), scale=5)

                src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
                dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
                mat = umeyama(src_points, dst_points, True)[0:2]
                target_img = cv2.warpAffine(img, mat, (64, 64))

                cv2.imwrite(join(target_dir, name), target_img * 255)

    def train(self):
        model = self._load_model(len(self.IDs))
        optimizers = {id: self._get_optimizer(model, id) for id in self.IDs}

        id_imgs_256, id_imgs_64 = {}, {}
        for id in self.IDs:
            imgs_256, imgs_64 = self._load_train_imgs(id)
            id_imgs_256[id] = imgs_256
            id_imgs_64[id] = imgs_64

        l2_loss = nn.MSELoss().cuda()
        best_model, best_loss = None, float("inf")
        checkpoint_dir = join(self.config.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        for epoch in range(1, self.config.third_party.worker.epochs + 1):
            losses = {}
            id_target_imgs = {}
            for id in self.IDs:
                warp_imgs, target_imgs = self._get_iter_imgs(
                    id_imgs_256[id], id_imgs_64[id]
                )
                id_target_imgs[id] = target_imgs
                optimizers[id].zero_grad()
                warp_imgs = model(warp_imgs, id)

                losses[id] = l2_loss(warp_imgs, target_imgs)
                losses[id].backward()
                optimizers[id].step()

            loss_item = [loss.item() for _, loss in losses.items()]
            loss = sum(loss_item)
            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()
                torch.save(best_model, join(checkpoint_dir, "faceswap.pth"))
                self.logger.info(
                    f"Save the model at epoch {epoch} with loss {loss:.5f}"
                )

            loss_item = [f"{loss:.5f}" for loss in loss_item]
            self.logger.info(
                f"[{epoch:5d}/{self.config.third_party.worker.epochs:5d}]Loss: {loss:.5f}({loss_item})"
            )

            if epoch % self.config.third_party.worker.log_interval == 0:
                results = []
                for id in self.IDs:
                    rebuild_imgs = model(id_target_imgs[id], id)
                    random_id = random.choice([n for n in self.IDs if n != id])
                    swap_imgs = model(id_target_imgs[id], random_id)
                    results.extend(
                        [id_target_imgs[id][:16], rebuild_imgs[:16], swap_imgs[:16]]
                    )

                results = [self._bgr_to_rgb(result) for result in results]
                summary = torch.cat(
                    results,
                    dim=0,
                )
                save_image(
                    summary,
                    join(self.config.log_dir, "image", f"summary_{epoch}.png"),
                    nrow=len(id_target_imgs[1][:16]),
                )

    def test(self):
        model = self._load_model(len(self.IDs))
        for i in self.IDs:
            imgs_i = self._load_test_imgs(i)
            imgs_i_rebuild = model(imgs_i, i)
            swap_ids = [x for x in self.IDs if x != i]
            swap_imgs = []
            for j in swap_ids:
                imgs_i_swap_j = model(imgs_i_rebuild, j)
                swap_imgs.append(imgs_i_swap_j)

            self._save_imgs(i, imgs_i, imgs_i_rebuild, swap_ids, swap_imgs)

    def _load_model(self, id_count: int) -> Autoencoder:
        model = Autoencoder(id_count).cuda()
        if self.config.third_party.function == "test":
            model_path = self.config.third_party.model_path
            try:
                model.load_state_dict(torch.load(model_path))
                self.logger.info(f"load the model {model_path}")
            except FileNotFoundError:
                self.logger.error("can't find the model {model_path}")

        return model

    def _get_optimizer(self, model: Autoencoder, decoder_index: int) -> optim.Optimizer:
        optimizer = optim.Adam(
            [
                {"params": model.encoder.parameters()},
                {"params": model.decoders[str(decoder_index)].parameters()},
            ],
            lr=5e-5,
            betas=(0.5, 0.999),
        )

        return optimizer

    def _load_train_imgs(
        self, identity: int
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        imgs_256_dir = join(self.config.third_party.data_dir, f"{identity}_256")
        imgs_name = os.listdir(imgs_256_dir)

        imgs_256 = [cv2.imread(join(imgs_256_dir, name)) / 255.0 for name in imgs_name]

        clean_imgs_64_dir = join(self.config.third_party.data_dir, f"{identity}_64")
        clean_imgs_name = os.listdir(clean_imgs_64_dir)
        if self.config.third_party.worker.pert_prefix is None:
            imgs_64 = [
                cv2.imread(join(clean_imgs_64_dir, name)) / 255.0
                for name in clean_imgs_name
            ]
        else:
            poison_imgs_64_dir = join(
                self.config.third_party.data_dir,
                f"{self.config.third_party.worker.pert_prefix}_{identity}_64",
            )
            poison_imgs_name = os.listdir(poison_imgs_64_dir)
            poison_imgs_name = random.sample(
                poison_imgs_name,
                int(
                    len(poison_imgs_name)
                    * self.config.third_party.worker.poison_percent
                    / 100
                ),
            )
            imgs_64 = [
                cv2.imread(join(poison_imgs_64_dir, name)) / 255.0
                for name in poison_imgs_name
            ]

            clean_imgs_name = random.sample(
                clean_imgs_name, len(clean_imgs_name) - len(poison_imgs_name)
            )
            imgs_64.extend(
                [
                    cv2.imread(join(clean_imgs_64_dir, name)) / 255.0
                    for name in clean_imgs_name
                ]
            )

        return imgs_256, imgs_64

    def _random_transform(
        self, img_256: np.ndarray, img_64: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        trans_range = {"rotation": 10, "zoom": 0.05, "shift": 0.05, "flip": 0.4}

        rotation = np.random.uniform(-trans_range["rotation"], trans_range["rotation"])
        scale = np.random.uniform(1 - trans_range["zoom"], 1 + trans_range["zoom"])
        shift_x = np.random.uniform(-trans_range["shift"], trans_range["shift"])
        shift_y = np.random.uniform(-trans_range["shift"], trans_range["shift"])

        mat_256 = cv2.getRotationMatrix2D((256 // 2, 256 // 2), rotation, scale)
        mat_256[:, 2] += (shift_x * 256, shift_y * 256)
        img_256_transform = cv2.warpAffine(
            img_256, mat_256, (256, 256), borderMode=cv2.BORDER_REPLICATE
        )

        mat_64 = cv2.getRotationMatrix2D((64 // 2, 64 // 2), rotation, scale)
        mat_64[:, 2] += (shift_x * 64, shift_y * 64)
        img_64_transform = cv2.warpAffine(
            img_64, mat_64, (64, 64), borderMode=cv2.BORDER_REPLICATE
        )

        return img_256_transform, img_64_transform

    def _warp_training_data(
        self, img_256: np.ndarray, img_64: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        img_256_trans, img_64_trans = self._random_transform(img_256, img_64)

        map_range = np.linspace(128 - 80, 128 + 80, 5)
        mapx = np.broadcast_to(map_range, (5, 5))
        mapy = mapx.T
        mapx = mapx + np.random.normal(size=(5, 5), scale=5)
        mapy = mapy + np.random.normal(size=(5, 5), scale=5)
        interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype("float32")
        interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype("float32")

        warped_img = cv2.remap(
            img_256_trans, interp_mapx, interp_mapy, cv2.INTER_LINEAR
        )

        return warped_img, img_64_trans

    def _get_iter_imgs(
        self, imgs_256: list[np.ndarray], imgs_64: list[np.ndarray]
    ) -> tuple[Tensor, Tensor]:
        indices = np.random.randint(
            len(imgs_256), size=self.config.third_party.worker.batch_size
        )
        warped_imgs = np.empty(
            (self.config.third_party.worker.batch_size,) + imgs_64[0].shape,
            imgs_64[0].dtype,
        )
        target_imgs = np.empty(
            (self.config.third_party.worker.batch_size,) + imgs_64[0].shape,
            imgs_64[0].dtype,
        )
        for i, index in enumerate(indices):
            img_256, img_64 = imgs_256[index], imgs_64[index]
            warped_img, target_img = self._warp_training_data(img_256, img_64)

            warped_imgs[i] = warped_img
            target_imgs[i] = target_img

        warped_tensor = torch.from_numpy(warped_imgs.transpose((0, 3, 1, 2)))
        target_tensor = torch.from_numpy(target_imgs.transpose((0, 3, 1, 2)))

        return warped_tensor.cuda().float(), target_tensor.cuda().float()

    def _bgr_to_rgb(self, imgs: Tensor) -> Tensor:
        return imgs[:, [2, 1, 0], :, :]

    def _load_test_imgs(self, identity: int, img_size: int = 64) -> Tensor:
        imgs_64_dir = join(self.config.third_party.data_dir, f"{identity}_{img_size}")
        imgs_name = os.listdir(imgs_64_dir)
        imgs_64 = [cv2.imread(join(imgs_64_dir, name)) / 255.0 for name in imgs_name]

        stacked_imgs = np.stack(imgs_64, axis=0)
        imgs_tensor = torch.from_numpy(stacked_imgs.transpose((0, 3, 1, 2)))

        return imgs_tensor.float().cuda()

    def _save_imgs_worker(self, imgs_bgr: Tensor, name: str) -> Tensor:
        imgs_rgb = self._bgr_to_rgb(imgs_bgr)
        dir = join(self.config.log_dir, "image", name)
        os.makedirs(dir, exist_ok=True)
        for i, img in enumerate(imgs_rgb, start=1):
            save_image(img, join(dir, f"{i}.png"))

        return imgs_rgb

    def _save_imgs(
        self,
        identity: int,
        origin_imgs: Tensor,
        rebuild_imgs: Tensor,
        swap_ids: list[int],
        swap_imgs: list[Tensor],
    ) -> None:
        origin_imgs = self._save_imgs_worker(origin_imgs, str(identity))
        rebuild_imgs = self._save_imgs_worker(rebuild_imgs, f"{identity}_rebuild")

        results = [origin_imgs[:16], rebuild_imgs[:16]]
        for id, imgs in zip(swap_ids, swap_imgs):
            saved_imgs = self._save_imgs_worker(imgs, f"{identity}_{id}")
            results.append(saved_imgs[:16])

        summary = torch.cat(results, dim=0)
        save_image(
            summary,
            join(self.config.log_dir, "image", f"summary_{identity}.png"),
            nrow=len(origin_imgs[:16]),
        )
