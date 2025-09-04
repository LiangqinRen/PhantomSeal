from test_Dual_Mark import seed_torch, get_path
from network.Dual_Mark import Network
from utils.mask_img_loader import maskImgDataset
from simswap.base import Base as SimSwapBase
from simswap.robustness import Robustness

import torch
import os
import numpy as np
import random
from torch import Tensor
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path


class Base(SimSwapBase):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.robustness = Robustness(logger, config)
        self.test_imgs_path = self._get_all_test_imgs_path()

    def operate_robustness(self, img1: Tensor) -> dict:
        logo = self.robustness.load_logo()

        noise_img1 = self.robustness.gauss_noise(img1, 0, 0.1)
        compress_img1 = self.robustness.webp_compress(img1, 80)
        crop_img1 = self.robustness.crop(img1, 20)
        logo_img1 = self.robustness.logo(img1, logo)
        inc_bright_img1 = self.robustness.brightness(img1, 1.25)
        dec_bright_img1 = self.robustness.brightness(img1, 0.75)

        img2_path = random.sample(self.test_imgs_path, 1)
        img2 = self.robustness._load_imgs(img2_path)

        img1_noise_identity = self._get_imgs_identity(noise_img1)
        img1_noise_src_swap = self.target(None, img2, img1_noise_identity, None, True)

        img1_compress_identity = self._get_imgs_identity(compress_img1)
        img1_compress_src_swap = self.target(
            None, img2, img1_compress_identity, None, True
        )

        img1_crop_identity = self._get_imgs_identity(crop_img1)
        img1_crop_src_swap = self.target(None, img2, img1_crop_identity, None, True)

        img1_logo_identity = self._get_imgs_identity(logo_img1)
        img1_logo_src_swap = self.target(None, img2, img1_logo_identity, None, True)

        img1_inc_bright_identity = self._get_imgs_identity(inc_bright_img1)
        img1_inc_bright_src_swap = self.target(
            None, img2, img1_inc_bright_identity, None, True
        )

        img1_dec_bright_identity = self._get_imgs_identity(dec_bright_img1)
        img1_dec_bright_src_swap = self.target(
            None, img2, img1_dec_bright_identity, None, True
        )

        img2_identity = self._get_imgs_identity(img2)
        img1_noise_tgt_swap = self.target(None, noise_img1, img2_identity, None, True)
        img1_compress_tgt_swap = self.target(
            None, compress_img1, img2_identity, None, True
        )
        img1_crop_tgt_swap = self.target(None, crop_img1, img2_identity, None, True)
        img1_logo_tgt_swap = self.target(None, logo_img1, img2_identity, None, True)
        img1_inc_bright_tgt_swap = self.target(
            None, inc_bright_img1, img2_identity, None, True
        )
        img1_dec_bright_tgt_swap = self.target(
            None, dec_bright_img1, img2_identity, None, True
        )

        img1_identity = self._get_imgs_identity(img1)
        img1_src_swap = self.target(None, img2, img1_identity, None, True)
        img1_tgt_swap = self.target(None, img1, img2_identity, None, True)

        results = {
            "img1_src_swap": img1_src_swap,
            "img1_tgt_swap": img1_tgt_swap,
            "img1_noise_src_swap": img1_noise_src_swap,
            "img1_compress_src_swap": img1_compress_src_swap,
            "img1_crop_src_swap": img1_crop_src_swap,
            "img1_logo_src_swap": img1_logo_src_swap,
            "img1_inc_bright_src_swap": img1_inc_bright_src_swap,
            "img1_dec_bright_src_swap": img1_dec_bright_src_swap,
            "img1_noise_tgt_swap": img1_noise_tgt_swap,
            "img1_compress_tgt_swap": img1_compress_tgt_swap,
            "img1_crop_tgt_swap": img1_crop_tgt_swap,
            "img1_logo_tgt_swap": img1_logo_tgt_swap,
            "img1_inc_bright_tgt_swap": img1_inc_bright_tgt_swap,
            "img1_dec_bright_tgt_swap": img1_dec_bright_tgt_swap,
        }

        return results

    def sepmark(self):
        seed_torch(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = Network(
            self.config.third_party.message_length,
            self.config.third_party.noise_layers.pool_R,
            self.config.third_party.noise_layers.pool_F,
            device,
            self.config.third_party.batch_size,
            self.config.third_party.lr,
            self.config.third_party.beta1,
            self.config.third_party.attention_encoder,
            self.config.third_party.attention_decoder,
            self.config.third_party.weight,
        )

        network.load_model_ed(self.config.third_party.model_path)

        test_dataset = maskImgDataset(
            self.config.third_party.dataset_path, self.config.third_party.image_size
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.third_party.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        test_result: dict[str, tuple[float, float]] = {
            "origin": (0, 0),
            "as_src": (0, 0),
            "as_tgt": (0, 0),
            "noise_as_src": (0, 0),
            "compress_as_src": (0, 0),
            "crop_as_src": (0, 0),
            "logo_as_src": (0, 0),
            "inc_bright_as_src": (0, 0),
            "dec_bright_as_src": (0, 0),
            "noise_as_tgt": (0, 0),
            "compress_as_tgt": (0, 0),
            "crop_as_tgt": (0, 0),
            "logo_as_tgt": (0, 0),
            "inc_bright_as_tgt": (0, 0),
            "dec_bright_as_tgt": (0, 0),
        }

        messages = torch.Tensor(
            np.random.choice(
                [
                    -self.config.third_party.message_range,
                    self.config.third_party.message_range,
                ],
                (1, self.config.third_party.message_length),
            )
        ).to(device)

        for step, (image, mask) in enumerate(test_dataloader, 1):
            image = image.to(device)
            network.encoder_decoder.eval()
            network.discriminator.eval()

            with torch.no_grad():
                images = image.cuda()

                encoded_images = network.encoder_decoder.module.encoder(
                    images, messages
                )
                encoded_images = (
                    images
                    + (encoded_images - images)
                    * self.config.third_party.strength_factor
                )

                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]
                )

                for index in range(encoded_images.shape[0]):
                    single_image = (
                        (
                            (encoded_images[index].clamp(-1, 1).permute(1, 2, 0) + 1)
                            / 2
                            * 255
                        )
                        .add(0.5)
                        .clamp(0, 255)
                        .to("cpu", torch.uint8)
                        .numpy()
                    )
                    im = Image.fromarray(single_image)
                    file = get_path()
                    while os.path.exists(file):
                        file = get_path()
                    im.save(file)
                    read = np.array(Image.open(file), dtype=np.uint8)
                    os.remove(file)

                    encoded_images[index] = (
                        transform(read).unsqueeze(0).to(image.device)
                    )

                noised_images = encoded_images
                for index in range(noised_images.shape[0]):
                    single_image = (
                        (
                            (noised_images[index].clamp(-1, 1).permute(1, 2, 0) + 1)
                            / 2
                            * 255
                        )
                        .add(0.5)
                        .clamp(0, 255)
                        .to("cpu", torch.uint8)
                        .numpy()
                    )
                    im = Image.fromarray(single_image)
                    file = get_path()
                    while os.path.exists(file):
                        file = get_path()
                    im.save(file)
                    read = np.array(Image.open(file), dtype=np.uint8)
                    os.remove(file)

                    noised_images[index] = transform(read).unsqueeze(0).to(image.device)

                results = self.operate_robustness(noised_images)
                results["original"] = noised_images

                decoded_messages = {}
                for item, img in results.items():
                    messages_C = network.encoder_decoder.module.decoder_C(img)
                    messages_RF = network.encoder_decoder.module.decoder_RF(img)
                    decoded_messages[item] = (messages_C, messages_RF)

            for item, message in decoded_messages.items():
                error_rate_C = network.decoded_message_error_rate_batch(
                    messages, message[0]
                )
                error_rate_RF = network.decoded_message_error_rate_batch(
                    messages, message[1]
                )
                test_result[item] = (
                    test_result[item][0] + float(error_rate_C),
                    test_result[item][1] + float(error_rate_RF),
                )

            self.logger.info(f"[{step:4}]{test_result}")

    def _get_all_test_imgs_path(self) -> list[Path]:
        root = Path(self.config.third_party.test_dir)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
        all_images = [
            p for p in root.rglob("*") if p.suffix.lower() in image_extensions
        ]

        return all_images
