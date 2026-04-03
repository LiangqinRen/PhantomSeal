from src.common_utils import cd, use_project, save_tensor_imgs
from src.dataset import AFDataset

import math
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn


class Base:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda"
        root_dir = Path(config.third_party.project_root)
        with use_project([root_dir]), cd(root_dir):
            from models import StegaStampEncoder, StegaStampDecoder

            self.encoder = StegaStampEncoder(
                config.third_party.origin.image_resolution,
                config.third_party.origin.image_channel,
                config.third_party.origin.fingerprint_length,
                return_residual=False,
            )
            self.decoder = StegaStampDecoder(
                config.third_party.origin.image_resolution,
                config.third_party.origin.image_channel,
                config.third_party.origin.fingerprint_length,
            )

            if config.third_party.function != "train":
                self.encoder.load_state_dict(
                    torch.load(config.third_party.defense.encoder_path)
                )
                self.decoder.load_state_dict(
                    torch.load(config.third_party.defense.decoder_path)
                )

                self.encoder = self.encoder.to(self.device)
                self.decoder = self.decoder.to(self.device)

    def train(self) -> None:
        origin_config = self.config.third_party.origin

        dataset = AFDataset(self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=origin_config.batch_size,
            shuffle=True,
            num_workers=16,
        )

        encoder = self.encoder.to(self.device)
        decoder = self.decoder.to(self.device)

        decoder_encoder_optim = Adam(
            params=list(decoder.parameters()) + list(encoder.parameters()),
            lr=origin_config.lr,
        )

        checkpoints = Path(self.config.log_dir) / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        steps_since_l2_loss_activated = -1
        for epoch in range(1, origin_config.num_epochs + 1):
            encoder.train()
            decoder.train()
            for batch_idx, images in enumerate(dataloader, start=1):
                batch_size = min(origin_config.batch_size, images.size(0))
                fingerprints = self._generate_random_fingerprints(
                    origin_config.fingerprint_length, batch_size
                )

                l2_loss_weight = min(
                    max(
                        0,
                        origin_config.l2_loss_weight
                        * (steps_since_l2_loss_activated - origin_config.l2_loss_await)
                        / origin_config.l2_loss_ramp,
                    ),
                    origin_config.l2_loss_weight,
                )
                BCE_loss_weight = origin_config.BCE_loss_weight

                clean_images = images.to(self.device)
                fingerprints = fingerprints.to(self.device)
                fingerprinted_images = encoder(fingerprints, clean_images)

                decoder_output = decoder(fingerprinted_images)
                criterion = nn.MSELoss()
                l2_loss = criterion(fingerprinted_images, clean_images)
                criterion = nn.BCEWithLogitsLoss()
                BCE_loss = criterion(decoder_output.view(-1), fingerprints.view(-1))

                loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss

                encoder.zero_grad()
                decoder.zero_grad()

                loss.backward()
                decoder_encoder_optim.step()

                fingerprints_predicted = (decoder_output > 0).float()
                bitwise_accuracy = 1.0 - torch.mean(
                    torch.abs(fingerprints - fingerprints_predicted)
                )
                if steps_since_l2_loss_activated == -1:
                    if bitwise_accuracy.item() > 0.9:
                        steps_since_l2_loss_activated = 0
                else:
                    steps_since_l2_loss_activated += 1

                if batch_idx % origin_config.log_interval == 0:
                    self.logger.info(
                        f"[Epoch {epoch:3}/{origin_config.num_epochs:3}][Batch {batch_idx:5}/{len(dataloader):5}] "
                        f"loss: {loss.item():.5f}, "
                        f"bit accuracy: {bitwise_accuracy.item()*100:.3f}"
                    )

                    save_tensor_imgs(
                        self.image_dir,
                        f"{epoch}_{batch_idx}",
                        ["clean\nimages", "finger\nprinted\nimages"],
                        [clean_images, fingerprinted_images],
                        only_save_summary=self.config.third_party.defense.only_save_summary,
                    )

                if loss.item() < best_loss and math.isclose(
                    l2_loss_weight, origin_config.l2_loss_weight
                ):
                    best_loss = loss.item()
                    torch.save(
                        decoder_encoder_optim.state_dict(), checkpoints / "optim.pth"
                    )
                    torch.save(
                        encoder.state_dict(),
                        checkpoints / f"{epoch}_{batch_idx}_encoder.pth",
                    )
                    torch.save(
                        decoder.state_dict(),
                        checkpoints / f"{epoch}_{batch_idx}_decoder.pth",
                    )

    def _generate_random_fingerprints(self, fingerprint_length, batch_size):
        z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(
            0, 2
        )
        return z
