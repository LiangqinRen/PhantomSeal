from src.common_utils import save_tensor_imgs
from src.dataset import FFHQMetric
from src.diffswap.base import Base

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        notes_path = Path(self.config.notes_path)
        notes_path.touch(exist_ok=True)

    @torch.no_grad()
    def swap(self) -> None:
        config = self.config.third_party
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (config.dataset.image_size, config.dataset.image_size)
                ),
                transforms.ToTensor(),
            ]
        )
        dataset = FFHQMetric(
            Path(config.dataset.metric_dir), config.dataset.metric_pairs, transform
        )
        dataloader = DataLoader(dataset, batch_size=config.dataset.batch_size)
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            source_swap = self.swap_face(imgs_A, imgs_B)
            target_swap = self.swap_face(imgs_B, imgs_A)

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "source\nswap",
                    "target\nswap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    source_swap,
                    target_swap,
                ],
                only_save_summary=True,
            )

    def sample(self) -> None:
        pass

    def metric(self) -> None:
        pass
