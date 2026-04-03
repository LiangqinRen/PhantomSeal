from src import metric
from src.e4s.base import Base
from src.dataset import FFHQDataset
from src.evaluate import ScoreCalculator
from src.utils import check_tensor_info, save_tensor_imgs

import torch
import textwrap
import torch.nn.functional as F
from torch import Tensor, tensor
from torch.utils.data import DataLoader
from pathlib import Path


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        notes_path = Path(self.config.notes_path)
        notes_path.touch(exist_ok=True)

        self.score_calculator = ScoreCalculator(logger, config)

    @torch.no_grad()
    def swap(self) -> None:
        config = self.config.third_party
        dataset = FFHQDataset(self.config)
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

    def metric(
        self,
    ) -> None:
        pass
