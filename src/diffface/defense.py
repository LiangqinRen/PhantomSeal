from diffface.base import Base
from dataset import FFHQDataset

from pathlib import Path
from torch.utils.data import DataLoader


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def metric(self) -> None:
        dataset = FFHQDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        print(len(dataloader))
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            print(imgs_A.shape, imgs_B.shape)
            break
