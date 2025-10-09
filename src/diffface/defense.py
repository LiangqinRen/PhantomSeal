from src.diffface.base import Base
from src.dataset import FFHQDataset
from src.utils import save_tensor_imgs

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
        for idx, (img_A, img_B) in enumerate(dataloader, start=1):
            img_A, img_B = img_A.cuda(), img_B.cuda()
            result = self.swap_face(img_A, img_B)
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "img_A",
                    "img_B",
                    "result",
                ],
                [img_A, img_B, result],
                only_save_summary=False,
            )
            break
