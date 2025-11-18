from simswap.base import Base
from dataset import MetricDataset, FFHQDataset

import torch
from torch.utils.data import DataLoader


class Misc(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

    def compute_misclassification_rate(self) -> None:
        torch.set_grad_enabled(False)
        datasets = {"vggface2": MetricDataset, "FFHQ": FFHQDataset}

        dataset = datasets[self.config.third_party.misc.dataset](self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size
        )
        total_count = 0
        data = {"facenet": (0, 0), "face++": (0, 0)}
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)

            results = self.effectiveness.calculate_effectiveness(
                imgs_A, None, imgs_B, None, None
            )
            data = {
                "facenet": (
                    data["facenet"][0] + int(results["facenet"]["swap"][0]),
                    data["facenet"][1] + int(results["facenet"]["swap"][1]),
                ),
                "face++": (
                    data["face++"][0] + int(results["face++"]["swap"][0]),
                    data["face++"][1] + int(results["face++"]["swap"][1]),
                ),
            }
            self.logger.info(f"{idx:3d}| {total_count:4d}, {data}")
