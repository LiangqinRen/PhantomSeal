from simswap.base import Base
from dataset import MetricDataset_512
from utils import Timer, get_customized_logger

import inspect
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.utils import save_image


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    logger = get_customized_logger()

    timer = Timer(inspect.currentframe().f_code.co_name, logger)
    base = Base(logger, config)

    dataset = MetricDataset_512(config.data.dataset_dir, 30)
    dataloader = DataLoader(
        dataset, batch_size=config.third_party.batch_size, shuffle=True
    )
    for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
        imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()

        imgs_A_identity = base._get_imgs_identity(imgs_A)
        imgs_A_src_swap = base.target(None, imgs_B, imgs_A_identity, None, True)
        save_image(imgs_A_src_swap, "trash.png")
        break


if __name__ == "__main__":
    main()
