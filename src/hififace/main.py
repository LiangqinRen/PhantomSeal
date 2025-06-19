import utils
from hififace.base import Base
from dataset import MetricDataset

import inspect
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.utils import save_image


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig):
    logger = utils.get_customized_logger()

    utils.check_cuda_availability(logger)
    utils.fix_random_seed(logger, config.random_seed)
    timer = utils.Timer(inspect.currentframe().f_code.co_name, logger)

    base = Base(logger, config)
    dataset = MetricDataset(config)
    dataloader = DataLoader(
        dataset, batch_size=config.third_party.defense.batch_size, shuffle=True
    )
    for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
        imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
        imgs_A_src_swap = base.net(imgs_A, imgs_B)
        save_image(imgs_A_src_swap, "trash.png")
        quit()


if __name__ == "__main__":
    main()
