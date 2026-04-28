import src.metric as metric
from src.e4s.base import Base
from src.dataset import FFHQMetric
from src.evaluate import ScoreCalculator
from src.common_utils import check_tensor_info, save_tensor_imgs

import torch
import textwrap
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms


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
        metrics = self._get_swap_success_metric_data_template(self.effectiveness)
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)
            source_swap = self.swap_face(imgs_A * 2 - 1, imgs_B * 2 - 1)
            target_swap = self.swap_face(imgs_B * 2 - 1, imgs_A * 2 - 1)

            source_effectiveness = self.effectiveness.calculate_effectiveness(
                imgs_A,
                None,
                (source_swap + 1) / 2,
                None,
                None,
            )
            target_effectiveness = self.effectiveness.calculate_effectiveness(
                imgs_B,
                None,
                (target_swap + 1) / 2,
                None,
                None,
            )
            self._merge_swap_success_metric(
                metrics, source_effectiveness, target_effectiveness
            )

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "source_swap",
                    "target_swap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    source_swap,
                    target_swap,
                ],
                only_save_summary=True,
            )

            iter_log_str = textwrap.dedent(
                f"""
            effectiveness ({', '.join(self.effectiveness.candi_funcs.keys())})
            source effectiveness: {metric.generate_iter_effectiveness_log(source_effectiveness)}
            target effectiveness: {metric.generate_iter_effectiveness_log(target_effectiveness)}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            source effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'source_effectiveness')}
            target effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'target_effectiveness')}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

    @staticmethod
    def _get_swap_success_metric_data_template(effectiveness) -> dict:
        data = {
            "source_effectiveness": {},
            "target_effectiveness": {},
        }

        for function in effectiveness.candi_funcs.keys():
            data["source_effectiveness"][function] = {"swap": (0, 0)}
            data["target_effectiveness"][function] = {"swap": (0, 0)}

        return data

    @staticmethod
    def _merge_swap_success_metric(
        metrics: dict, source_effectiveness: dict, target_effectiveness: dict
    ) -> None:
        for effec in source_effectiveness.keys():
            source_prev = metrics["source_effectiveness"][effec]["swap"]
            source_cur = source_effectiveness[effec]["swap"]
            metrics["source_effectiveness"][effec]["swap"] = (
                source_prev[0] + source_cur[0],
                source_prev[1] + source_cur[1],
            )

            target_prev = metrics["target_effectiveness"][effec]["swap"]
            target_cur = target_effectiveness[effec]["swap"]
            metrics["target_effectiveness"][effec]["swap"] = (
                target_prev[0] + target_cur[0],
                target_prev[1] + target_cur[1],
            )

    def sample(self) -> None:
        pass

    def metric(
        self,
    ) -> None:
        pass
