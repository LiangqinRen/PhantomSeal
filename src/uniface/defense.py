from src import metric
from src.uniface.base import Base
from src.dataset import VGGFace2Dataset
from src.evaluate import ScoreCalculator
from src.common_utils import check_tensor_info, save_tensor_imgs

import textwrap
import torch
import torch.nn.functional as F
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
        dataset = VGGFace2Dataset(config)
        dataloader = DataLoader(dataset, batch_size=config.dataset.batch_size)
        metrics = self._get_swap_success_metric_data_template(self.effectiveness)
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)
            source_swap = self.swap_face(imgs_A, imgs_B)
            target_swap = self.swap_face(imgs_B, imgs_A)

            source_effectiveness = self.effectiveness.calculate_effectiveness(
                self._denormalize(imgs_A),
                None,
                source_swap,
                None,
                None,
            )
            target_effectiveness = self.effectiveness.calculate_effectiveness(
                self._denormalize(imgs_B),
                None,
                target_swap,
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

    @staticmethod
    def _denormalize(imgs: torch.Tensor) -> torch.Tensor:
        return (imgs + 1) / 2

    def sample(self) -> None:
        pass

    def metric(
        self,
    ) -> None:
        pass
