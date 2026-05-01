import src.metric as metric
from src.common_utils import save_tensor_imgs
from src.dataset import FFHQMetric
from src.diffswap.base import Base
from src.evaluate import ScoreCalculator

import textwrap
import torch
import torch.nn.functional as F
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

        self.score_calculator = ScoreCalculator(logger, config)

    @torch.no_grad()
    def swap(self) -> None:
        config = self.config.third_party
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = FFHQMetric(
            Path(config.dataset.metric_dir), config.dataset.metric_pairs, transform
        )
        dataloader = DataLoader(dataset, batch_size=config.dataset.batch_size)
        metrics = self._get_swap_success_metric_data_template(self.effectiveness)
        total_count = 0
        source_eval_count = 0
        target_eval_count = 0
        source_skip_count = 0
        target_skip_count = 0

        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)
            source_swap = self.swap_face(imgs_A, imgs_B)
            source_valid_indices = getattr(self, "last_valid_indices", [])
            source_failed_indices = getattr(self, "last_failed_indices", [])
            target_swap = self.swap_face(imgs_B, imgs_A)
            target_valid_indices = getattr(self, "last_valid_indices", [])
            target_failed_indices = getattr(self, "last_failed_indices", [])

            source_eval_count += len(source_valid_indices)
            target_eval_count += len(target_valid_indices)
            source_skip_count += len(source_failed_indices)
            target_skip_count += len(target_failed_indices)

            source_effectiveness = (
                self.effectiveness.calculate_effectiveness(
                    imgs_A[source_valid_indices],
                    None,
                    source_swap[source_valid_indices],
                    None,
                    None,
                )
                if source_valid_indices
                else self._get_empty_effectiveness()
            )
            target_effectiveness = (
                self.effectiveness.calculate_effectiveness(
                    imgs_B[target_valid_indices],
                    None,
                    target_swap[target_valid_indices],
                    None,
                    None,
                )
                if target_valid_indices
                else self._get_empty_effectiveness()
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
                    F.interpolate(
                        imgs_A,
                        size=source_swap.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    F.interpolate(
                        imgs_B,
                        size=target_swap.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
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
            evaluated/skipped source: {source_eval_count}/{source_skip_count}
            evaluated/skipped target: {target_eval_count}/{target_skip_count}
            source effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'source_effectiveness')}
            target effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'target_effectiveness')}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))
            self._free_gpu()

    def _get_empty_effectiveness(self) -> dict:
        data = {}
        for function in self.effectiveness.candi_funcs.keys():
            data[function] = {"swap": (0, 0)}
        return data

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

    def metric(self) -> None:
        pass
