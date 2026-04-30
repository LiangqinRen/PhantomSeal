from src import metric
from src.simswap.base import Base
from src.dataset import SampleDataset, MetricDataset, AdaptiveMetricDataset
from src.simswap.robustness import Robustness
from src.evaluate import ScoreCalculator
from src.common_utils import save_tensor_imgs


import textwrap
import torch
from torch import tensor, Tensor
from torch.utils.data import DataLoader
from pathlib import Path
from typing import cast


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        notes_path = Path(self.config.notes_path)
        notes_path.touch(exist_ok=True)

        self.robustness = Robustness(logger, config)
        self.score_calculator = ScoreCalculator(logger, config)

        if self.config.third_party.robustness.ai_beauty:
            logger.info(
                f"AI Beauty enabled with {self.config.third_party.robustness.ai_beauty_tool}"
            )

    @torch.no_grad()
    def swap(self) -> None:
        dataset = MetricDataset(self.config)
        swap_batch_size = self.config.third_party.dataset.swap_batch_size
        dataloader = DataLoader(dataset, batch_size=swap_batch_size, shuffle=False)
        metrics = self._get_swap_success_metric_data_template(self.effectiveness)
        total_count = 0

        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)
            source_swap = self.swap_face(imgs_A, imgs_B)
            target_swap = self.swap_face(imgs_B, imgs_A)

            source_effectiveness = self.effectiveness.calculate_effectiveness(
                imgs_A, None, source_swap, None, None
            )
            target_effectiveness = self.effectiveness.calculate_effectiveness(
                imgs_B, None, target_swap, None, None
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

            del imgs_A, imgs_B, source_swap, target_swap
            self._free_gpu()

    def sample(self) -> None:
        dataset = SampleDataset(self.config.third_party.dataset.sample_dir)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            pert_imgs = self._perturb_imgs(imgs_A, cloak_imgs)

            (
                source_swap,
                pert_source_swap,
                target_swap,
                pert_target_swap,
            ) = self._get_full_swap_results(imgs_A, imgs_B, pert_imgs)

            (
                utility,
                source_utility,
                target_utility,
                source_effectiveness,
                target_effectiveness,
            ) = metric.get_defense_metric(
                self.utility,
                self.effectiveness,
                imgs_A,
                imgs_B,
                pert_imgs,
                cloak_imgs,
                source_swap,
                pert_source_swap,
                target_swap,
                pert_target_swap,
            )

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "perturb_imgs",
                    "cloak_imgs",
                    "source_swap",
                    "perturb_source_swap",
                    "target_swap",
                    "perturb_target_swap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    pert_imgs,
                    cloak_imgs,
                    source_swap,
                    pert_source_swap,
                    target_swap,
                    pert_target_swap,
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            scores = self.score_calculator.calculate_score(
                source_effectiveness, target_effectiveness, None
            )

            iter_log_str = textwrap.dedent(
                f"""
            utility (mse, psnr, ssim, lpips), effectiveness ({', '.join(self.effectiveness.candi_funcs.keys())}), identity ({', '.join(next(iter(source_effectiveness.values())).keys())}), context ({', '.join(next(iter(target_effectiveness.values())).keys())})
            utility: {metric.generate_iter_utility_log(utility)}
            source utility: {metric.generate_iter_utility_log(source_utility)}
            target utility: {metric.generate_iter_utility_log(target_utility)}
            source effectiveness: {metric.generate_iter_effectiveness_log(source_effectiveness)}
            target effectiveness: {metric.generate_iter_effectiveness_log(target_effectiveness)}
            scores: {metric.generate_iter_score_log(scores)}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))

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

    def metric(
        self,
    ) -> None:
        metrics = metric.get_metric_data_template(self.effectiveness)

        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            if self.config.third_party.robustness.ai_beauty:
                ai_beauty_tool = self.config.third_party.robustness.ai_beauty_tool
                if ai_beauty_tool == "ai_lab_tools":
                    imgs_A, count = self.aiediting.face_beauty_via_ailabtools(imgs_A)
                elif ai_beauty_tool == "tencent_cloud":
                    imgs_A, count = self.aiediting.face_beauty_via_tencentcloud(imgs_A)
                else:
                    self.logger.critical(f"Unknown ai_beauty_tool: {ai_beauty_tool}")
                    raise ValueError(f"Unknown ai_beauty_tool: {ai_beauty_tool}")
                total_count += count
            elif not self.config.third_party.defense.failure_defense_tracing:
                total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            pert_imgs = self._perturb_imgs(imgs_A, cloak_imgs)

            torch.set_grad_enabled(False)
            (
                source_swap,
                pert_source_swap,
                target_swap,
                pert_target_swap,
            ) = self._get_full_swap_results(imgs_A, imgs_B, pert_imgs)

            if self.config.third_party.defense.failure_defense_tracing:
                success_swap_indices = self.effectiveness.get_success_swap_indices(
                    imgs_A, pert_source_swap
                )
                if len(success_swap_indices) == 0:
                    continue
                else:
                    total_count += len(success_swap_indices)
                    idx_cpu = success_swap_indices.to("cpu")

                    imgs_A_cpu = imgs_A.detach().to("cpu")
                    del imgs_A
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    imgs_A = self._pick_on_cpu_then_to_gpu(
                        imgs_A_cpu, idx_cpu, device="cuda"
                    )
                    imgs_B_cpu = imgs_B.detach().to("cpu")
                    del imgs_B
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    imgs_B = self._pick_on_cpu_then_to_gpu(
                        imgs_B_cpu, idx_cpu, device="cuda"
                    )
                    x_imgs_cpu = pert_imgs.detach().to("cpu")
                    del pert_imgs
                    self._free_gpu()
                    pert_imgs = self._pick_on_cpu_then_to_gpu(
                        x_imgs_cpu, idx_cpu, device="cuda"
                    )
                    cloak_imgs_cpu = cloak_imgs.detach().to("cpu")
                    del cloak_imgs
                    self._free_gpu()
                    cloak_imgs = self._pick_on_cpu_then_to_gpu(
                        cloak_imgs_cpu, idx_cpu, device="cuda"
                    )
                    imgs_A_src_swap_cpu = source_swap.detach().to("cpu")
                    del source_swap
                    self._free_gpu()
                    source_swap = self._pick_on_cpu_then_to_gpu(
                        imgs_A_src_swap_cpu, idx_cpu, device="cuda"
                    )
                    pert_imgs_A_src_swap_cpu = pert_source_swap.detach().to("cpu")
                    del pert_source_swap
                    self._free_gpu()
                    pert_source_swap = self._pick_on_cpu_then_to_gpu(
                        pert_imgs_A_src_swap_cpu, idx_cpu, device="cuda"
                    )
                    imgs_A_tgt_swap_cpu = target_swap.detach().to("cpu")
                    del target_swap
                    self._free_gpu()
                    target_swap = self._pick_on_cpu_then_to_gpu(
                        imgs_A_tgt_swap_cpu, idx_cpu, device="cuda"
                    )
                    pert_imgs_A_tgt_swap_cpu = pert_target_swap.detach().to("cpu")
                    del pert_target_swap
                    self._free_gpu()
                    pert_target_swap = self._pick_on_cpu_then_to_gpu(
                        pert_imgs_A_tgt_swap_cpu, idx_cpu, device="cuda"
                    )

                    del (
                        idx_cpu,
                        imgs_A_cpu,
                        imgs_B_cpu,
                        x_imgs_cpu,
                        cloak_imgs_cpu,
                        imgs_A_src_swap_cpu,
                        imgs_A_tgt_swap_cpu,
                        pert_imgs_A_src_swap_cpu,
                        pert_imgs_A_tgt_swap_cpu,
                    )
                    self._free_gpu()

            (
                utility,
                source_utility,
                target_utility,
                source_effectiveness,
                target_effectiveness,
            ) = metric.get_defense_metric(
                self.utility,
                self.effectiveness,
                imgs_A,
                imgs_B,
                pert_imgs,
                cloak_imgs,
                source_swap,
                pert_source_swap,
                target_swap,
                pert_target_swap,
            )

            metric.merge_metric(
                self.effectiveness,
                metrics,
                utility,
                source_utility,
                target_utility,
                source_effectiveness,
                target_effectiveness,
            )

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "perturb_imgs",
                    "cloak_imgs",
                    "source_swap",
                    "perturb_source_swap",
                    "target_swap",
                    "perturb_target_swap",
                ],
                [
                    imgs_A,
                    imgs_B,
                    pert_imgs,
                    cloak_imgs,
                    source_swap,
                    pert_source_swap,
                    target_swap,
                    pert_target_swap,
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            del imgs_A, imgs_B, pert_imgs, cloak_imgs
            del (
                source_swap,
                pert_source_swap,
                target_swap,
                pert_target_swap,
            )
            self._free_gpu()

            scores = self.score_calculator.calculate_score(
                source_effectiveness, target_effectiveness, metrics
            )

            iter_log_str = textwrap.dedent(
                f"""
            utility (mse, psnr, ssim, lpips), effectiveness ({', '.join(self.effectiveness.candi_funcs.keys())}), identity ({', '.join(next(iter(source_effectiveness.values())).keys())}), context ({', '.join(next(iter(target_effectiveness.values())).keys())})
            utility: {metric.generate_iter_utility_log(utility)}
            source utility: {metric.generate_iter_utility_log(source_utility)}
            target utility: {metric.generate_iter_utility_log(target_utility)}
            source effectiveness: {metric.generate_iter_effectiveness_log(source_effectiveness)}
            target effectiveness: {metric.generate_iter_effectiveness_log(target_effectiveness)}
            scores: {metric.generate_iter_score_log(scores)}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            utility: {metric.generate_summary_utility_log(metrics, 'utility', idx)}
            source utility: {metric.generate_summary_utility_log(metrics, 'pert_source_utility', idx)}
            target utility: {metric.generate_summary_utility_log(metrics, 'pert_target_utility', idx)}
            source effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'pert_source_effectiveness')}
            target effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'pert_target_effectiveness')}
            scores: {metric.generate_summary_score_log(scores)}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

    def protection_robustness_sample(self) -> None:
        dataset = SampleDataset(self.config.third_party.dataset.sample_dir)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        logo = self.robustness.load_logo()
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            pert_imgs = self._perturb_imgs(imgs_A, cloak_imgs)
            pert_imgs = self.robustness.webp_compress(pert_imgs, 80)
            pert_imgs = self._perturb_imgs(pert_imgs, cloak_imgs)
            pert_imgs = self.robustness.webp_compress(pert_imgs, 80)
            pert_imgs = self._perturb_imgs(pert_imgs, cloak_imgs)

            utility = self.utility.calculate_utility(imgs_A, pert_imgs)

            (
                noise_source_effectivenesses,
                noise_target_effectivenesses,
            ) = self.robustness.get_gauss_noise_metrics(
                idx, imgs_A, imgs_B, pert_imgs, cloak_imgs, self.image_dir
            )

            (
                compress_source_effectivenesses,
                compress_target_effectivenesses,
            ) = self.robustness.get_compress_metrics(
                idx, imgs_A, imgs_B, pert_imgs, cloak_imgs, self.image_dir
            )

            (
                crop_source_effectivenesses,
                crop_target_effectivenesses,
            ) = self.robustness.get_crop_metrics(
                idx, imgs_A, imgs_B, pert_imgs, cloak_imgs, self.image_dir
            )

            (
                logo_source_effectivenesses,
                logo_target_effectivenesses,
            ) = self.robustness.get_logo_metrics(
                idx, imgs_A, imgs_B, pert_imgs, logo, cloak_imgs, self.image_dir
            )

            (
                brighten_source_effectivenesses,
                brighten_target_effectivenesses,
            ) = self.robustness.get_brightness_metrics(
                idx, imgs_A, imgs_B, pert_imgs, 1.25, cloak_imgs, self.image_dir
            )

            (
                darken_source_effectivenesses,
                darken_target_effectivenesses,
            ) = self.robustness.get_brightness_metrics(
                idx, imgs_A, imgs_B, pert_imgs, 0.75, cloak_imgs, self.image_dir
            )

            self._free_gpu()

            iter_log_str = textwrap.dedent(
                f"""
                utility (mse, psnr, ssim, lpips), effectiveness ({', '.join(self.effectiveness.candi_funcs.keys())}), identity ({', '.join(next(iter(noise_source_effectivenesses.values())).keys())}), context ({', '.join(next(iter(noise_source_effectivenesses.values())).keys())})
                utility: {metric.generate_iter_utility_log(utility)}
                noise: {metric.generate_iter_robustness_log(noise_source_effectivenesses,noise_target_effectivenesses)}
                compress: {metric.generate_iter_robustness_log(compress_source_effectivenesses,compress_target_effectivenesses)}
                crop: {metric.generate_iter_robustness_log(crop_source_effectivenesses,crop_target_effectivenesses)}
                logo: {metric.generate_iter_robustness_log(logo_source_effectivenesses,logo_target_effectivenesses)}
                brighten: {metric.generate_iter_robustness_log(brighten_source_effectivenesses,brighten_target_effectivenesses)}
                darken: {metric.generate_iter_robustness_log(darken_source_effectivenesses,darken_target_effectivenesses)}
                """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))

    def protection_robustness_metric(self) -> None:
        metrics = metric.get_robustness_metric_data_template(
            self.config, self.effectiveness
        )

        logo = self.robustness.load_logo()
        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            pert_imgs = self._perturb_imgs(imgs_A, cloak_imgs)
            pert_imgs = self.robustness.webp_compress(pert_imgs, 80)
            pert_imgs = self._perturb_imgs(pert_imgs, cloak_imgs)
            pert_imgs = self.robustness.webp_compress(pert_imgs, 80)
            pert_imgs = self._perturb_imgs(pert_imgs, cloak_imgs)
            torch.set_grad_enabled(False)

            utility = cast(dict, self.utility.calculate_utility(imgs_A, pert_imgs))
            metric.merge_single_dict(metrics["utility"], utility)

            (
                noise_source_effectivenesses,
                noise_target_effectivenesses,
            ) = self.robustness.get_gauss_noise_metrics(
                idx, imgs_A, imgs_B, pert_imgs, cloak_imgs, self.image_dir
            )
            metric.merge_single_robustness_metric(
                metrics,
                noise_source_effectivenesses,
                noise_target_effectivenesses,
                "noise",
            )

            (
                compress_source_effectivenesses,
                compress_target_effectivenesses,
            ) = self.robustness.get_compress_metrics(
                idx, imgs_A, imgs_B, pert_imgs, cloak_imgs, self.image_dir
            )
            metric.merge_single_robustness_metric(
                metrics,
                compress_source_effectivenesses,
                compress_target_effectivenesses,
                "compress",
            )

            (
                crop_source_effectivenesses,
                crop_target_effectivenesses,
            ) = self.robustness.get_crop_metrics(
                idx, imgs_A, imgs_B, pert_imgs, cloak_imgs, self.image_dir
            )
            metric.merge_single_robustness_metric(
                metrics,
                crop_source_effectivenesses,
                crop_target_effectivenesses,
                "crop",
            )

            (
                logo_source_effectivenesses,
                logo_target_effectivenesses,
            ) = self.robustness.get_logo_metrics(
                idx, imgs_A, imgs_B, pert_imgs, logo, cloak_imgs, self.image_dir
            )
            metric.merge_single_robustness_metric(
                metrics,
                logo_source_effectivenesses,
                logo_target_effectivenesses,
                "logo",
            )

            (
                brighten_source_effectivenesses,
                brighten_target_effectivenesses,
            ) = self.robustness.get_brightness_metrics(
                idx, imgs_A, imgs_B, pert_imgs, 1.25, cloak_imgs, self.image_dir
            )
            metric.merge_single_robustness_metric(
                metrics,
                brighten_source_effectivenesses,
                brighten_target_effectivenesses,
                "brighten",
            )

            (
                darken_source_effectivenesses,
                darken_target_effectivenesses,
            ) = self.robustness.get_brightness_metrics(
                idx, imgs_A, imgs_B, pert_imgs, 0.75, cloak_imgs, self.image_dir
            )
            metric.merge_single_robustness_metric(
                metrics,
                darken_source_effectivenesses,
                darken_target_effectivenesses,
                "darken",
            )

            self._free_gpu()

            iter_log_str = textwrap.dedent(
                f"""
            utility (mse, psnr, ssim, lpips), effectiveness ({', '.join(self.effectiveness.candi_funcs.keys())}), identity ({', '.join(next(iter(noise_source_effectivenesses.values())).keys())}), context ({', '.join(next(iter(noise_source_effectivenesses.values())).keys())})
            utility: {metric.generate_iter_utility_log(utility)}
            noise: {metric.generate_iter_robustness_log(noise_source_effectivenesses,noise_target_effectivenesses)}
            compress: {metric.generate_iter_robustness_log(compress_source_effectivenesses,compress_target_effectivenesses)}
            crop: {metric.generate_iter_robustness_log(crop_source_effectivenesses,crop_target_effectivenesses)}
            logo: {metric.generate_iter_robustness_log(logo_source_effectivenesses,logo_target_effectivenesses)}
            brighten: {metric.generate_iter_robustness_log(brighten_source_effectivenesses,brighten_target_effectivenesses)}
            darken: {metric.generate_iter_robustness_log(darken_source_effectivenesses,darken_target_effectivenesses)}
            """
            )

            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            utility: {metric.generate_summary_robustness_utility_log(metrics['utility'], idx)}
            noise: {metric.generate_summary_robustness_log(metrics['noise'])}
            compress: {metric.generate_summary_robustness_log(metrics['compress'])}
            crop: {metric.generate_summary_robustness_log(metrics['crop'])}
            logo: {metric.generate_summary_robustness_log(metrics['logo'])}
            brighten: {metric.generate_summary_robustness_log(metrics['brighten'])}
            darken: {metric.generate_summary_robustness_log(metrics['darken'])}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

    def forensics_robustness_sample(self):
        dataset = SampleDataset(self.config.third_party.dataset.sample_dir)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        logo = self.robustness.load_logo()
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            pert_imgs = self._perturb_imgs(imgs_A, cloak_imgs)
            pert_imgs = self.robustness.webp_compress(pert_imgs, 80)
            pert_imgs = self._perturb_imgs(pert_imgs, cloak_imgs)
            pert_imgs = self.robustness.webp_compress(pert_imgs, 80)
            pert_imgs = self._perturb_imgs(pert_imgs, cloak_imgs)

            imgs_A_identity = self._get_imgs_identity(imgs_A)
            source_swap = self.target(None, imgs_B, imgs_A_identity, None, True)
            pert_imgs_A_identity = self._get_imgs_identity(pert_imgs)
            pert_source_swap = self.target(
                None, imgs_B, pert_imgs_A_identity, None, True
            )

            results = {"clean": pert_source_swap}
            results["noise"] = self.robustness.gauss_noise(pert_source_swap)
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "source_swap",
                    "perturb_source_swap",
                    "noise",
                ],
                [
                    imgs_A,
                    imgs_B,
                    source_swap,
                    pert_source_swap,
                    results["noise"],
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            results["compress"] = self.robustness.webp_compress(pert_source_swap)
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "source_swap",
                    "perturb_source_swap",
                    "compress",
                ],
                [
                    imgs_A,
                    imgs_B,
                    source_swap,
                    pert_source_swap,
                    results["compress"],
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            results["crop"] = self.robustness.crop(pert_source_swap)
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "source_swap",
                    "perturb_source_swap",
                    "crop",
                ],
                [
                    imgs_A,
                    imgs_B,
                    source_swap,
                    pert_source_swap,
                    results["crop"],
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            results["logo"] = self.robustness.logo(pert_source_swap, logo)
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "source_swap",
                    "perturb_source_swap",
                    "logo",
                ],
                [
                    imgs_A,
                    imgs_B,
                    source_swap,
                    pert_source_swap,
                    results["logo"],
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            results["brighten"] = self.robustness.brightness(pert_source_swap, 1.25)
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "source_swap",
                    "perturb_source_swap",
                    "brighten",
                ],
                [
                    imgs_A,
                    imgs_B,
                    source_swap,
                    pert_source_swap,
                    results["brighten"],
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            results["darken"] = self.robustness.brightness(pert_source_swap, 0.75)
            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "source_swap",
                    "perturb_source_swap",
                    "darken",
                ],
                [
                    imgs_A,
                    imgs_B,
                    source_swap,
                    pert_source_swap,
                    results["darken"],
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            forensic_metrics = metric.get_robustness_forensics_metric(
                self.effectiveness, cloak_imgs, results
            )

            self._free_gpu()

            iter_log_str = textwrap.dedent(
                f"""
            cloak ({', '.join(self.effectiveness.candi_funcs.keys())})
            origin: {metric.generate_forensics_robustness_log(forensic_metrics['clean'])}
            noise: {metric.generate_forensics_robustness_log(forensic_metrics['noise'])}
            compress: {metric.generate_forensics_robustness_log(forensic_metrics['compress'])}
            crop: {metric.generate_forensics_robustness_log(forensic_metrics['crop'])}
            logo: {metric.generate_forensics_robustness_log(forensic_metrics['logo'])}
            brighten: {metric.generate_forensics_robustness_log(forensic_metrics['brighten'])}
            darken: {metric.generate_forensics_robustness_log(forensic_metrics['darken'])}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))

    def forensics_robustness_metric(self):
        metrics = metric.get_robustness_forensics_metric_data_template(
            self.effectiveness
        )

        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        logo = self.robustness.load_logo()
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            pert_imgs = self._perturb_imgs(imgs_A, cloak_imgs)
            pert_imgs = self.robustness.webp_compress(pert_imgs, 80)
            pert_imgs = self._perturb_imgs(pert_imgs, cloak_imgs)
            pert_imgs = self.robustness.webp_compress(pert_imgs, 80)
            pert_imgs = self._perturb_imgs(pert_imgs, cloak_imgs)

            torch.set_grad_enabled(False)
            pert_imgs_A_identity = self._get_imgs_identity(pert_imgs)
            pert_source_swap = self.target(
                None, imgs_B, pert_imgs_A_identity, None, True
            )

            swap_results = {"clean": pert_source_swap}
            swap_results["noise"] = self.robustness.gauss_noise(pert_source_swap)
            swap_results["compress"] = self.robustness.webp_compress(pert_source_swap)
            swap_results["crop"] = self.robustness.crop(pert_source_swap)
            swap_results["logo"] = self.robustness.logo(pert_source_swap, logo)
            swap_results["brighten"] = self.robustness.brightness(
                pert_source_swap, 1.25
            )
            swap_results["darken"] = self.robustness.brightness(pert_source_swap, 0.75)

            forensic_metrics = metric.get_robustness_forensics_metric(
                self.effectiveness, cloak_imgs, swap_results
            )
            metric.merge_single_dict(metrics, forensic_metrics)

            self._free_gpu()

            iter_log_str = textwrap.dedent(
                f"""
            cloak ({', '.join(self.effectiveness.candi_funcs.keys())})
            origin: {metric.generate_forensics_robustness_log(forensic_metrics['clean'])}
            noise: {metric.generate_forensics_robustness_log(forensic_metrics['noise'])}
            compress: {metric.generate_forensics_robustness_log(forensic_metrics['compress'])}
            crop: {metric.generate_forensics_robustness_log(forensic_metrics['crop'])}
            logo: {metric.generate_forensics_robustness_log(forensic_metrics['logo'])}
            brighten: {metric.generate_forensics_robustness_log(forensic_metrics['brighten'])}
            darken: {metric.generate_forensics_robustness_log(forensic_metrics['darken'])}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            origin: {metric.generate_forensics_robustness_log(metrics['clean'])}
            noise: {metric.generate_forensics_robustness_log(metrics['noise'])}
            compress: {metric.generate_forensics_robustness_log(metrics['compress'])}
            crop: {metric.generate_forensics_robustness_log(metrics['crop'])}
            logo: {metric.generate_forensics_robustness_log(metrics['logo'])}
            brighten: {metric.generate_forensics_robustness_log(metrics['brighten'])}
            darken: {metric.generate_forensics_robustness_log(metrics['darken'])}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

    def image_robustness_metric(self):
        metrics = metric.get_image_robustness_data_template(self.effectiveness)
        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        logo = self.robustness.load_logo()
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)

            noise_imgs_A = self.robustness.gauss_noise(imgs_A)
            self._get_single_robustness_operate_metric(
                imgs_B,
                noise_imgs_A,
                cloak_imgs,
                metrics["noise"],
            )

            compress_imgs_A = self.robustness.webp_compress(imgs_A)
            self._get_single_robustness_operate_metric(
                imgs_B,
                compress_imgs_A,
                cloak_imgs,
                metrics["compress"],
            )

            crop_imgs_A = self.robustness.crop(imgs_A)
            self._get_single_robustness_operate_metric(
                imgs_B,
                crop_imgs_A,
                cloak_imgs,
                metrics["crop"],
            )

            logo_imgs_A = self.robustness.logo(imgs_A, logo)
            self._get_single_robustness_operate_metric(
                imgs_B,
                logo_imgs_A,
                cloak_imgs,
                metrics["logo"],
            )

            brighten_imgs_A = self.robustness.brightness(imgs_A, 1.25)
            self._get_single_robustness_operate_metric(
                imgs_B,
                brighten_imgs_A,
                cloak_imgs,
                metrics["brighten"],
            )

            darken_imgs_A = self.robustness.brightness(imgs_A, 0.75)
            self._get_single_robustness_operate_metric(
                imgs_B,
                darken_imgs_A,
                cloak_imgs,
                metrics["darken"],
            )

            torch.cuda.empty_cache()

            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            noise
            utility: {metric.generate_summary_utility_log(metrics['noise'], 'utility', idx)}
            pert source utility: {metric.generate_summary_utility_log(metrics['noise'], 'pert_source_utility', idx)}
            pert target utility: {metric.generate_summary_utility_log(metrics['noise'], 'pert_target_utility', idx)}
            pert source effectiveness: {metric.generate_summary_effectiveness_log(metrics['noise'], 'pert_source_effectiveness')}
            pert target effectiveness: {metric.generate_summary_effectiveness_log(metrics['noise'], 'pert_target_effectiveness')}
            compress
            utility: {metric.generate_summary_utility_log(metrics['compress'], 'utility', idx)}
            pert source utility: {metric.generate_summary_utility_log(metrics['compress'], 'pert_source_utility', idx)}
            pert target utility: {metric.generate_summary_utility_log(metrics['compress'], 'pert_target_utility', idx)}
            pert source effectiveness: {metric.generate_summary_effectiveness_log(metrics['compress'], 'pert_source_effectiveness')}
            pert target effectiveness: {metric.generate_summary_effectiveness_log(metrics['compress'], 'pert_target_effectiveness')}
            crop
            utility: {metric.generate_summary_utility_log(metrics['crop'], 'utility', idx)}
            pert source utility: {metric.generate_summary_utility_log(metrics['crop'], 'pert_source_utility', idx)}
            pert target utility: {metric.generate_summary_utility_log(metrics['crop'], 'pert_target_utility', idx)}
            pert source effectiveness: {metric.generate_summary_effectiveness_log(metrics['crop'], 'pert_source_effectiveness')}
            pert target effectiveness: {metric.generate_summary_effectiveness_log(metrics['crop'], 'pert_target_effectiveness')}
            logo
            utility: {metric.generate_summary_utility_log(metrics['logo'], 'utility', idx)}
            pert source utility: {metric.generate_summary_utility_log(metrics['logo'], 'pert_source_utility', idx)}
            pert target utility: {metric.generate_summary_utility_log(metrics['logo'], 'pert_target_utility', idx)}
            pert source effectiveness: {metric.generate_summary_effectiveness_log(metrics['logo'], 'pert_source_effectiveness')}
            pert target effectiveness: {metric.generate_summary_effectiveness_log(metrics['logo'], 'pert_target_effectiveness')}
            brighten
            utility: {metric.generate_summary_utility_log(metrics['brighten'], 'utility', idx)}
            pert source utility: {metric.generate_summary_utility_log(metrics['brighten'], 'pert_source_utility', idx)}
            pert target utility: {metric.generate_summary_utility_log(metrics['brighten'], 'pert_target_utility', idx)}
            pert source effectiveness: {metric.generate_summary_effectiveness_log(metrics['brighten'], 'pert_source_effectiveness')}
            pert target effectiveness: {metric.generate_summary_effectiveness_log(metrics['brighten'], 'pert_target_effectiveness')}
            darken
            utility: {metric.generate_summary_utility_log(metrics['darken'], 'utility', idx)}
            pert source utility: {metric.generate_summary_utility_log(metrics['darken'], 'pert_source_utility', idx)}
            pert target utility: {metric.generate_summary_utility_log(metrics['darken'], 'pert_target_utility', idx)}
            pert source effectiveness: {metric.generate_summary_effectiveness_log(metrics['darken'], 'pert_source_effectiveness')}
            pert target effectiveness: {metric.generate_summary_effectiveness_log(metrics['darken'], 'pert_target_effectiveness')}
            """
            )

            self.logger.info(textwrap.indent(summary_log_str, "    "))

    def adaptive_attack_with_other_image(self) -> None:
        metrics = metric.get_metric_data_template(self.effectiveness)

        dataset = AdaptiveMetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B, imgs_C) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B, imgs_C = imgs_A.cuda(), imgs_B.cuda(), imgs_C.cuda()
            total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            pert_imgs_A = self._perturb_imgs(imgs_A, cloak_imgs)
            x_imgs_B = self._perturb_imgs(imgs_B, cloak_imgs)

            torch.set_grad_enabled(False)
            pert_imgs = pert_imgs_A - (x_imgs_B - imgs_B)
            (
                source_swap,
                pert_source_swap,
                target_swap,
                pert_target_swap,
            ) = self._get_full_swap_results(imgs_A, imgs_C, pert_imgs)

            (
                utility,
                source_utility,
                target_utility,
                source_effectiveness,
                target_effectiveness,
            ) = metric.get_defense_metric(
                self.utility,
                self.effectiveness,
                imgs_A,
                imgs_C,
                pert_imgs,
                cloak_imgs,
                source_swap,
                pert_source_swap,
                target_swap,
                pert_target_swap,
            )

            metric.merge_metric(
                self.effectiveness,
                metrics,
                utility,
                source_utility,
                target_utility,
                source_effectiveness,
                target_effectiveness,
            )

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "imgs_C",
                    "cloak_imgs",
                    "perturb_imgs",
                    "imgs_A_diff",
                    "imgs_B_diff",
                ],
                [
                    imgs_A,
                    imgs_B,
                    imgs_C,
                    cloak_imgs,
                    pert_imgs,
                    (x_imgs_B - imgs_B) * 10,
                    (pert_imgs_A - imgs_A) * 10,
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            del imgs_A, imgs_B, imgs_C, pert_imgs, cloak_imgs
            del (
                source_swap,
                pert_source_swap,
                target_swap,
                pert_target_swap,
            )
            self._free_gpu()

            scores = self.score_calculator.calculate_score(
                source_effectiveness, target_effectiveness, metrics
            )

            iter_log_str = textwrap.dedent(
                f"""
            utility (mse, psnr, ssim, lpips), effectiveness ({', '.join(self.effectiveness.candi_funcs.keys())}), identity ({', '.join(next(iter(source_effectiveness.values())).keys())}), context ({', '.join(next(iter(target_effectiveness.values())).keys())})
            utility: {metric.generate_iter_utility_log(utility)}
            source utility: {metric.generate_iter_utility_log(source_utility)}
            target utility: {metric.generate_iter_utility_log(target_utility)}
            source effectiveness: {metric.generate_iter_effectiveness_log(source_effectiveness)}
            target effectiveness: {metric.generate_iter_effectiveness_log(target_effectiveness)}
            scores: {metric.generate_iter_score_log(scores)}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            utility: {metric.generate_summary_utility_log(metrics, 'utility', idx)}
            pert source utility: {metric.generate_summary_utility_log(metrics, 'pert_source_utility', idx)}
            pert target utility: {metric.generate_summary_utility_log(metrics, 'pert_target_utility', idx)}
            pert source effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'pert_source_effectiveness')}
            pert target effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'pert_target_effectiveness')}
            scores: {metric.generate_summary_score_log(scores)}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

    def cloak_transfer(self) -> None:
        dataset = AdaptiveMetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )

        metrics = {
            effec_name: {
                "B": (0, 0),
                "A": (0, 0),
                "cloak_A": (0, 0),
            }
            for effec_name in self.effectiveness.candi_funcs.keys()
        }
        total_count = 0

        def merge_matches(matches: dict) -> None:
            for effec_name, item in matches.items():
                for key, value in item.items():
                    prev = metrics[effec_name][key]
                    metrics[effec_name][key] = (
                        prev[0] + value[0],
                        prev[1] + value[1],
                    )

        def format_matches(matches: dict) -> str:
            parts = []
            for effec_name in matches:
                vals = (
                    f"{v[0] / v[1] * 100:.3f}/{v[1]:.0f}"
                    for v in matches[effec_name].values()
                )
                parts.append(f"({', '.join(vals)})")
            return " ".join(parts)

        for idx, (imgs_A, imgs_B, imgs_C) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B, imgs_C = imgs_A.cuda(), imgs_B.cuda(), imgs_C.cuda()
            total_count += len(imgs_A)

            cloak_imgs_A = self.cloak.find_best_cloaks(imgs_A)
            pert_imgs_A = self._perturb_imgs(imgs_A, cloak_imgs_A)
            pert_imgs_B = self._perturb_imgs(imgs_B, pert_imgs_A.detach())

            torch.set_grad_enabled(False)
            final_swap = self.swap_face(pert_imgs_B, imgs_C)

            iter_matches = {
                effec_name: {
                    "B": func(imgs_B, final_swap),
                    "A": func(imgs_A, final_swap),
                    "cloak_A": func(cloak_imgs_A, final_swap),
                }
                for effec_name, func in self.effectiveness.candi_funcs.items()
            }
            merge_matches(iter_matches)

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "imgs_C",
                    "cloak_A",
                    "protected_A",
                    "protected_B",
                    "final_swap",
                    "protected_A_diff",
                    "protected_B_diff",
                ],
                [
                    imgs_A,
                    imgs_B,
                    imgs_C,
                    cloak_imgs_A,
                    pert_imgs_A,
                    pert_imgs_B,
                    final_swap,
                    (pert_imgs_A - imgs_A) * 10,
                    (pert_imgs_B - imgs_B) * 10,
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            iter_log_str = textwrap.dedent(
                f"""
                [Cloak Transfer][Iter][Batch {idx:4}/{len(dataloader):4}]
                result matching (B, A, cloak_A): {format_matches(iter_matches)}
                """
            )
            summary_log_str = textwrap.dedent(
                f"""
                [Cloak Transfer][Summary][Batch {idx:4}/{len(dataloader):4}, {total_count} triples]
                result matching (B, A, cloak_A): {format_matches(metrics)}
                """
            )
            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

            del (
                imgs_A,
                imgs_B,
                imgs_C,
                cloak_imgs_A,
                pert_imgs_A,
                pert_imgs_B,
                final_swap,
            )
            self._free_gpu()

    def adaptive_attack_with_self_image(self) -> None:
        metrics = metric.get_metric_data_template(self.effectiveness)

        dataset = MetricDataset(self.config)
        dataloader = DataLoader(
            dataset, batch_size=self.config.third_party.defense.batch_size, shuffle=True
        )
        total_count = 0
        for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
            torch.set_grad_enabled(True)
            imgs_A, imgs_B = imgs_A.cuda(), imgs_B.cuda()
            total_count += len(imgs_A)

            cloak_imgs = self.cloak.find_best_cloaks(imgs_A)
            pert_imgs_A = self._perturb_imgs(imgs_A, cloak_imgs)
            pert_pert_imgs_A = self._perturb_imgs(pert_imgs_A, cloak_imgs)

            torch.set_grad_enabled(False)
            pert_imgs = pert_imgs_A - (pert_pert_imgs_A - pert_imgs_A)
            (
                source_swap,
                pert_source_swap,
                target_swap,
                pert_target_swap,
            ) = self._get_full_swap_results(imgs_A, imgs_B, pert_imgs)

            (
                utility,
                source_utility,
                target_utility,
                source_effectiveness,
                target_effectiveness,
            ) = metric.get_defense_metric(
                self.utility,
                self.effectiveness,
                imgs_A,
                imgs_B,
                pert_imgs,
                cloak_imgs,
                source_swap,
                pert_source_swap,
                target_swap,
                pert_target_swap,
            )

            metric.merge_metric(
                self.effectiveness,
                metrics,
                utility,
                source_utility,
                target_utility,
                source_effectiveness,
                target_effectiveness,
            )

            save_tensor_imgs(
                self.image_dir,
                idx,
                [
                    "imgs_A",
                    "imgs_B",
                    "cloak_imgs",
                    "perturb_imgs",
                    "perturb_imgs_A_source_swap",
                    "perturb_imgs_A_target_swap",
                    "protect_diff",
                    "attack_diff",
                ],
                [
                    imgs_A,
                    imgs_B,
                    cloak_imgs,
                    pert_imgs,
                    pert_source_swap,
                    pert_target_swap,
                    (pert_imgs_A - imgs_A) * 10,
                    (pert_pert_imgs_A - pert_imgs_A) * 10,
                ],
                only_save_summary=self.config.third_party.defense.only_save_summary,
            )

            del imgs_A, imgs_B, cloak_imgs, pert_imgs
            del (
                source_swap,
                pert_source_swap,
                target_swap,
                pert_target_swap,
            )
            self._free_gpu()

            scores = self.score_calculator.calculate_score(
                source_effectiveness, target_effectiveness, metrics
            )

            iter_log_str = textwrap.dedent(
                f"""
            utility (mse, psnr, ssim, lpips), effectiveness ({', '.join(self.effectiveness.candi_funcs.keys())}), identity ({', '.join(next(iter(source_effectiveness.values())).keys())}), context ({', '.join(next(iter(target_effectiveness.values())).keys())})
            utility: {metric.generate_iter_utility_log(utility)}
            source utility: {metric.generate_iter_utility_log(source_utility)}
            target utility: {metric.generate_iter_utility_log(target_utility)}
            source effectiveness: {metric.generate_iter_effectiveness_log(source_effectiveness)}
            target effectiveness: {metric.generate_iter_effectiveness_log(target_effectiveness)}
            scores: {metric.generate_iter_score_log(scores)}
            """
            )
            summary_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} pairs of pictures
            utility: {metric.generate_summary_utility_log(metrics, 'utility', idx)}
            pert source utility: {metric.generate_summary_utility_log(metrics, 'pert_source_utility', idx)}
            pert target utility: {metric.generate_summary_utility_log(metrics, 'pert_target_utility', idx)}
            pert source effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'pert_source_effectiveness')}
            pert target effectiveness: {metric.generate_summary_effectiveness_log(metrics, 'pert_target_effectiveness')}
            scores: {metric.generate_summary_score_log(scores)}
            """
            )

            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))

    def _perturb_imgs(self, imgs: Tensor, cloak_imgs: Tensor) -> Tensor:
        def l2_per_image(x: Tensor, y: Tensor) -> Tensor:
            return ((x - y) ** 2).view(x.size(0), -1).mean(dim=1)

        x_imgs = imgs.clone().detach() + torch.randn_like(imgs) * 1e-5

        with torch.no_grad():
            self_identity = self._get_imgs_identity(imgs)
            cloak_identity = self._get_imgs_identity(cloak_imgs)
            imgs_latent_code = self.target.netG.encoder(x_imgs)  # type: ignore

        epsilon = (
            self.config.third_party.defense.epsilon
            * (torch.max(x_imgs) - torch.min(x_imgs))
            / 2
        )
        limits = (
            tensor(
                [
                    self.config.third_party.defense.limit.R,
                    self.config.third_party.defense.limit.G,
                    self.config.third_party.defense.limit.B,
                ]
            )
            .view(1, 3, 1, 1)
            .to(imgs.device)
        )

        B = imgs.size(0)
        best_imgs = imgs.clone()
        best_loss = torch.full((B,), float("inf"), device=imgs.device)

        for epoch in range(self.config.third_party.defense.epochs):
            x_imgs = x_imgs.clone().detach().requires_grad_(True)

            pert_diff_loss = (
                self.config.third_party.defense.weight.perturb
                * l2_per_image(x_imgs, imgs.detach())
            )

            x_identity = self._get_imgs_identity(x_imgs)
            identity_diff = torch.clamp(
                l2_per_image(x_identity, self_identity),
                0,
                self.config.third_party.defense.limit.identity,
            )
            identity_diff_loss = (
                -self.config.third_party.defense.weight.identity * identity_diff
            )

            cloak_diff_loss = (
                self.config.third_party.defense.weight.cloak
                * l2_per_image(x_identity, cloak_identity)
            )

            x_latent_code = self.target.netG.encoder(x_imgs)  # type: ignore
            context_diff = torch.clamp(
                l2_per_image(x_latent_code, imgs_latent_code),
                0,
                self.config.third_party.defense.limit.context,
            )
            context_diff_loss = (
                -self.config.third_party.defense.weight.context * context_diff
            )

            loss_per_img = (
                pert_diff_loss
                + identity_diff_loss
                + cloak_diff_loss
                + context_diff_loss
            )
            loss = loss_per_img.mean()
            loss.backward()

            if x_imgs.grad is not None:
                grad_sign = x_imgs.grad.sign().detach()
            else:
                grad_sign = torch.zeros_like(x_imgs)

            x_imgs = x_imgs.detach() - epsilon * grad_sign
            x_imgs = torch.clamp(
                x_imgs,
                min=imgs - limits,
                max=imgs + limits,
            )
            x_imgs = torch.clamp(x_imgs, 0, 1)

            loss_per_img_detached = loss_per_img.detach()
            improved = loss_per_img_detached < best_loss
            best_loss[improved] = loss_per_img_detached[improved]
            best_imgs[improved] = x_imgs[improved].detach()

            if (
                not self.config.third_party.defense.silent_perturb
                and (epoch + 1) % self.config.third_party.defense.log_interval == 0
                or (epoch + 1) == self.config.third_party.defense.epochs
            ):
                self.logger.info(
                    f"[Epoch {epoch+1:4}/{self.config.third_party.defense.epochs:4}] "
                    f"loss: {loss.item():.5f}("
                    f"{pert_diff_loss.mean().item():.5f}, "
                    f"{identity_diff_loss.mean().item():.5f}, "
                    f"{cloak_diff_loss.mean().item():.5f}, "
                    f"{context_diff_loss.mean().item():.5f})"
                )

        return best_imgs

    def _get_full_swap_results(
        self, imgs_A: Tensor, imgs_B: Tensor, pert_imgs_A: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        imgs_A_identity = self._get_imgs_identity(imgs_A)
        source_swap = self.target(None, imgs_B, imgs_A_identity, None, True)

        pert_imgs_A_identity = self._get_imgs_identity(pert_imgs_A)
        pert_source_swap = self.target(None, imgs_B, pert_imgs_A_identity, None, True)

        imgs_B_identity = self._get_imgs_identity(imgs_B)
        target_swap = self.target(None, imgs_A, imgs_B_identity, None, True)

        pert_target_swap = self.target(None, pert_imgs_A, imgs_B_identity, None, True)

        return (
            source_swap,
            pert_source_swap,
            target_swap,
            pert_target_swap,
        )

    def _get_single_robustness_operate_metric(
        self,
        imgs_B: Tensor,
        operate_imgs_A: Tensor,
        cloak_imgs: Tensor,
        data: dict,
    ) -> None:
        torch.set_grad_enabled(True)
        operate_x_imgs = self._perturb_imgs(operate_imgs_A, cloak_imgs)
        torch.set_grad_enabled(False)
        (
            source_swap,
            pert_source_swap,
            target_swap,
            pert_target_swap,
        ) = self._get_full_swap_results(operate_imgs_A, imgs_B, operate_x_imgs)

        (
            utility,
            source_utility,
            target_utility,
            source_effectiveness,
            target_effectiveness,
        ) = metric.get_defense_metric(
            self.utility,
            self.effectiveness,
            operate_imgs_A,
            imgs_B,
            operate_x_imgs,
            cloak_imgs,
            source_swap,
            pert_source_swap,
            target_swap,
            pert_target_swap,
        )

        metric.merge_metric(
            self.effectiveness,
            data,
            utility,
            source_utility,
            target_utility,
            source_effectiveness,
            target_effectiveness,
        )

    def _pick_on_cpu_then_to_gpu(
        self, big_cpu: Tensor, idx_cpu: Tensor, device="cuda"
    ) -> Tensor:
        with torch.no_grad():
            small_cpu = torch.index_select(big_cpu, 0, idx_cpu)
            small_gpu = small_cpu.to(device, non_blocking=True)
            return small_gpu

    def _free_gpu(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
