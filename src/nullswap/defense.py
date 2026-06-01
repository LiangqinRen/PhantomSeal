from src.nullswap.base import Base
from src.evaluate import Effectiveness, ScoreCalculator
import src.metric as metric

import textwrap
import torch


class Defense(Base):
    def train(self) -> None:
        dataloader = self.get_train_dataloader()
        defense_config = self.config.third_party.defense

        best_loss = float("inf")
        global_step = 0

        for epoch in range(1, defense_config.epochs + 1):
            self.generator.train()
            self.discriminator.train()

            for batch_idx, imgs in enumerate(dataloader, start=1):
                global_step += 1
                imgs = imgs.to(self.device)

                outputs = self.generator(imgs)
                discriminator_loss = self.compute_discriminator_loss(
                    imgs,
                    outputs["cloak"],
                )
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                outputs = self.generator(imgs)
                generator_loss, log_items = self.compute_generator_loss(imgs, outputs)
                self.generator_optimizer.zero_grad()
                generator_loss.backward()
                self.generator_optimizer.step()

                if global_step % defense_config.log_interval == 0:
                    log_str = textwrap.dedent(
                        f"""
                        [Epoch {epoch:4}/{defense_config.epochs:4}][Batch {batch_idx:4}/{len(dataloader):4}]
                        generator_loss: {generator_loss.item():.3f}, discriminator_loss: {discriminator_loss.item():.3f}
                        losses: ({log_items['loss_mse']:.5f}, {log_items['loss_lpips']:.5f}, {log_items['loss_adv']:.5f}, {log_items['loss_identity']:.5f})
                        recognizer_losses: ({log_items['loss_arcface']:.5f}, {log_items['loss_facenet']:.5f})
                        perturb_mse: {log_items['perturb_mse']:.3f}, perturb_mse_ema: {log_items['perturb_mse_ema']:.3f}, perturb_weight_scale: {log_items['perturb_weight_scale']:.3f}
                        dlw_progress: ({log_items['dlw_progress_arcface']:.5f}, {log_items['dlw_progress_facenet']:.5f})
                        dlw_variance: ({log_items['dlw_variance_arcface']:.5f}, {log_items['dlw_variance_facenet']:.5f})
                        dlw_weight: ({log_items['dlw_weight_arcface']:.5f}, {log_items['dlw_weight_facenet']:.5f})
                        delta_l2: {log_items['delta_l2']:.6f}
                        """
                    )
                    self.logger.info(textwrap.indent(log_str, "    "))

                    self.save_training_images(
                        global_step,
                        imgs.detach(),
                        outputs["cloak"].detach(),
                        outputs["delta"].detach(),
                    )

                if (
                    defense_config.eval_interval_batches > 0
                    and global_step % defense_config.eval_interval_batches == 0
                ):
                    eval_result = self.run_periodic_simswap_eval()
                    eval_log_str = textwrap.dedent(
                        f"""
                        [Periodic Eval][Step {global_step:6}]
                        protection utility: {metric.generate_iter_utility_log(eval_result['utility'])}
                        𝒯_identity utility: {metric.generate_iter_utility_log(eval_result['source_utility'])}
                        𝒯_context utility: {metric.generate_iter_utility_log(eval_result['target_utility'])}
                        𝒯_identity effectiveness {metric.generate_iter_effectiveness_label(eval_result['source_effectiveness'])}: {metric.generate_iter_effectiveness_log(eval_result['source_effectiveness'], include_labels=False)}
                        𝒯_context effectiveness {metric.generate_iter_effectiveness_label(eval_result['target_effectiveness'])}: {metric.generate_iter_effectiveness_log(eval_result['target_effectiveness'], include_labels=False)}
                        scores: {metric.generate_iter_score_log(eval_result['scores'])}
                        """
                    )
                    self.logger.info(textwrap.indent(eval_log_str, "    "))

                if global_step % defense_config.checkpoint_interval == 0:
                    best_loss = min(best_loss, generator_loss.item())
                    self.save_checkpoint(epoch, global_step, best_loss)

                if generator_loss.item() < best_loss:
                    best_loss = generator_loss.item()
                    torch.save(
                        {
                            "generator": self.generator.state_dict(),
                            "discriminator": self.discriminator.state_dict(),
                            "generator_optimizer": self.generator_optimizer.state_dict(),
                            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                            "epoch": epoch,
                            "step": global_step,
                            "best_loss": best_loss,
                        },
                        self.checkpoint_dir / "best.pth",
                    )

    def sample(self) -> None:
        dataloader = self.get_sample_dataloader()
        self.generator.eval()

        with torch.no_grad():
            for idx, imgs in enumerate(dataloader, start=1):
                imgs = imgs.to(self.device)
                outputs = self.generator(imgs)
                self.save_training_images(
                    idx,
                    imgs,
                    outputs["cloak"],
                    outputs["delta"],
                )

    def metric(self) -> None:
        self._build_metric_targets()
        if not hasattr(self, "effectiveness"):
            self.effectiveness = Effectiveness(self.logger, self.config)
        if not hasattr(self, "score_calculator"):
            self.score_calculator = ScoreCalculator(self.logger, self.config)

        dataloader = self.get_metric_dataloader()
        self.generator.eval()

        metrics_by_target = {
            target_name: metric.get_metric_data_template(self.effectiveness)
            for target_name in self.config.third_party.defense.metric_targets
        }
        total_count = 0

        with torch.no_grad():
            for idx, (imgs_A, imgs_B) in enumerate(dataloader, start=1):
                imgs_A = imgs_A.to(self.device)
                imgs_B = imgs_B.to(self.device)
                total_count += len(imgs_A)

                outputs = self.generator(imgs_A)
                pert_imgs = outputs["cloak"]
                perturbation = pert_imgs - imgs_A
                pert_imgs = torch.clamp(imgs_A + perturbation * 2.75, 0, 1)

                for target_name in self.config.third_party.defense.metric_targets:
                    source_swap = self.swap_face_with_target(
                        target_name, imgs_A, imgs_B
                    )
                    pert_source_swap = self.swap_face_with_target(
                        target_name, pert_imgs, imgs_B
                    )
                    target_swap = self.swap_face_with_target(
                        target_name, imgs_B, imgs_A
                    )
                    pert_target_swap = self.swap_face_with_target(
                        target_name, imgs_B, pert_imgs
                    )

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
                        None,
                        source_swap,
                        pert_source_swap,
                        target_swap,
                        pert_target_swap,
                    )
                    for effec_name in source_effectiveness.keys():
                        if "cloak" not in source_effectiveness[effec_name]:
                            source_effectiveness[effec_name]["cloak"] = (0, 1)

                    metric.merge_metric(
                        self.effectiveness,
                        metrics_by_target[target_name],
                        utility,
                        source_utility,
                        target_utility,
                        source_effectiveness,
                        target_effectiveness,
                    )

                    iter_scores = self.score_calculator.calculate_score(
                        source_effectiveness,
                        target_effectiveness,
                        metrics_by_target[target_name],
                    )
                    summary_scores = self.score_calculator.calculate_score(
                        metrics_by_target[target_name]["pert_source_effectiveness"],
                        metrics_by_target[target_name]["pert_target_effectiveness"],
                        metrics_by_target[target_name],
                    )
                    iter_log_str = textwrap.dedent(
                        f"""
                        [{target_name}][Iter][Batch {idx:4}/{len(dataloader):4}]
                        protection utility: {metric.generate_iter_utility_log(utility)}
                        𝒯_identity utility: {metric.generate_iter_utility_log(source_utility)}
                        𝒯_context utility: {metric.generate_iter_utility_log(target_utility)}
                        𝒯_identity effectiveness {metric.generate_iter_effectiveness_label(source_effectiveness)}: {metric.generate_iter_effectiveness_log(source_effectiveness, include_labels=False)}
                        𝒯_context effectiveness {metric.generate_iter_effectiveness_label(target_effectiveness)}: {metric.generate_iter_effectiveness_log(target_effectiveness, include_labels=False)}
                        scores: {metric.generate_iter_score_log(iter_scores)}
                        """
                    )
                    summary_log_str = textwrap.dedent(
                        f"""
                        [{target_name}][Summary][Batch {idx:4}/{len(dataloader):4}, {total_count} pairs]
                        protection utility: {metric.generate_summary_utility_log(metrics_by_target[target_name], 'utility', idx)}
                        𝒯_identity utility: {metric.generate_summary_utility_log(metrics_by_target[target_name], 'pert_source_utility', idx)}
                        𝒯_context utility: {metric.generate_summary_utility_log(metrics_by_target[target_name], 'pert_target_utility', idx)}
                        𝒯_identity effectiveness {metric.generate_summary_effectiveness_label(metrics_by_target[target_name], 'pert_source_effectiveness')}: {metric.generate_summary_effectiveness_log(metrics_by_target[target_name], 'pert_source_effectiveness', include_labels=False)}
                        𝒯_context effectiveness {metric.generate_summary_effectiveness_label(metrics_by_target[target_name], 'pert_target_effectiveness')}: {metric.generate_summary_effectiveness_log(metrics_by_target[target_name], 'pert_target_effectiveness', include_labels=False)}
                        scores: {metric.generate_summary_score_log(summary_scores)}
                        """
                    )
                    self.logger.info(textwrap.indent(iter_log_str, "    "))
                    self.logger.info(textwrap.indent(summary_log_str, "    "))

                self.save_training_images(
                    idx,
                    imgs_A,
                    pert_imgs,
                    outputs["delta"],
                )
