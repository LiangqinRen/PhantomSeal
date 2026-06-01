from src.common_utils import save_tensor_imgs, cd, use_project
from src.dataset import FFHQSample, FFHQMetric
from src.evaluate import Utility, Effectiveness, ScoreCalculator
from src.nullswap.model import NullSwap, NullSwapDiscriminator
from src.simswap.options import build_simswap_test_options
import src.metric as metric

import inspect
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision import transforms
from collections import deque
from torch.serialization import add_safe_globals
from typing import Callable, Any, cast
from omegaconf import OmegaConf


class NullSwapFFHQTrainDataset(Dataset):
    def __init__(self, root_dir: Path, image_size: int):
        self.root_dir = Path(root_dir)
        self.images = sorted([f for f in self.root_dir.iterdir() if f.is_file()])
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tensor:
        idx = np.random.randint(0, len(self.images))
        from PIL import Image

        return self.transform(Image.open(self.images[idx]).convert("RGB"))


class Base:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.device = torch.device("cuda")

        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        notes_path = Path(self.config.notes_path)
        notes_path.touch(exist_ok=True)

        model_config = self.config.third_party.model
        self.generator = NullSwap(
            image_channels=model_config.image_channels,
            epsilon=model_config.epsilon,
            id_base_channels=model_config.id_base_channels,
            id_bottleneck_channels=model_config.id_bottleneck_channels,
            id_num_blocks=model_config.id_num_blocks,
            feature_conv_channels=tuple(model_config.feature_conv_channels),
            feature_bottleneck_channels=model_config.feature_bottleneck_channels,
            feature_num_blocks=model_config.feature_num_blocks,
            perturb_refine_channels=model_config.perturb_refine_channels,
            perturb_bottleneck_channels=model_config.perturb_bottleneck_channels,
            perturb_num_blocks=model_config.perturb_num_blocks,
            perturb_feature_size=tuple(model_config.perturb_feature_size),
            cloak_hidden_channels=model_config.cloak_hidden_channels,
            cloak_bottleneck_channels=model_config.cloak_bottleneck_channels,
            cloak_num_blocks=model_config.cloak_num_blocks,
            reduction=model_config.reduction,
            alpha_init=model_config.alpha_init,
            beta_init=model_config.beta_init,
        ).to(self.device)
        self.discriminator = NullSwapDiscriminator(
            in_channels=model_config.image_channels,
            base_channels=model_config.discriminator_base_channels,
        ).to(self.device)

        self._build_recognizers()

        with torch.no_grad():
            self.lpips_distance = lpips.LPIPS(net="alex", verbose=False).to(self.device)
            self.lpips_distance.eval()
        self.utility = Utility(logger, config)

        self.generator_optimizer = Adam(
            self.generator.parameters(),
            lr=self.config.third_party.defense.lr_generator,
            betas=tuple(self.config.third_party.defense.betas),
        )
        self.discriminator_optimizer = Adam(
            self.discriminator.parameters(),
            lr=self.config.third_party.defense.lr_discriminator,
            betas=tuple(self.config.third_party.defense.betas),
        )
        self.gan_criterion = nn.BCEWithLogitsLoss()

        self.checkpoint_dir = Path(self.config.log_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.config.third_party.function != "train":
            checkpoint_path = Path(self.config.third_party.defense.checkpoint_path)
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.generator.load_state_dict(checkpoint["generator"])
                self.discriminator.load_state_dict(checkpoint["discriminator"])
            else:
                self.logger.warning(f"Checkpoint not found: {checkpoint_path}")

        self.loss_history = {
            "arcface": deque(maxlen=self.config.third_party.defense.dynamic_weight_k),
            "facenet": deque(maxlen=self.config.third_party.defense.dynamic_weight_k),
        }
        self.utility_mse_ema: float | None = None
        self.periodic_eval_loader = None
        self.periodic_eval_iter = None
        self.eval_targets: dict[str, object] = {}
        if self.config.third_party.defense.eval_interval_batches > 0:
            self.effectiveness = Effectiveness(logger, config)
            self.score_calculator = ScoreCalculator(logger, config)
            self._build_simswap_evaluator()
            self.periodic_eval_loader = self.get_periodic_eval_dataloader()
            self.periodic_eval_iter = iter(self.periodic_eval_loader)

    def _build_recognizers(self) -> None:
        from facenet_pytorch import InceptionResnetV1

        faceshifter_root = Path(self.config.third_party.faceshifter_dir)
        import sys

        if str(faceshifter_root) not in sys.path:
            sys.path.insert(0, str(faceshifter_root))
        if str(faceshifter_root / "face_modules") not in sys.path:
            sys.path.insert(0, str(faceshifter_root / "face_modules"))

        from face_modules.model import Backbone

        self.arcface = Backbone(50, 0.6, "ir_se").to(self.device)
        arcface_model_path = (
            faceshifter_root / self.config.third_party.origin.faceshifter.model_path
        )
        self.arcface.load_state_dict(
            torch.load(
                arcface_model_path,
                weights_only=True,
            ),
            strict=False,
        )
        self.arcface.eval()
        for param in self.arcface.parameters():
            param.requires_grad_(False)

        self.facenet = InceptionResnetV1(
            classify=False,
            pretrained="vggface2",
        ).to(self.device)
        self.facenet.eval()
        for param in self.facenet.parameters():
            param.requires_grad_(False)

    def get_train_dataloader(self) -> DataLoader:
        dataset = NullSwapFFHQTrainDataset(
            root_dir=Path(self.config.third_party.dataset.metric_dir),
            image_size=self.config.third_party.dataset.image_size,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.third_party.defense.batch_size,
            shuffle=True,
            num_workers=self.config.third_party.defense.num_workers,
        )

    def get_sample_dataloader(self) -> DataLoader:
        dataset = FFHQSample(
            Path(self.config.third_party.dataset.sample_dir),
            self.config.third_party.dataset.metric_pairs,
            transforms.Compose(
                [
                    transforms.Resize(
                        (
                            self.config.third_party.dataset.image_size,
                            self.config.third_party.dataset.image_size,
                        )
                    ),
                    transforms.ToTensor(),
                ]
            ),
        )
        return DataLoader(
            dataset,
            batch_size=self.config.third_party.defense.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def get_periodic_eval_dataloader(self) -> DataLoader:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        self.config.third_party.dataset.image_size,
                        self.config.third_party.dataset.image_size,
                    )
                ),
                transforms.ToTensor(),
            ]
        )
        dataset = FFHQMetric(
            Path(self.config.third_party.dataset.metric_dir),
            self.config.third_party.defense.eval_pairs,
            transform,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.third_party.defense.eval_batch_size,
            shuffle=True,
            num_workers=0,
        )

    def get_metric_dataloader(self) -> DataLoader:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        self.config.third_party.dataset.image_size,
                        self.config.third_party.dataset.image_size,
                    )
                ),
                transforms.ToTensor(),
            ]
        )
        dataset = FFHQMetric(
            Path(self.config.third_party.dataset.metric_dir),
            self.config.third_party.dataset.metric_pairs,
            transform,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.third_party.defense.metric_batch_size,
            shuffle=True,
            num_workers=0,
        )

    def save_checkpoint(self, epoch: int, step: int, best_loss: float) -> None:
        checkpoint = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_loss": best_loss,
        }
        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")
        torch.save(
            checkpoint,
            self.checkpoint_dir / f"epoch_{epoch:04d}_step_{step:06d}.pth",
        )

    def save_training_images(
        self,
        step: int,
        imgs: Tensor,
        pert_imgs: Tensor,
        delta: Tensor,
    ) -> None:
        delta_vis = torch.clamp(
            delta / (2 * self.config.third_party.model.epsilon) + 0.5,
            0,
            1,
        )
        save_tensor_imgs(
            self.image_dir,
            step,
            ["imgs", "pert_imgs", "delta"],
            [imgs, pert_imgs, delta_vis],
            only_save_summary=self.config.third_party.defense.only_save_summary,
        )

    @staticmethod
    def l2_per_image(x: Tensor, y: Tensor) -> Tensor:
        return ((x - y) ** 2).view(x.size(0), -1).mean(dim=1)

    def get_utility_mse(self, x: Tensor, y: Tensor) -> Tensor:
        return self.l2_per_image(x, y) * (255.0**2)

    def get_perturb_weight_scale(self, utility_mse_ema: float, batch_size: int) -> Tensor:
        lower_bound = self.config.third_party.defense.utility_mse_lower_bound
        upper_bound = self.config.third_party.defense.utility_mse_upper_bound
        smoothness = self.config.third_party.defense.utility_mse_smoothness

        utility_mse = torch.full(
            (batch_size,),
            utility_mse_ema,
            device=self.device,
            dtype=torch.float32,
        )
        lower_gate = torch.sigmoid((utility_mse - lower_bound) / smoothness)
        upper_gate = torch.sigmoid((utility_mse - upper_bound) / smoothness)
        return torch.clamp(lower_gate + upper_gate, min=0.0, max=2.0)

    def get_arcface_embedding(self, imgs: Tensor) -> Tensor:
        return F.normalize(
            self.arcface(
                F.interpolate(
                    imgs[:, :, 19:237, 19:237],
                    (112, 112),
                    mode="bilinear",
                    align_corners=True,
                )
            ),
            dim=-1,
            p=2,
        )

    def get_facenet_embedding(self, imgs: Tensor) -> Tensor:
        imgs = F.interpolate(imgs, size=(160, 160), mode="bilinear", align_corners=False)
        imgs = (imgs - 0.5) / 0.5
        return F.normalize(self.facenet(imgs), dim=-1, p=2)

    def get_identity_losses(
        self,
        imgs: Tensor,
        pert_imgs: Tensor,
    ) -> dict[str, Tensor]:
        clean_arcface = self.get_arcface_embedding(imgs)
        pert_arcface = self.get_arcface_embedding(pert_imgs)
        clean_facenet = self.get_facenet_embedding(imgs)
        pert_facenet = self.get_facenet_embedding(pert_imgs)

        return {
            "arcface": F.cosine_similarity(clean_arcface, pert_arcface, dim=1).mean(),
            "facenet": F.cosine_similarity(clean_facenet, pert_facenet, dim=1).mean(),
        }

    def get_dlw_weights(self, identity_losses: dict[str, Tensor]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        eps = self.config.third_party.defense.dynamic_weight_epsilon
        variance_weight = self.config.third_party.defense.dynamic_weight_variance_weight

        variances: dict[str, float] = {}
        progresses: dict[str, float] = {}
        raw_weights: dict[str, float] = {}

        for name, loss in identity_losses.items():
            current = float(loss.detach().item())
            history = self.loss_history[name]
            history.append(current)

            if len(history) > 1:
                variance = float(np.var(history))
                reference = float(np.mean(list(history)[:-1]))
                progress = current / max(reference, eps)
            else:
                variance = 0.0
                progress = 1.0

            variances[name] = variance
            progresses[name] = progress

        variance_max = max(max(variances.values()), eps)
        for name in identity_losses:
            variance_penalty = variances[name] / variance_max
            raw_weights[name] = progresses[name] / (1.0 + variance_weight * variance_penalty)

        raw_sum = sum(raw_weights.values())
        normalized = {
            name: len(raw_weights) * raw_weights[name] / max(raw_sum, eps)
            for name in raw_weights
        }

        return normalized, progresses, variances

    def compute_generator_loss(
        self,
        imgs: Tensor,
        outputs: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, float]]:
        pert_imgs = outputs["cloak"]
        delta = outputs["delta"]

        mse_loss = F.mse_loss(pert_imgs, imgs)
        lpips_loss = self.lpips_distance(pert_imgs, imgs).mean()

        utility_mse = self.get_utility_mse(pert_imgs, imgs)
        utility_mse_mean = float(utility_mse.mean().item())
        if self.utility_mse_ema is None:
            self.utility_mse_ema = utility_mse_mean
        else:
            decay = self.config.third_party.defense.utility_mse_ema_decay
            self.utility_mse_ema = decay * self.utility_mse_ema + (1 - decay) * utility_mse_mean
        perturb_weight_scale = self.get_perturb_weight_scale(self.utility_mse_ema, imgs.size(0))
        perturb_reg = (
            self.config.third_party.defense.weight.perturb
            * perturb_weight_scale
            * self.l2_per_image(pert_imgs, imgs)
        ).mean()

        fake_logits = self.discriminator(pert_imgs)
        adv_loss = self.gan_criterion(fake_logits, torch.ones_like(fake_logits))

        identity_losses = self.get_identity_losses(imgs, pert_imgs)
        dlw_weights, progresses, variances = self.get_dlw_weights(identity_losses)
        identity_loss = sum(dlw_weights[name] * identity_losses[name] for name in identity_losses)

        total_loss = (
            self.config.third_party.defense.loss_weight.mse * mse_loss
            + self.config.third_party.defense.loss_weight.lpips * lpips_loss
            + self.config.third_party.defense.loss_weight.adv * adv_loss
            + self.config.third_party.defense.loss_weight.identity * identity_loss
            + perturb_reg
        )

        log_items = {
            "loss_total": float(total_loss.item()),
            "loss_mse": float(mse_loss.item()),
            "loss_lpips": float(lpips_loss.item()),
            "loss_adv": float(adv_loss.item()),
            "loss_identity": float(identity_loss.item()),
            "loss_arcface": float(identity_losses["arcface"].item()),
            "loss_facenet": float(identity_losses["facenet"].item()),
            "perturb_mse": utility_mse_mean,
            "perturb_mse_ema": float(self.utility_mse_ema),
            "perturb_weight_scale": float(perturb_weight_scale.mean().item()),
            "dlw_progress_arcface": progresses["arcface"],
            "dlw_progress_facenet": progresses["facenet"],
            "dlw_variance_arcface": variances["arcface"],
            "dlw_variance_facenet": variances["facenet"],
            "dlw_weight_arcface": dlw_weights["arcface"],
            "dlw_weight_facenet": dlw_weights["facenet"],
            "delta_l2": float(self.l2_per_image(pert_imgs, imgs).mean().item()),
        }

        return total_loss, log_items

    def compute_discriminator_loss(self, imgs: Tensor, pert_imgs: Tensor) -> Tensor:
        real_logits = self.discriminator(imgs)
        fake_logits = self.discriminator(pert_imgs.detach())
        real_loss = self.gan_criterion(real_logits, torch.ones_like(real_logits))
        fake_loss = self.gan_criterion(fake_logits, torch.zeros_like(fake_logits))
        return 0.5 * (real_loss + fake_loss)

    def _build_simswap_evaluator(self) -> None:
        simswap_root = Path(self.config.third_party.simswap_dir)
        with use_project([simswap_root]), cd(simswap_root):
            from models.models import create_model
            from models import arcface_models

            self.simswap_test_options = build_simswap_test_options(self.config)

            add_safe_globals(
                [
                    nn.Conv2d,
                    nn.Linear,
                    nn.BatchNorm2d,
                    nn.BatchNorm1d,
                    nn.ReLU,
                    nn.PReLU,
                    nn.Sigmoid,
                    nn.Dropout,
                    nn.Sequential,
                    nn.MaxPool2d,
                    nn.AdaptiveAvgPool2d,
                ]
            )

            for _, obj in inspect.getmembers(arcface_models):
                if inspect.isfunction(obj):
                    add_safe_globals([obj])
                elif inspect.isclass(obj):
                    add_safe_globals([cast(Callable[..., Any], obj)])

            self.simswap_target = create_model(self.simswap_test_options)
            self.simswap_target.cuda().eval()

        self.simswap_arcface_normalize = transforms.Compose(
            [
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _build_metric_targets(self) -> None:
        if self.eval_targets:
            return

        def build_target(config_name: str, model_class):
            target_config = OmegaConf.create(
                OmegaConf.to_container(self.config, resolve=False)
            )
            target_config.third_party = OmegaConf.load(
                Path(self.config.root_dir) / f"config/third_party/{config_name}.yaml"
            )
            merged = OmegaConf.create(
                OmegaConf.to_container(target_config, resolve=True)
            )
            return model_class(self.logger, merged)

        from src.simswap.base import Base as SimSwapBase
        from src.infoswap.base import Base as InfoSwapBase
        from src.uniface.base import Base as UniFaceBase
        from src.e4s.base import Base as E4SBase

        self.eval_targets = {
            "simswap": build_target("simswap", SimSwapBase),
            "infoswap": build_target("infoswap", InfoSwapBase),
            "uniface": build_target("uniface", UniFaceBase),
            "e4s": build_target("e4s", E4SBase),
        }

    def _get_simswap_identity(self, imgs: Tensor) -> Tensor:
        imgs = self.simswap_arcface_normalize(imgs)
        imgs_downsample = F.interpolate(imgs, size=(112, 112))
        prior = self.simswap_target.netArc(imgs_downsample)
        prior = F.normalize(prior, p=2, dim=1)
        return prior.cuda()

    def _get_simswap_eval_identity_from_target(
        self, target, imgs: Tensor
    ) -> Tensor:
        imgs = target.transformer_Arcface(imgs)
        imgs_downsample = F.interpolate(imgs, size=(112, 112))
        netArc = cast(nn.Module, target.target.netArc)
        prior = netArc(imgs_downsample)
        prior = F.normalize(prior, p=2, dim=1)
        return prior.cuda()

    def _get_simswap_eval_results(
        self, imgs_A: Tensor, imgs_B: Tensor, pert_imgs_A: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        imgs_A_identity = self._get_simswap_identity(imgs_A)
        source_swap = self.simswap_target(None, imgs_B, imgs_A_identity, None, True)

        pert_imgs_A_identity = self._get_simswap_identity(pert_imgs_A)
        pert_source_swap = self.simswap_target(
            None, imgs_B, pert_imgs_A_identity, None, True
        )

        imgs_B_identity = self._get_simswap_identity(imgs_B)
        target_swap = self.simswap_target(None, imgs_A, imgs_B_identity, None, True)
        pert_target_swap = self.simswap_target(
            None, pert_imgs_A, imgs_B_identity, None, True
        )

        return source_swap, pert_source_swap, target_swap, pert_target_swap

    def run_periodic_simswap_eval(self) -> dict[str, object]:
        if self.periodic_eval_loader is None or self.periodic_eval_iter is None:
            raise RuntimeError("Periodic evaluation is not enabled.")

        self.generator.eval()
        with torch.no_grad():
            try:
                imgs_A, imgs_B = next(self.periodic_eval_iter)
            except StopIteration:
                self.periodic_eval_iter = iter(self.periodic_eval_loader)
                imgs_A, imgs_B = next(self.periodic_eval_iter)

            imgs_A = imgs_A.to(self.device)
            imgs_B = imgs_B.to(self.device)
            outputs = self.generator(imgs_A)
            pert_imgs = outputs["cloak"]

            (
                source_swap,
                pert_source_swap,
                target_swap,
                pert_target_swap,
            ) = self._get_simswap_eval_results(imgs_A, imgs_B, pert_imgs)

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
            scores = self.score_calculator.calculate_score(
                source_effectiveness,
                target_effectiveness,
                None,
            )

        self.generator.train()
        return {
            "utility": utility,
            "source_utility": source_utility,
            "target_utility": target_utility,
            "source_effectiveness": source_effectiveness,
            "target_effectiveness": target_effectiveness,
            "scores": scores,
        }

    def swap_face_with_target(
        self,
        target_name: str,
        source_imgs: Tensor,
        target_imgs: Tensor,
    ) -> Tensor:
        target = self.eval_targets[target_name]

        if target_name == "simswap":
            source_identity = self._get_simswap_eval_identity_from_target(
                target, source_imgs
            )
            results = target.target(None, target_imgs, source_identity, None, True)
            return results

        if target_name == "infoswap":
            source_imgs = F.interpolate(
                source_imgs, size=(512, 512), mode="bilinear", align_corners=False
            )
            target_imgs = F.interpolate(
                target_imgs, size=(512, 512), mode="bilinear", align_corners=False
            )
            source_imgs = source_imgs * 2 - 1
            target_imgs = target_imgs * 2 - 1
            results = target.swap_face(source_imgs, target_imgs)
        elif target_name == "uniface":
            source_imgs = source_imgs * 2 - 1
            target_imgs = target_imgs * 2 - 1
            results = target.swap_face(source_imgs, target_imgs)
        elif target_name == "e4s":
            source_imgs = F.interpolate(
                source_imgs, size=(1024, 1024), mode="bilinear", align_corners=False
            )
            target_imgs = F.interpolate(
                target_imgs, size=(1024, 1024), mode="bilinear", align_corners=False
            )
            source_imgs = source_imgs * 2 - 1
            target_imgs = target_imgs * 2 - 1
            results = target.swap_face(source_imgs, target_imgs)
        else:
            raise ValueError(f"Unsupported target: {target_name}")

        results = ((results + 1) / 2).clamp(0, 1)
        results = F.interpolate(
            results,
            size=(
                self.config.third_party.dataset.image_size,
                self.config.third_party.dataset.image_size,
            ),
            mode="bilinear",
            align_corners=False,
        )
        return results
