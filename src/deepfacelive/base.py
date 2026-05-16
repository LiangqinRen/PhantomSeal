from dataclasses import dataclass
from pathlib import Path
import collections
import collections.abc
import sys
import types
from typing import Any

import onnx
from onnx import numpy_helper
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.common_utils import suppress_third_party_noise, use_project


@dataclass
class ContextState:
    detected: bool
    rect: np.ndarray | None = None
    marker_landmarks: np.ndarray | None = None
    landmarks: np.ndarray | None = None
    align_mat: np.ndarray | None = None
    aligned_face: np.ndarray | None = None


class _YoloV5FaceTorch(nn.Module):
    """Tiny differentiable runner for DeepFaceLive's YoloV5Face.onnx graph."""

    def __init__(self, onnx_path: Path):
        super().__init__()
        model = onnx.load(str(onnx_path))
        self.nodes = list(model.graph.node)
        self.input_name = model.graph.input[0].name
        self.output_names = [output.name for output in model.graph.output]

        initializers = {}
        for initializer in model.graph.initializer:
            value = torch.from_numpy(numpy_helper.to_array(initializer).copy())
            self.register_buffer(self._buffer_name(initializer.name), value)
            initializers[initializer.name] = self._buffer_name(initializer.name)
        self.initializers = initializers

    @staticmethod
    def _buffer_name(name: str) -> str:
        return "_onnx_" + "".join(ch if ch.isalnum() else "_" for ch in name)

    @staticmethod
    def _attrs(node) -> dict[str, Any]:
        return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}

    def _initializer(self, name: str) -> Tensor:
        return getattr(self, self.initializers[name])

    def forward(self, x: Tensor) -> list[Tensor]:
        values: dict[str, Tensor] = {self.input_name: x}
        for name in self.initializers:
            values[name] = self._initializer(name).to(device=x.device, dtype=x.dtype)

        for node in self.nodes:
            attrs = self._attrs(node)
            inputs = [values[name] for name in node.input if name]

            if node.op_type == "Conv":
                pads = attrs.get("pads", [0, 0, 0, 0])
                value = F.conv2d(
                    inputs[0],
                    inputs[1],
                    inputs[2] if len(inputs) > 2 else None,
                    stride=tuple(attrs.get("strides", [1, 1])),
                    padding=(pads[0], pads[1]),
                    dilation=tuple(attrs.get("dilations", [1, 1])),
                    groups=int(attrs.get("group", 1)),
                )
            elif node.op_type == "Sigmoid":
                value = torch.sigmoid(inputs[0])
            elif node.op_type == "Mul":
                value = inputs[0] * inputs[1]
            elif node.op_type == "Add":
                value = inputs[0] + inputs[1]
            elif node.op_type == "Concat":
                value = torch.cat(inputs, dim=int(attrs["axis"]))
            elif node.op_type == "MaxPool":
                pads = attrs.get("pads", [0, 0, 0, 0])
                value = F.max_pool2d(
                    inputs[0],
                    kernel_size=tuple(attrs["kernel_shape"]),
                    stride=tuple(attrs.get("strides", attrs["kernel_shape"])),
                    padding=(pads[0], pads[1]),
                    ceil_mode=bool(attrs.get("ceil_mode", 0)),
                )
            elif node.op_type == "Resize":
                scales = inputs[-1]
                value = F.interpolate(
                    inputs[0],
                    scale_factor=(float(scales[-2]), float(scales[-1])),
                    mode="nearest",
                )
            elif node.op_type == "Constant":
                tensor = attrs.get("value")
                if tensor is None:
                    value = x.new_tensor(0.0)
                else:
                    value = torch.from_numpy(numpy_helper.to_array(tensor).copy()).to(
                        device=x.device, dtype=x.dtype
                    )
            else:
                raise NotImplementedError(f"Unsupported YoloV5Face ONNX op: {node.op_type}")

            values[node.output[0]] = value

        return [values[name] for name in self.output_names]


class Base:
    """
    DeepFaceLive detector/FaceMesh/align defense.

    The source identity in DeepFaceLive is selected by a DFM/Insight model, so
    PhantomSeal targets the target-side geometry context instead. By default it
    first suppresses the same YOLOv5Face detector used by DeepFaceLive with a
    differentiable Torch runner, then optionally falls back to the real
    detector -> Google FaceMesh -> aligner chain for black-box refinement.
    """

    def __init__(self, logger, config):
        super().__init__()
        self.logger = logger
        self.config = config
        self.device = self._select_torch_device()

        defense_config = config.third_party.defense
        weight_config = getattr(defense_config, "weight", {})

        self.perturb_level = self._cfg(defense_config, "perturb_level", 0.03, float)
        self.max_perturb_level = self._cfg(
            defense_config, "max_perturb_level", self.perturb_level, float
        )
        self.perturb_expand = max(
            1.0, self._cfg(defense_config, "perturb_expand", 1.4, float)
        )
        self.steps = self._cfg(defense_config, "steps", 80, int)
        self.step_scale = self._cfg(defense_config, "step_scale", 0.35, float)
        self.attack_mode = self._cfg(defense_config, "attack_mode", "yolo", str)
        self.yolo_pgd_steps = self._cfg(defense_config, "yolo_pgd_steps", self.steps, int)
        self.yolo_step_size = self._cfg(defense_config, "yolo_step_size", 1.0 / 255.0, float)
        self.yolo_topk = max(1, self._cfg(defense_config, "yolo_topk", 256, int))
        self.yolo_success_score = self._cfg(
            defense_config, "yolo_success_score", 0.02, float
        )
        self.yolo_loss_topk = max(1, self._cfg(defense_config, "yolo_loss_topk", 64, int))
        self.yolo_restarts = max(1, self._cfg(defense_config, "yolo_restarts", 3, int))
        self.yolo_min_steps = max(1, self._cfg(defense_config, "yolo_min_steps", 20, int))
        self.yolo_verify_interval = max(0, self._cfg(defense_config, "yolo_verify_interval", 20, int))
        self.yolo_quantize = self._cfg(defense_config, "yolo_quantize", True, bool)
        self.yolo_eot = self._cfg(defense_config, "yolo_eot", True, bool)
        self.yolo_eot_shifts = self._cfg(
            defense_config, "yolo_eot_shifts", [[0.0, 0.0], [0.02, 0.0], [-0.02, 0.0], [0.0, 0.02], [0.0, -0.02]]
        )
        self.yolo_eot_scales = self._cfg(defense_config, "yolo_eot_scales", [1.0, 0.96, 1.04])
        self.yolo_mask_mode = self._cfg(defense_config, "yolo_mask_mode", "full", str)
        self.yolo_mask_expand = self._cfg(defense_config, "yolo_mask_expand", 1.8, float)
        self.use_blackbox_refine = self._cfg(
            defense_config, "use_blackbox_refine", False, bool
        )
        self.max_blackbox_rounds = max(
            1, self._cfg(defense_config, "max_blackbox_rounds", 4, int)
        )
        self.require_undetected = self._cfg(
            defense_config, "require_undetected", False, bool
        )
        self.marker_success_score = self._cfg(
            defense_config, "marker_success_score", 1.0, float
        )
        self.aligned_success_score = self._cfg(
            defense_config, "aligned_success_score", 0.060, float
        )
        target_mse_255 = self._cfg(
            defense_config, "target_mse_255", [25.0, 50.0, 100.0]
        )
        if isinstance(target_mse_255, (int, float)):
            target_mse_255 = [target_mse_255]
        self.target_mse_255 = [float(value) for value in target_mse_255]

        self.population = self._cfg(defense_config, "population", 4, int)
        self.noise_sigma = self._cfg(defense_config, "noise_sigma", 0.01, float)
        self.marker_mask_sigma = self._cfg(
            defense_config, "marker_mask_sigma", 0.030, float
        )
        self.marker_rect_weight = self._cfg(
            defense_config, "marker_rect_weight", 0.55, float
        )
        self.detector_threshold = self._cfg(
            defense_config, "detector_threshold", 0.5, float
        )
        self.fixed_window_size = self._cfg(
            defense_config, "fixed_window_size", 480, int
        )
        self.marker_coverage = self._cfg(defense_config, "marker_coverage", 1.4, float)
        self.aligner_coverage = self._cfg(
            defense_config, "aligner_coverage", 2.2, float
        )
        self.aligner_resolution = self._cfg(
            defense_config, "aligner_resolution", 224, int
        )
        self.exclude_moving_parts = self._cfg(
            defense_config, "exclude_moving_parts", True, bool
        )
        self.log_interval = max(1, self._cfg(defense_config, "log_interval", 20, int))
        self.quiet_third_party = self._cfg(
            defense_config, "quiet_third_party", True, bool
        )

        self.detector_weight = self._cfg(weight_config, "detector", 2.0, float)
        self.marker_weight = self._cfg(weight_config, "marker", 24.0, float)
        self.align_weight = self._cfg(weight_config, "align", 12.0, float)
        self.fidelity_weight = self._cfg(weight_config, "fidelity", 1.0, float)
        self.tv_weight = self._cfg(weight_config, "tv", 0.04, float)

        self.last_loss_summary: dict[str, float] = {}
        self.yolo_surrogate: _YoloV5FaceTorch | None = None
        self._load_deepfacelive_models()

    @staticmethod
    def _select_torch_device() -> torch.device:
        if torch.cuda.is_available():
            try:
                torch.empty(1, device="cuda") * 1
                torch.cuda.synchronize()
                return torch.device("cuda")
            except Exception:
                torch.cuda.empty_cache()
        return torch.device("cpu")

    @staticmethod
    def _cfg(cfg: Any, key: str, default: Any, cast=None):
        value = default
        if isinstance(cfg, dict):
            value = cfg.get(key, default)
        elif hasattr(cfg, key):
            value = getattr(cfg, key)
        return cast(value) if cast is not None else value

    @staticmethod
    def _total_variation(delta: Tensor) -> Tensor:
        tv_h = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
        tv_w = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
        return tv_h + tv_w

    def _make_landmark_mask(
        self, clean_state: ContextState, height: int, width: int, device
    ) -> Tensor:
        ys = torch.linspace(0, 1, height, device=device).view(1, 1, height, 1)
        xs = torch.linspace(0, 1, width, device=device).view(1, 1, 1, width)
        mask = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
        sigma = max(0.006, self.marker_mask_sigma)

        if clean_state.landmarks is not None:
            points = torch.as_tensor(
                clean_state.landmarks, dtype=torch.float32, device=device
            ).clamp(0, 1)
            for chunk in points.split(64):
                px = chunk[:, 0].view(-1, 1, 1, 1)
                py = chunk[:, 1].view(-1, 1, 1, 1)
                heat = torch.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma**2))
                mask = torch.maximum(mask, heat.max(dim=0, keepdim=True).values)

        if clean_state.rect is not None:
            rect = torch.as_tensor(clean_state.rect, dtype=torch.float32, device=device)
            l, r = rect[:, 0].min(), rect[:, 0].max()
            t, b = rect[:, 1].min(), rect[:, 1].max()
            cx = (l + r) * 0.5
            cy = (t + b) * 0.5
            rx = (r - l).clamp_min(1e-6) * 0.70
            ry = (b - t).clamp_min(1e-6) * 0.80
            face = (((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0).float()
            mask = torch.maximum(mask, self.marker_rect_weight * face)

        mask = mask.clamp(0, 1)
        return mask[0]

    def perturb_imgs(self, imgs: Tensor) -> Tensor:
        imgs = imgs.detach().to(self.device, dtype=torch.float32).clamp(0, 1)
        batch = imgs.size(0)
        clean_states = [self._get_context_state(img) for img in imgs]
        initially_detected = torch.tensor(
            [state.detected for state in clean_states],
            dtype=torch.bool,
            device=self.device,
        )

        best_imgs = None
        best_stats = None
        active = initially_detected.clone()
        if self.attack_mode.lower() in {"yolo", "yolov5", "detector"}:
            best_imgs, best_stats = self._yolo_pgd_attack(
                imgs, clean_states, initially_detected, self.perturb_level
            )
            active = self._refresh_active(
                best_imgs, clean_states, initially_detected, best_stats
            )
            if not self.use_blackbox_refine or not active.any():
                self.last_loss_summary = {
                    key: float(np.mean([stats[key] for stats in best_stats]))
                    for key in [
                        "total",
                        "detector",
                        "marker",
                        "align",
                        "aligned_face",
                        "fidelity",
                        "tv",
                        "detected",
                        "yolo_score",
                    ]
                }
                return best_imgs.detach()

        x_imgs = (imgs + torch.empty_like(imgs).uniform_(-1e-5, 1e-5)).clamp(0, 1)
        if best_imgs is None or best_stats is None:
            best_imgs = x_imgs.detach().clone()
            best_stats = []
            for idx, (img, state) in enumerate(zip(imgs, clean_states)):
                if state.detected:
                    _, stats = self._score_candidate(img, best_imgs[idx], state)
                else:
                    stats = self._empty_stats(total=0.0, detected=0.0, detector=1.0)
                best_stats.append(stats)
        else:
            x_imgs = best_imgs.detach().clone()

        budget = self.perturb_level
        self._log_batch(0, 0, budget, active, best_stats)

        for round_idx in range(self.max_blackbox_rounds):
            if not active.any():
                break

            for step in range(1, self.steps + 1):
                active_idx = torch.nonzero(active, as_tuple=False).flatten()
                if active_idx.numel() == 0:
                    break

                self._blackbox_update_batch(
                    imgs,
                    x_imgs,
                    best_imgs,
                    best_stats,
                    clean_states,
                    active,
                    active_idx,
                    budget,
                    step,
                )
                active = self._refresh_active(
                    best_imgs, clean_states, initially_detected, best_stats
                )

                if (
                    not self.config.third_party.defense.silent_perturb
                    and step % self.log_interval == 0
                ):
                    self._log_batch(round_idx, step, budget, active, best_stats)

                if not active.any():
                    break

            if active.any() and budget < self.max_perturb_level:
                budget = min(self.max_perturb_level, budget * self.perturb_expand)
                self.logger.info(
                    "DeepFaceLive marker target still active for %d/%d images; "
                    "expanding perturb budget to %.5f",
                    int(active.sum().item()),
                    batch,
                    budget,
                )
                x_imgs[active] = torch.max(
                    torch.min(x_imgs[active], imgs[active] + budget),
                    imgs[active] - budget,
                ).clamp(0, 1)
            elif active.any():
                self.logger.warning(
                    "DeepFaceLive marker target still active for %d/%d images after "
                    "%d rounds at budget %.5f",
                    int(active.sum().item()),
                    batch,
                    self.max_blackbox_rounds,
                    budget,
                )
                break

        final_stats = []
        for idx, (clean_img, best_img, state) in enumerate(
            zip(imgs, best_imgs, clean_states)
        ):
            if state.detected:
                _, stats = self._score_candidate(clean_img, best_img, state)
            else:
                stats = self._empty_stats(total=0.0, detected=0.0, detector=1.0)
            final_stats.append(stats)
            best_stats[idx] = stats

        self.last_loss_summary = {
            key: float(np.mean([stats[key] for stats in final_stats]))
            for key in ["total", "detector", "marker", "align", "aligned_face", "fidelity", "tv", "detected", "yolo_score"]
        }
        return best_imgs.detach()

    def _yolo_pgd_attack(
        self,
        clean_imgs: Tensor,
        clean_states: list[ContextState],
        initially_detected: Tensor,
        budget: float,
    ) -> tuple[Tensor, list[dict[str, float]]]:
        if self.yolo_surrogate is None:
            self.logger.warning(
                "DeepFaceLive YOLO surrogate is unavailable; falling back to black-box perturbation."
            )
            return clean_imgs.detach().clone(), [
                self._score_candidate(clean_img, clean_img, state)[1]
                if state.detected
                else self._empty_stats(total=0.0, detected=0.0, detector=1.0)
                for clean_img, state in zip(clean_imgs, clean_states)
            ]

        detected_idx = torch.nonzero(initially_detected, as_tuple=False).flatten()
        if detected_idx.numel() == 0:
            return clean_imgs.detach().clone(), [
                self._empty_stats(total=0.0, detected=0.0, detector=1.0)
                for _ in clean_states
            ]

        masks = self._make_yolo_attack_masks(clean_imgs, clean_states)
        best_imgs = self._maybe_quantize(clean_imgs).detach().clone()
        best_detector = self._yolo_detector_score(best_imgs).detach()
        best_stats = [
            self._score_candidate(clean_img, best_img, state)[1]
            if state.detected
            else self._empty_stats(total=0.0, detected=0.0, detector=1.0)
            for clean_img, best_img, state in zip(clean_imgs, best_imgs, clean_states)
        ]
        for stats, score in zip(best_stats, best_detector):
            stats["yolo_score"] = float(score.item())

        step_size = min(
            max(self.yolo_step_size, budget / max(1, self.yolo_pgd_steps) * 1.5),
            budget,
        )

        for restart in range(self.yolo_restarts):
            if restart == 0:
                adv = (clean_imgs + torch.empty_like(clean_imgs).uniform_(-1e-5, 1e-5)).clamp(0, 1)
            else:
                noise = torch.empty_like(clean_imgs).uniform_(-budget, budget) * masks
                adv = self._project(clean_imgs + noise, clean_imgs, budget)

            for step in range(1, self.yolo_pgd_steps + 1):
                adv = adv.detach().requires_grad_(True)
                attack_view = self._quantize_ste(adv) if self.yolo_quantize else adv
                scores = self._yolo_detector_score(attack_view)
                loss = self._yolo_detector_loss(attack_view, detected_idx)
                grad = torch.autograd.grad(loss, adv, only_inputs=True)[0]
                with torch.no_grad():
                    adv = adv - step_size * grad.sign() * masks
                    adv = self._project(adv, clean_imgs, budget)
                    eval_adv = self._maybe_quantize(adv)
                    new_scores = self._yolo_detector_score(eval_adv, robust=True).detach()
                    improved = new_scores < best_detector
                    best_detector[improved] = new_scores[improved]
                    best_imgs[improved] = eval_adv[improved]

                if self.yolo_verify_interval and step % self.yolo_verify_interval == 0:
                    self._update_best_with_real_detector(
                        clean_imgs,
                        eval_adv.detach(),
                        clean_states,
                        best_imgs,
                        best_stats,
                        best_detector,
                        new_scores,
                    )

                if (
                    not self.config.third_party.defense.silent_perturb
                    and step % self.log_interval == 0
                ):
                    active_count = int((best_detector[detected_idx] > self.yolo_success_score).sum().item())
                    true_active = sum(
                        1 for idx in detected_idx.tolist() if best_stats[idx]["detected"] > 0.0
                    )
                    self.logger.info(
                        "[YOLO PGD r%d/%d %4d/%4d] budget=%.5f, yolo=%.5f, surrogate_active=%d/%d, true_active=%d/%d",
                        restart + 1,
                        self.yolo_restarts,
                        step,
                        self.yolo_pgd_steps,
                        budget,
                        float(best_detector[detected_idx].mean().item()),
                        active_count,
                        int(detected_idx.numel()),
                        true_active,
                        int(detected_idx.numel()),
                    )

                if (
                    step >= self.yolo_min_steps
                    and bool((best_detector[detected_idx] <= self.yolo_success_score).all().item())
                    and all(best_stats[idx]["detected"] == 0.0 for idx in detected_idx.tolist())
                ):
                    break

            final_adv = self._maybe_quantize(adv.detach())
            final_scores = self._yolo_detector_score(final_adv, robust=True).detach()
            self._update_best_with_real_detector(
                clean_imgs,
                final_adv,
                clean_states,
                best_imgs,
                best_stats,
                best_detector,
                final_scores,
            )

            if (
                bool((best_detector[detected_idx] <= self.yolo_success_score).all().item())
                and all(best_stats[idx]["detected"] == 0.0 for idx in detected_idx.tolist())
            ):
                break

        for idx, (clean_img, best_img, clean_state) in enumerate(
            zip(clean_imgs, best_imgs, clean_states)
        ):
            if clean_state.detected and best_stats[idx]["detected"] > 0.0:
                _, stats = self._score_candidate(clean_img, best_img, clean_state)
                stats["yolo_score"] = float(best_detector[idx].item())
                stats["score"] = -float(best_detector[idx].item())
                stats["total"] = stats["score"]
                best_stats[idx] = stats
            elif clean_state.detected:
                best_stats[idx]["yolo_score"] = float(best_detector[idx].item())
                best_stats[idx]["score"] = -float(best_detector[idx].item())
                best_stats[idx]["total"] = best_stats[idx]["score"]
        return best_imgs.detach(), best_stats

    def _update_best_with_real_detector(
        self,
        clean_imgs: Tensor,
        candidates: Tensor,
        clean_states: list[ContextState],
        best_imgs: Tensor,
        best_stats: list[dict[str, float]],
        best_detector: Tensor,
        candidate_scores: Tensor,
    ) -> None:
        for idx, (clean_img, candidate, clean_state) in enumerate(
            zip(clean_imgs, candidates, clean_states)
        ):
            if not clean_state.detected:
                continue
            _, stats = self._score_candidate(clean_img, candidate, clean_state)
            stats["yolo_score"] = float(candidate_scores[idx].item())
            stats["score"] = -float(candidate_scores[idx].item())
            stats["total"] = stats["score"]
            if self._real_detector_candidate_is_better(stats, best_stats[idx]):
                best_stats[idx] = stats
                best_imgs[idx] = candidate.detach()
                best_detector[idx] = candidate_scores[idx]

    @staticmethod
    def _real_detector_candidate_is_better(
        candidate: dict[str, float], incumbent: dict[str, float]
    ) -> bool:
        candidate_success = candidate["detected"] == 0.0
        incumbent_success = incumbent["detected"] == 0.0
        if candidate_success and not incumbent_success:
            return True
        if incumbent_success and not candidate_success:
            return False
        if candidate_success and incumbent_success:
            candidate_yolo = candidate.get("yolo_score", 1.0)
            incumbent_yolo = incumbent.get("yolo_score", 1.0)
            if abs(candidate_yolo - incumbent_yolo) > 0.01:
                return candidate_yolo < incumbent_yolo
            return candidate["fidelity"] + 0.02 * candidate["tv"] < incumbent["fidelity"] + 0.02 * incumbent["tv"]
        return candidate.get("yolo_score", 1.0) < incumbent.get("yolo_score", 1.0)

    def _quantize_ste(self, imgs: Tensor) -> Tensor:
        if not self.yolo_quantize:
            return imgs
        quantized = torch.round(imgs.clamp(0, 1) * 255.0) / 255.0
        return imgs + (quantized - imgs).detach()

    def _maybe_quantize(self, imgs: Tensor) -> Tensor:
        if not self.yolo_quantize:
            return imgs.detach()
        return torch.round(imgs.detach().clamp(0, 1) * 255.0) / 255.0

    def _yolo_objectness_logits(self, imgs: Tensor) -> Tensor:
        feed = self._preprocess_yolo_fixed_window(imgs)
        outputs = self.yolo_surrogate(feed)
        logits = []
        for pred in outputs:
            n, channels, height, width = pred.shape
            anchors = channels // 16
            raw = pred.reshape(n, anchors, 16, height, width).permute(0, 1, 3, 4, 2)
            logits.append(raw[..., 4].reshape(n, -1))
        return torch.cat(logits, dim=1)

    def _yolo_detector_score(self, imgs: Tensor, robust: bool = False) -> Tensor:
        scores = []
        views = self._yolo_eot_views(imgs) if robust else [imgs]
        for view in views:
            logits = self._yolo_objectness_logits(view)
            k = min(self.yolo_topk, logits.shape[1])
            scores.append(torch.sigmoid(logits.topk(k=k, dim=1).values).amax(dim=1))
        return torch.stack(scores, dim=0).amax(dim=0)

    def _yolo_detector_loss(self, imgs: Tensor, detected_idx: Tensor) -> Tensor:
        losses = []
        views = self._yolo_eot_views(imgs) if self.yolo_eot else [imgs]
        for view in views:
            logits = self._yolo_objectness_logits(view)[detected_idx]
            k = min(self.yolo_loss_topk, logits.shape[1])
            top_logits = logits.topk(k=k, dim=1).values
            losses.append(torch.logsumexp(top_logits, dim=1))
        return torch.stack(losses, dim=0).amax(dim=0).mean()

    def _yolo_eot_views(self, imgs: Tensor) -> list[Tensor]:
        if not self.yolo_eot:
            return [imgs]
        views = []
        _, _, height, width = imgs.shape
        shifts = self.yolo_eot_shifts or [[0.0, 0.0]]
        scales = self.yolo_eot_scales or [1.0]
        for scale in scales:
            for shift in shifts:
                tx, ty = float(shift[0]), float(shift[1])
                if abs(float(scale) - 1.0) < 1e-8 and abs(tx) < 1e-8 and abs(ty) < 1e-8:
                    views.append(imgs)
                    continue
                theta = imgs.new_zeros((imgs.shape[0], 2, 3))
                theta[:, 0, 0] = float(scale)
                theta[:, 1, 1] = float(scale)
                theta[:, 0, 2] = tx
                theta[:, 1, 2] = ty
                grid = F.affine_grid(theta, imgs.shape, align_corners=False)
                views.append(F.grid_sample(imgs, grid, mode="bilinear", padding_mode="border", align_corners=False))
        return views

    def _make_yolo_attack_masks(
        self, clean_imgs: Tensor, clean_states: list[ContextState]
    ) -> Tensor:
        mode = self.yolo_mask_mode.lower()
        if mode == "full":
            return torch.ones_like(clean_imgs)

        masks = []
        _, _, height, width = clean_imgs.shape
        ys = torch.linspace(0, 1, height, device=clean_imgs.device).view(1, height, 1)
        xs = torch.linspace(0, 1, width, device=clean_imgs.device).view(1, 1, width)
        for clean_img, state in zip(clean_imgs, clean_states):
            if not state.detected or state.rect is None:
                masks.append(torch.zeros_like(clean_img))
                continue
            rect = torch.as_tensor(state.rect, dtype=torch.float32, device=clean_imgs.device)
            l, r = rect[:, 0].min(), rect[:, 0].max()
            t, b = rect[:, 1].min(), rect[:, 1].max()
            cx, cy = (l + r) * 0.5, (t + b) * 0.5
            half_w = (r - l).clamp_min(1e-6) * 0.5 * self.yolo_mask_expand
            half_h = (b - t).clamp_min(1e-6) * 0.5 * self.yolo_mask_expand
            if mode in {"box", "face_box"}:
                mask = ((xs >= cx - half_w) & (xs <= cx + half_w) & (ys >= cy - half_h) & (ys <= cy + half_h)).float()
            else:
                mask = ((((xs - cx) / half_w) ** 2 + ((ys - cy) / half_h) ** 2) <= 1.0).float()
            masks.append(mask.expand_as(clean_img))
        return torch.stack(masks, dim=0).clamp(0, 1)

    def _preprocess_yolo_fixed_window(self, imgs: Tensor) -> Tensor:
        _, _, height, width = imgs.shape
        target = self.fixed_window_size
        if target and target > 0:
            scale = min(target / width, target / height)
            if scale > 1.0:
                scale = 1.0
            if scale != 1.0:
                new_h = max(1, int(height * scale))
                new_w = max(1, int(width * scale))
                imgs = F.interpolate(
                    imgs, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
                height, width = new_h, new_w
            pad_h = max(0, target - height)
            pad_w = max(0, target - width)
            if pad_h or pad_w:
                imgs = F.pad(imgs, (0, pad_w, 0, pad_h))
        else:
            pad_h = (64 - height % 64) % 64
            pad_w = (64 - width % 64) % 64
            if pad_h or pad_w:
                imgs = F.pad(imgs, (0, pad_w, 0, pad_h))
        return imgs

    def _blackbox_update_batch(
        self,
        clean_imgs: Tensor,
        x_imgs: Tensor,
        best_imgs: Tensor,
        best_stats: list[dict[str, float]],
        clean_states: list[ContextState],
        active: Tensor,
        active_idx: Tensor,
        budget: float,
        step: int,
    ) -> None:
        for tensor_idx in active_idx.tolist():
            if not bool(active[tensor_idx].item()):
                continue

            for candidate in self._candidate_batch(
                x_imgs[tensor_idx],
                clean_imgs[tensor_idx],
                clean_states[tensor_idx],
                budget,
                step,
            ):
                score, stats = self._score_candidate(
                    clean_imgs[tensor_idx],
                    candidate,
                    clean_states[tensor_idx],
                )
                if self._candidate_is_better(stats, best_stats[tensor_idx]):
                    best_stats[tensor_idx] = stats
                    best_imgs[tensor_idx] = candidate.detach()
                    x_imgs[tensor_idx] = candidate.detach()
                if stats["detected"] == 0.0:
                    break

    def _candidate_is_better(
        self, candidate: dict[str, float], incumbent: dict[str, float]
    ) -> bool:
        candidate_success = (
            candidate["detected"] == 0.0
            or candidate["aligned_face"] >= self.aligned_success_score
        )
        incumbent_success = (
            incumbent["detected"] == 0.0
            or incumbent["aligned_face"] >= self.aligned_success_score
        )
        if candidate_success and not incumbent_success:
            return True
        if incumbent_success and not candidate_success:
            return False
        if candidate_success and incumbent_success:
            candidate_noise = candidate["fidelity"] + 0.02 * candidate["tv"]
            incumbent_noise = incumbent["fidelity"] + 0.02 * incumbent["tv"]
            return candidate_noise < incumbent_noise
        return candidate["score"] > incumbent["score"]

    def _refresh_active(
        self,
        best_imgs: Tensor,
        clean_states: list[ContextState],
        initially_detected: Tensor,
        best_stats: list[dict[str, float]] | None = None,
    ) -> Tensor:
        active_values = []
        if best_stats is None:
            best_stats = []
            for best_img, clean_state in zip(best_imgs, clean_states):
                _, stats = self._score_candidate(best_img, best_img, clean_state)
                best_stats.append(stats)

        for clean_state, was_detected, stats in zip(
            clean_states, initially_detected, best_stats
        ):
            if not bool(was_detected.item()) or not clean_state.detected:
                active_values.append(False)
                continue
            if self.require_undetected:
                active_values.append(stats["detected"] > 0.0)
            else:
                active_values.append(stats["aligned_face"] < self.aligned_success_score)
        return torch.tensor(active_values, dtype=torch.bool, device=self.device)

    def _candidate_batch(
        self,
        x_img: Tensor,
        clean_img: Tensor,
        clean_state: ContextState,
        budget: float,
        step: int,
    ) -> list[Tensor]:
        candidates = [x_img.detach()]
        context_mask = self._make_landmark_mask(
            clean_state, clean_img.shape[1], clean_img.shape[2], clean_img.device
        )
        decay = max(0.25, 1.0 - (step - 1) / max(1, self.steps))
        local_scale = max(0.10, self.step_scale * decay)

        for scale in (0.20, 0.35, 0.55, 0.80, 1.10):
            noise = torch.sign(torch.randn_like(x_img)) * context_mask
            candidates.append(
                self._project(
                    x_img + budget * scale * local_scale * noise,
                    clean_img,
                    budget,
                )
            )
            candidates.append(
                self._project(
                    clean_img + budget * scale * local_scale * noise,
                    clean_img,
                    budget,
                )
            )

        targets = self.target_mse_255 or [100.0]
        for idx in range(max(0, self.population)):
            target_mse = targets[idx % len(targets)]
            if idx % 2 == 0:
                noise = torch.randn_like(x_img) * context_mask
            else:
                noise = torch.sign(torch.randn_like(x_img)) * context_mask
            candidates.append(
                self._project_to_target_mse(noise, clean_img, budget, target_mse)
            )
        return candidates

    @staticmethod
    def _project(candidate: Tensor, clean_img: Tensor, budget: float) -> Tensor:
        candidate = torch.max(
            torch.min(candidate, clean_img + budget),
            clean_img - budget,
        )
        return candidate.clamp(0, 1).detach()

    def _project_to_target_mse(
        self, noise: Tensor, clean_img: Tensor, budget: float, target_mse_255: float
    ) -> Tensor:
        target_mse = max(0.0, float(target_mse_255)) / (255.0 * 255.0)
        current_mse = (noise**2).mean().clamp_min(1e-12)
        scale = torch.sqrt(noise.new_tensor(target_mse) / current_mse)
        return self._project(clean_img + noise * scale, clean_img, budget)

    def _score_candidate(
        self,
        clean_img: Tensor,
        candidate_img: Tensor,
        clean_state: ContextState,
    ) -> tuple[float, dict[str, float]]:
        pert_state = self._get_context_state(candidate_img)
        (
            rect_shift,
            marker_shift,
            landmark_shift,
            align_shift,
            aligned_face_shift,
        ) = self._state_distance(clean_state, pert_state)
        detected = float(pert_state.detected)
        detector_score = 1.0 if not pert_state.detected else rect_shift

        delta = candidate_img - clean_img
        fidelity = float((delta**2).mean().item())
        tv = float(self._total_variation(delta.unsqueeze(0))[0].item())
        score = (
            5.0 * detector_score
            + 180.0 * marker_shift
            + 120.0 * landmark_shift
            + 70.0 * align_shift
            + 260.0 * aligned_face_shift
            - 8.0 * fidelity
            - 0.5 * tv
        )
        stats = {
            "score": float(score),
            "total": float(score),
            "detector": float(detector_score),
            "marker": float(marker_shift),
            "align": float(align_shift),
            "aligned_face": float(aligned_face_shift),
            "fidelity": fidelity,
            "tv": tv,
            "detected": detected,
            "yolo_score": 0.0,
        }
        return float(score), stats

    @staticmethod
    def _state_distance(
        clean: ContextState, pert: ContextState
    ) -> tuple[float, float, float, float, float]:
        if not pert.detected:
            return 1.0, 1.0, 1.0, 1.0, 1.0

        rect_shift = 0.0
        if clean.rect is not None and pert.rect is not None:
            rect_shift = float(np.linalg.norm(clean.rect - pert.rect, axis=1).mean())

        marker_shift = 0.0
        if clean.marker_landmarks is not None and pert.marker_landmarks is not None:
            count = min(len(clean.marker_landmarks), len(pert.marker_landmarks))
            marker_shift = float(
                np.linalg.norm(
                    clean.marker_landmarks[:count] - pert.marker_landmarks[:count],
                    axis=1,
                ).mean()
            )

        landmark_shift = 0.0
        if clean.landmarks is not None and pert.landmarks is not None:
            count = min(len(clean.landmarks), len(pert.landmarks))
            landmark_shift = float(
                np.linalg.norm(clean.landmarks[:count] - pert.landmarks[:count], axis=1).mean()
            )

        align_shift = 0.0
        if clean.align_mat is not None and pert.align_mat is not None:
            align_shift = float(np.abs(clean.align_mat - pert.align_mat).mean())

        aligned_face_shift = 0.0
        if clean.aligned_face is not None and pert.aligned_face is not None:
            aligned_face_shift = float(
                np.mean(np.abs(clean.aligned_face - pert.aligned_face)) / 255.0
            )

        return rect_shift, marker_shift, landmark_shift, align_shift, aligned_face_shift

    def _get_context_state(self, img: Tensor) -> ContextState:
        image = self._tensor_to_uint8_hwc(img)
        height, width = image.shape[:2]

        try:
            with suppress_third_party_noise(self.quiet_third_party):
                rects = self.face_detector.extract(
                    image,
                    threshold=self.detector_threshold,
                    fixed_window=self.fixed_window_size,
                )[0]
        except Exception as exc:
            self.logger.debug("DeepFaceLive detector failed: %s", exc)
            return ContextState(detected=False)

        if len(rects) == 0:
            return ContextState(detected=False)

        u_rects = [
            self._FRect.from_ltrb((l / width, t / height, r / width, b / height))
            for l, t, r, b in rects
        ]
        face_urect = self._FRect.sort_by_area_size(u_rects)[0]
        rect_np = face_urect.as_4pts().astype(np.float32)

        try:
            face_image, face_uni_mat = face_urect.cut(image, self.marker_coverage, 192)
            lmrks = self.face_marker.extract(face_image)[0]
            marker_landmarks = (lmrks[..., 0:2] / (192, 192)).astype(np.float32)
            face_ulmrks = self._FLandmarks2D.create(
                self._ELandmarks2D.L468, marker_landmarks
            ).transform(face_uni_mat, invert=True)
            landmarks = face_ulmrks.as_numpy().astype(np.float32)
            aligned_face, align_uni_mat = face_ulmrks.cut(
                image,
                self.aligner_coverage,
                self.aligner_resolution,
                exclude_moving_parts=self.exclude_moving_parts,
                y_offset=-0.08,
            )
            align_mat = np.asarray(align_uni_mat, dtype=np.float32)
        except Exception as exc:
            self.logger.debug("DeepFaceLive marker/align failed: %s", exc)
            return ContextState(detected=False, rect=rect_np)

        return ContextState(
            detected=True,
            rect=rect_np,
            marker_landmarks=marker_landmarks,
            landmarks=landmarks,
            align_mat=align_mat,
            aligned_face=aligned_face.astype(np.float32),
        )

    def _load_deepfacelive_models(self) -> None:
        root_dir = Path(self.config.third_party.project_root)
        if not hasattr(collections, "Iterable"):
            collections.Iterable = collections.abc.Iterable
        if "numexpr" not in sys.modules:
            try:
                __import__("numexpr")
            except ModuleNotFoundError:
                shim = types.ModuleType("numexpr")

                def evaluate(expr, local_dict=None, global_dict=None, **kwargs):
                    scope = {}
                    if global_dict:
                        scope.update(global_dict)
                    if local_dict:
                        scope.update(local_dict)
                    return eval(expr, {"__builtins__": {}}, scope)

                shim.evaluate = evaluate
                sys.modules["numexpr"] = shim

        with suppress_third_party_noise(self.quiet_third_party):
            with use_project([root_dir], purge_prefixes=("modelhub", "xlib")):
                try:
                    from modelhub.onnx.FaceMesh.FaceMesh import FaceMesh
                    from modelhub.onnx.YoloV5Face.YoloV5Face import YoloV5Face
                    from xlib.face import ELandmarks2D, FLandmarks2D, FRect
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        "DeepFaceLive context perturb requires third_party/DeepFaceLive "
                        f"dependencies, but '{exc.name}' is missing."
                    ) from exc

                devices = YoloV5Face.get_available_devices()
                if len(devices) == 0:
                    raise RuntimeError("DeepFaceLive ONNXRuntime has no available devices")

                device = devices[0]
                self.face_detector = YoloV5Face(device)
                self.face_marker = FaceMesh(device)
                yolo_path = root_dir / "modelhub" / "onnx" / "YoloV5Face" / "YoloV5Face.onnx"
                self.yolo_surrogate = _YoloV5FaceTorch(yolo_path).to(self.device).eval()
                for param in self.yolo_surrogate.parameters():
                    param.requires_grad_(False)
                self._ELandmarks2D = ELandmarks2D
                self._FLandmarks2D = FLandmarks2D
                self._FRect = FRect

    @staticmethod
    def _tensor_to_uint8_hwc(img: Tensor) -> np.ndarray:
        return (
            img.detach()
            .clamp(0, 1)
            .permute(1, 2, 0)
            .mul(255)
            .round()
            .byte()
            .cpu()
            .numpy()
        )

    @staticmethod
    def _empty_stats(
        total: float = 0.0,
        detected: float = 0.0,
        detector: float = 0.0,
    ) -> dict[str, float]:
        return {
            "score": total,
            "total": total,
            "detector": detector,
            "marker": 0.0,
            "align": 0.0,
            "aligned_face": 0.0,
            "fidelity": 0.0,
            "tv": 0.0,
            "detected": detected,
            "yolo_score": 0.0,
        }

    def _log_stats(self, step: int, stats: dict[str, float], prefix: str = "") -> None:
        if self.config.third_party.defense.silent_perturb:
            return
        tag = f"{prefix} " if prefix else ""
        self.logger.info(
            f"{tag}[Step {step:4}/{self.steps:4}] "
            f"score={stats['score']:.5f}, "
            f"detector={stats['detector']:.5f}, "
            f"marker={stats['marker']:.5f}, "
            f"align={stats['align']:.5f}, "
            f"aligned_face={stats['aligned_face']:.5f}, "
            f"fidelity={stats['fidelity']:.5f}, "
            f"tv={stats['tv']:.5f}, "
            f"detected={stats['detected']:.0f}, "
            f"yolo={stats.get('yolo_score', 0.0):.5f}"
        )

    def _log_batch(
        self,
        round_idx: int,
        step: int,
        budget: float,
        active: Tensor,
        stats_list: list[dict[str, float]],
    ) -> None:
        if self.config.third_party.defense.silent_perturb:
            return
        detected = sum(1 for stats in stats_list if stats["detected"] > 0.0)
        total = len(stats_list)
        mean_score = float(np.mean([stats["score"] for stats in stats_list]))
        mean_marker = float(np.mean([stats["marker"] for stats in stats_list]))
        mean_align = float(np.mean([stats["align"] for stats in stats_list]))
        self.logger.info(
            "[Round %2d/%2d Step %4d/%4d] budget=%.5f, "
            "blackbox_detected=%d/%d, active=%d/%d, "
            "score=%.5f, marker=%.5f, align=%.5f",
            round_idx + 1,
            self.max_blackbox_rounds,
            step,
            self.steps,
            budget,
            detected,
            total,
            int(active.sum().item()),
            total,
            mean_score,
            mean_marker,
            mean_align,
        )

    @staticmethod
    def _free_gpu() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
