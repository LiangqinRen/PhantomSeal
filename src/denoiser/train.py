import argparse
import textwrap
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from src.common_utils import save_tensor_imgs

from .dataset import PairedImageDataset
from .model import SimpleDenoiserUNet


def to_image_range(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple perturb->clean denoiser")
    parser.add_argument("--data-root", required=True, help="Dataset root with clean/ and perturb/")
    parser.add_argument("--out-dir", default="runs/denoiser", help="Output directory")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def make_loaders(args: argparse.Namespace):
    dataset = PairedImageDataset(args.data_root, image_size=args.image_size)
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = None
    if val_size > 0:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    return train_loader, val_loader


def make_loaders_from_config(config: Any):
    dataset_config = config.third_party.dataset
    defense_config = config.third_party.defense
    train_set = PairedImageDataset(
        dataset_config.train_dir,
        image_size=dataset_config.image_size,
    )
    max_train_images = int(dataset_config.max_train_images)
    if max_train_images < 1 or max_train_images > len(train_set):
        raise ValueError(
            f"third_party.dataset.max_train_images must be in [1, {len(train_set)}], "
            f"got {max_train_images}"
        )
    train_set = Subset(train_set, range(max_train_images))

    val_set = PairedImageDataset(
        dataset_config.validate_dir,
        image_size=dataset_config.image_size,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=defense_config.batch_size,
        shuffle=True,
        num_workers=defense_config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=defense_config.batch_size,
        shuffle=False,
        num_workers=defense_config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    sample_dir: Path,
    epoch: int,
) -> float | None:
    if loader is None:
        return None

    model.eval()
    total_loss = 0.0
    sample_saved = False
    with torch.no_grad():
        for perturb, clean in loader:
            perturb = perturb.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            pred = model(perturb)
            loss = F.l1_loss(pred, clean)
            total_loss += loss.item() * perturb.size(0)

            if not sample_saved:
                n = min(4, perturb.size(0))
                save_tensor_imgs(
                    sample_dir,
                    f"epoch_{epoch:04d}",
                    ["perturb", "prediction", "clean"],
                    [
                        to_image_range(perturb[:n]),
                        to_image_range(pred[:n]),
                        to_image_range(clean[:n]),
                    ],
                    image_name="denoiser",
                    only_save_summary=True,
                )
                sample_saved = True

    return total_loss / len(loader.dataset)


def _format_effectiveness_with_names(effectiveness: dict | None) -> str:
    if effectiveness is None:
        return "()"

    def format_rate(value: tuple) -> str:
        if value[1] == 0:
            return "nan/0"
        return f"{value[0] / value[1] * 100:.3f}/{value[1]:.0f}"

    parts = []
    for verifier, values in effectiveness.items():
        metrics = ", ".join(
            f"{metric_name}={format_rate(metric_value)}"
            for metric_name, metric_value in values.items()
        )
        parts.append(f"{verifier}({metrics})")
    return " ".join(parts)


def _limit_dataset(dataset: Any, max_images: Any, config_name: str) -> Any:
    if max_images is None:
        return dataset

    max_images = int(max_images)
    if max_images < 1 or max_images > len(dataset):
        raise ValueError(
            f"{config_name} must be in [1, {len(dataset)}], got {max_images}"
        )
    return Subset(dataset, range(max_images))


@torch.no_grad()
def evaluate_internal_simswap(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Any,
    logger: Any,
    simswap: Any,
    effectiveness: Any,
    epoch: int,
    label: str = "Internal",
    eval_config: Any | None = None,
) -> None:
    import src.metric as metric
    from src.common_utils import save_tensor_imgs

    internal_config = eval_config or config.third_party.internal
    logger.info(
        f"Start denoiser {label.lower()} SimSwap evaluation: "
        f"samples={len(loader.dataset)}, batches={len(loader)}, "
        f"batch_size={loader.batch_size}"
    )
    model.eval()
    totals = {
        "clean/original": {"source": None, "target": None},
        "perturb": {"source": None, "target": None},
        "denoised perturb": {"source": None, "target": None},
    }
    total_count = 0
    sample_saved = False

    def merge_total(name: str, side: str, item: dict) -> None:
        if totals[name][side] is None:
            totals[name][side] = item
        else:
            metric.merge_single_dict(totals[name][side], item)

    def format_metric_block(metrics: dict) -> str:
        return textwrap.dedent(
            f"""
            clean/original source: {_format_effectiveness_with_names(metrics['clean/original']['source'])}
            clean/original target: {_format_effectiveness_with_names(metrics['clean/original']['target'])}
            perturb source: {_format_effectiveness_with_names(metrics['perturb']['source'])}
            perturb target: {_format_effectiveness_with_names(metrics['perturb']['target'])}
            denoised perturb source: {_format_effectiveness_with_names(metrics['denoised perturb']['source'])}
            denoised perturb target: {_format_effectiveness_with_names(metrics['denoised perturb']['target'])}
            """
        )

    def evaluate_candidate(
        name: str,
        candidate: torch.Tensor,
        source_swap: torch.Tensor,
        target_swap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
        candidate_source_swap = (
            source_swap if candidate is clean_01 else simswap.swap_face(candidate, target_01)
        )
        candidate_target_swap = (
            target_swap if candidate is clean_01 else simswap.swap_face(target_01, candidate)
        )
        source_effectiveness = effectiveness.calculate_effectiveness(
            clean_01,
            candidate,
            source_swap,
            candidate_source_swap,
            cloak,
        )
        target_effectiveness = effectiveness.calculate_effectiveness(
            target_01,
            None,
            target_swap,
            candidate_target_swap,
            None,
        )
        merge_total(name, "source", source_effectiveness)
        merge_total(name, "target", target_effectiveness)
        return (
            candidate_source_swap,
            candidate_target_swap,
            source_effectiveness,
            target_effectiveness,
        )

    for batch_idx, batch in enumerate(loader, start=1):
        if len(batch) == 4:
            perturb, clean, cloak, target = batch
            cloak = torch.clamp(
                (cloak.to(device, non_blocking=True) + 1.0) / 2.0,
                0.0,
                1.0,
            )
            target = target.to(device, non_blocking=True)
        elif len(batch) == 3:
            perturb, clean, target = batch
            target = target.to(device, non_blocking=True)
            cloak = None
        else:
            perturb, clean = batch
            target = clean
            cloak = None

        perturb = perturb.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        total_count += perturb.size(0)

        denoised = to_image_range(model(perturb))
        perturb_01 = to_image_range(perturb)
        clean_01 = to_image_range(clean)
        target_01 = to_image_range(target)

        source_swap = simswap.swap_face(clean_01, target_01)
        target_swap = simswap.swap_face(target_01, clean_01)
        batch_metrics = {
            "clean/original": {"source": None, "target": None},
            "perturb": {"source": None, "target": None},
            "denoised perturb": {"source": None, "target": None},
        }
        (
            clean_source_swap,
            clean_target_swap,
            batch_metrics["clean/original"]["source"],
            batch_metrics["clean/original"]["target"],
        ) = evaluate_candidate(
            "clean/original",
            clean_01,
            source_swap,
            target_swap,
        )
        (
            perturb_source_swap,
            perturb_target_swap,
            batch_metrics["perturb"]["source"],
            batch_metrics["perturb"]["target"],
        ) = evaluate_candidate(
            "perturb",
            perturb_01,
            source_swap,
            target_swap,
        )
        (
            denoised_source_swap,
            denoised_target_swap,
            batch_metrics["denoised perturb"]["source"],
            batch_metrics["denoised perturb"]["target"],
        ) = evaluate_candidate(
            "denoised perturb",
            denoised,
            source_swap,
            target_swap,
        )

        if internal_config.save_images and not sample_saved:
            labels = [
                "clean",
                "perturb",
                "denoised",
                "target",
                "clean_source_simswap",
                "perturb_source_simswap",
                "denoised_source_simswap",
                "clean_target_simswap",
                "perturb_target_simswap",
                "denoised_target_simswap",
            ]
            tensors = [
                clean_01,
                perturb_01,
                denoised,
                target_01,
                clean_source_swap,
                perturb_source_swap,
                denoised_source_swap,
                clean_target_swap,
                perturb_target_swap,
                denoised_target_swap,
            ]
            if cloak is not None:
                labels.append("cloak")
                tensors.append(cloak)
            save_tensor_imgs(
                Path(config.image_dir),
                f"internal_epoch_{epoch:04d}",
                labels,
                tensors,
                only_save_summary=internal_config.only_save_summary,
            )
            sample_saved = True

        del (
            perturb,
            clean,
            perturb_01,
            clean_01,
            target_01,
            denoised,
            source_swap,
            target_swap,
            clean_source_swap,
            perturb_source_swap,
            denoised_source_swap,
            clean_target_swap,
            perturb_target_swap,
            denoised_target_swap,
            target,
        )
        torch.cuda.empty_cache()

        iter_log_str = textwrap.dedent(
            f"""
            [Denoiser {label} SimSwap][Epoch {epoch:4}][Batch {batch_idx:4}/{len(loader):4}]
            enabled metrics are controlled by evaluate.effectiveness
            effectiveness format: verifier(metric=percent/total, ...)
            current batch:
            {textwrap.indent(format_metric_block(batch_metrics), "    ")}
            """
        )
        summary_log_str = textwrap.dedent(
            f"""
            [Denoiser {label} SimSwap Summary][Epoch {epoch:4}][Batch {batch_idx:4}/{len(loader):4}][Validate {total_count:4}]
            cumulative:
            {textwrap.indent(format_metric_block(totals), "    ")}
            """
        )
        logger.info(textwrap.indent(iter_log_str, "    "))
        logger.info(textwrap.indent(summary_log_str, "    "))

    if total_count == 0:
        logger.warning("Skip denoiser internal SimSwap validation: no samples evaluated")
        return

    log_str = textwrap.dedent(
        f"""
        [Denoiser {label} SimSwap][Epoch {epoch:4}][Validate {total_count:4}]
        enabled metrics are controlled by evaluate.effectiveness
        effectiveness format: verifier(metric=percent/total, ...)
        clean/original source: {_format_effectiveness_with_names(totals['clean/original']['source'])}
        clean/original target: {_format_effectiveness_with_names(totals['clean/original']['target'])}
        perturb source: {_format_effectiveness_with_names(totals['perturb']['source'])}
        perturb target: {_format_effectiveness_with_names(totals['perturb']['target'])}
        denoised perturb source: {_format_effectiveness_with_names(totals['denoised perturb']['source'])}
        denoised perturb target: {_format_effectiveness_with_names(totals['denoised perturb']['target'])}
        """
    )
    logger.info(textwrap.indent(log_str, "    "))


def train_with_config(config: Any, logger: Any) -> None:
    sample_dir = Path(config.image_dir)
    ckpt_dir = Path(config.log_dir) / "checkpoints"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    train_loader, val_loader = make_loaders_from_config(config)
    model = SimpleDenoiserUNet(
        base_channels=config.third_party.model.base_channels,
        depth=config.third_party.model.depth,
        residual=config.third_party.model.residual,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.third_party.defense.lr,
        weight_decay=config.third_party.defense.weight_decay,
    )
    internal_dataset = PairedImageDataset(
        config.third_party.dataset.validate_dir,
        image_size=config.third_party.dataset.image_size,
        cloak_dir=config.third_party.internal.cloak_dir,
        include_cloak=True,
        target_dir=config.third_party.internal.target_dir,
        include_target=True,
    )
    if internal_dataset.include_cloak:
        logger.info(f"Use denoiser internal cloak dir: {internal_dataset.cloak_dir}")
    else:
        logger.info(
            f"Denoiser internal cloak dir not found; tracing will be n/a: "
            f"{config.third_party.internal.cloak_dir}"
        )
    internal_dataset = _limit_dataset(
        internal_dataset,
        config.third_party.internal.max_images,
        "third_party.internal.max_images",
    )
    internal_loader = DataLoader(
        internal_dataset,
        batch_size=config.third_party.internal.batch_size,
        shuffle=False,
        num_workers=config.third_party.defense.num_workers,
        pin_memory=True,
    )
    internal_simswap = None
    effectiveness = None

    def run_internal_simswap(epoch: int, label: str = "Internal") -> None:
        nonlocal internal_simswap, effectiveness
        if internal_simswap is None:
            from src.evaluate import Effectiveness
            from .simswap_validation import SimSwapValidator

            internal_simswap = SimSwapValidator(config)
            effectiveness = Effectiveness(logger, config)
        evaluate_internal_simswap(
            model,
            internal_loader,
            device,
            config,
            logger,
            internal_simswap,
            effectiveness,
            epoch,
            label=label,
        )

    best_val = float("inf")
    best_ckpt_path = ckpt_dir / f"{len(train_loader.dataset)}.pth"
    for epoch in range(1, config.third_party.defense.epochs + 1):
        model.train()
        total_loss = 0.0
        for perturb, clean in train_loader:
            perturb = perturb.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            pred = model(perturb)
            loss = F.l1_loss(pred, clean)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * perturb.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, device, sample_dir, epoch)
        val_text = "n/a" if val_loss is None else f"{val_loss:.6f}"
        logger.info(f"epoch={epoch:04d} train_l1={train_loss:.6f} val_l1={val_text}")

        if (
            config.third_party.internal.simswap
            and config.third_party.internal.interval > 0
            and epoch % config.third_party.internal.interval == 0
        ):
            run_internal_simswap(epoch)

        is_best = val_loss is not None and val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt_path)

    if config.third_party.internal.simswap and config.third_party.internal.final:
        run_internal_simswap(config.third_party.defense.epochs, label="Final")


def test_with_config(config: Any, logger: Any) -> None:
    Path(config.image_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = config.third_party.test.checkpoint_path
    if checkpoint_path is None:
        raise ValueError(
            "third_party.test.checkpoint_path is required for denoiser test"
        )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(config.root_dir) / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Denoiser checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda")
    model = SimpleDenoiserUNet(
        base_channels=config.third_party.model.base_channels,
        depth=config.third_party.model.depth,
        residual=config.third_party.model.residual,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded denoiser checkpoint: {checkpoint_path}")

    test_dataset = PairedImageDataset(
        config.third_party.dataset.test_dir,
        image_size=config.third_party.dataset.image_size,
        cloak_dir=config.third_party.test.cloak_dir,
        include_cloak=True,
        target_dir=config.third_party.test.target_dir,
        include_target=True,
    )
    if test_dataset.include_cloak:
        logger.info(f"Use denoiser test cloak dir: {test_dataset.cloak_dir}")
    else:
        logger.info(
            f"Denoiser test cloak dir not found; TSR will be n/a: "
            f"{config.third_party.test.cloak_dir}"
        )
    if test_dataset.include_target:
        logger.info(f"Use denoiser test target dir: {test_dataset.target_dir}")
    else:
        logger.warning(
            f"Denoiser test target dir not found; falling back to clean as target: "
            f"{config.third_party.test.target_dir}"
        )
    test_dataset = _limit_dataset(
        test_dataset,
        config.third_party.test.max_images,
        "third_party.test.max_images",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.third_party.test.batch_size,
        shuffle=False,
        num_workers=config.third_party.defense.num_workers,
        pin_memory=True,
    )

    from src.evaluate import Effectiveness
    from .simswap_validation import SimSwapValidator

    simswap = SimSwapValidator(config)
    effectiveness = Effectiveness(logger, config)
    evaluate_internal_simswap(
        model,
        test_loader,
        device,
        config,
        logger,
        simswap,
        effectiveness,
        epoch=0,
        label="Test",
        eval_config=config.third_party.test,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    sample_dir = out_dir / "samples"
    ckpt_dir = out_dir / "checkpoints"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(args)
    model = SimpleDenoiserUNet(
        base_channels=args.base_channels,
        depth=args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    best_ckpt_path = ckpt_dir / f"{len(train_loader.dataset)}.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for perturb, clean in train_loader:
            perturb = perturb.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            pred = model(perturb)
            loss = F.l1_loss(pred, clean)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * perturb.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, device, sample_dir, epoch)
        val_text = "n/a" if val_loss is None else f"{val_loss:.6f}"
        print(f"epoch={epoch:04d} train_l1={train_loss:.6f} val_l1={val_text}")

        is_best = val_loss is not None and val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save(model.state_dict(), best_ckpt_path)


if __name__ == "__main__":
    main()
