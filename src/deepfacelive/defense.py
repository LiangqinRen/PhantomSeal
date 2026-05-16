import textwrap
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from src.common_utils import save_tensor_imgs
from src.dataset import FFHQMetric
from src.deepfacelive.base import Base
from src.evaluate import Utility


class Defense(Base):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self.image_dir = Path(self.config.image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.clean_image_dir = self.image_dir / "clean"
        self.perturb_image_dir = self.image_dir / "perturb"
        self.clean_image_dir.mkdir(parents=True, exist_ok=True)
        self.perturb_image_dir.mkdir(parents=True, exist_ok=True)
        self.clean_swap_dir = self._optional_path(
            self.config.third_party.dataset.get("clean_swap_dir", None)
        )
        self.protected_swap_dir = self._optional_path(
            self.config.third_party.dataset.get("protected_swap_dir", None)
        )
        Path(self.config.notes_path).touch(exist_ok=True)
        self.utility = Utility(logger, config)

    def perturb(self) -> None:
        config = self.config.third_party
        dataset_config = config.dataset
        transform = transforms.Compose(
            [
                transforms.Resize((dataset_config.input_size, dataset_config.input_size)),
                transforms.ToTensor(),
            ]
        )
        dataset = FFHQMetric(
            Path(dataset_config.metric_dir), dataset_config.metric_pairs, transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.defense.batch_size,
            shuffle=True,
        )

        loss_total = {
            "total": 0.0,
            "detector": 0.0,
            "marker": 0.0,
            "align": 0.0,
            "aligned_face": 0.0,
            "fidelity": 0.0,
            "tv": 0.0,
            "detected": 0.0,
            "yolo_score": 0.0,
        }
        utility_total = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        swap_utility_total = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        swap_utility_batches = 0
        total_count = 0
        for idx, (imgs_A, _) in enumerate(dataloader, start=1):
            imgs_A = imgs_A.to(self.device)
            batch_start_index = total_count
            total_count += len(imgs_A)
            pert_imgs = self.perturb_imgs(imgs_A)

            if dataset_config.output_size != dataset_config.input_size:
                save_imgs_A = F.interpolate(
                    imgs_A,
                    size=(dataset_config.output_size, dataset_config.output_size),
                    mode="bilinear",
                    align_corners=False,
                )
                save_pert_imgs = F.interpolate(
                    pert_imgs,
                    size=(dataset_config.output_size, dataset_config.output_size),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                save_imgs_A = imgs_A
                save_pert_imgs = pert_imgs

            loss_summary = self.last_loss_summary
            for key in loss_total:
                loss_total[key] += loss_summary[key]
            utility = self.utility.calculate_utility(save_imgs_A, save_pert_imgs)
            for key in utility_total:
                utility_total[key] += float(utility[key])
            swap_utility = self._calculate_swap_utility(
                start_index=batch_start_index,
                count=len(imgs_A),
                size=dataset_config.output_size,
            )
            if swap_utility is not None:
                swap_utility_batches += 1
                for key in swap_utility_total:
                    swap_utility_total[key] += float(swap_utility[key])

            save_tensor_imgs(
                self.image_dir,
                idx,
                ["imgs", "deepfacelive_facemesh_perturb"],
                [save_imgs_A, save_pert_imgs],
                only_save_summary=config.defense.only_save_summary,
            )
            saved_count = self._save_image_pairs(
                save_imgs_A, save_pert_imgs, start_index=batch_start_index
            )

            iter_log_str = textwrap.dedent(
                f"""
            Batch {idx:4}/{len(dataloader):4}, {total_count} images
            perturb_level: {config.defense.perturb_level:.5f}
            deepfacelive score: total={loss_summary['total']:.5f}, yolo={loss_summary.get('yolo_score', 0.0):.5f}, detector={loss_summary['detector']:.5f}, marker={loss_summary['marker']:.5f}, align={loss_summary['align']:.5f}, aligned_face={loss_summary['aligned_face']:.5f}, fidelity={loss_summary['fidelity']:.5f}, tv={loss_summary['tv']:.5f}, detected={loss_summary['detected']:.3f}
            protected image utility: {self._format_utility(utility)}
            saved {saved_count} clean/perturb pairs to {self.clean_image_dir} and {self.perturb_image_dir}
            """
            )
            if swap_utility is not None:
                iter_log_str += (
                    f"            swap result utility: {self._format_utility(swap_utility)}\n"
                )
            elif self.clean_swap_dir is not None or self.protected_swap_dir is not None:
                iter_log_str += (
                    "            swap result utility: unavailable "
                    "(missing clean/protected swap pairs for this batch)\n"
                )
            summary_log_str = textwrap.dedent(
                f"""
            summary deepfacelive score: total={loss_total['total'] / idx:.5f}, yolo={loss_total['yolo_score'] / idx:.5f}, detector={loss_total['detector'] / idx:.5f}, marker={loss_total['marker'] / idx:.5f}, align={loss_total['align'] / idx:.5f}, aligned_face={loss_total['aligned_face'] / idx:.5f}, fidelity={loss_total['fidelity'] / idx:.5f}, tv={loss_total['tv'] / idx:.5f}, detected={loss_total['detected'] / idx:.3f}
            summary protected image utility: {self._format_utility_average(utility_total, idx)}
            """
            )
            if swap_utility_batches != 0:
                summary_log_str += (
                    "            summary swap result utility: "
                    f"{self._format_utility_average(swap_utility_total, swap_utility_batches)}\n"
                )
            self.logger.info(textwrap.indent(iter_log_str, "    "))
            self.logger.info(textwrap.indent(summary_log_str, "    "))
            del imgs_A, pert_imgs, save_imgs_A, save_pert_imgs
            self._free_gpu()

    def _save_image_pairs(
        self,
        clean_imgs: torch.Tensor,
        perturb_imgs: torch.Tensor,
        start_index: int,
    ) -> int:
        clean_imgs = clean_imgs.detach().clamp(0, 1).cpu()
        perturb_imgs = perturb_imgs.detach().clamp(0, 1).cpu()
        for offset, (clean_img, perturb_img) in enumerate(zip(clean_imgs, perturb_imgs)):
            image_index = start_index + offset + 1
            filename = f"{image_index:06d}.png"
            save_image(clean_img, self.clean_image_dir / filename)
            save_image(perturb_img, self.perturb_image_dir / filename)
        return len(clean_imgs)

    @staticmethod
    def _optional_path(value) -> Path | None:
        if value in (None, "", "null"):
            return None
        return Path(value).resolve()

    def _calculate_swap_utility(
        self,
        start_index: int,
        count: int,
        size: int,
    ) -> dict | None:
        if self.clean_swap_dir is None or self.protected_swap_dir is None:
            return None

        clean_swap_imgs = []
        protected_swap_imgs = []
        for offset in range(count):
            image_index = start_index + offset + 1
            clean_path = self._find_numbered_image(self.clean_swap_dir, image_index)
            protected_path = self._find_numbered_image(
                self.protected_swap_dir, image_index
            )
            if clean_path is None or protected_path is None:
                continue

            clean_swap_imgs.append(self._load_image_tensor(clean_path, size))
            protected_swap_imgs.append(self._load_image_tensor(protected_path, size))

        if not clean_swap_imgs:
            return None

        return self.utility.calculate_utility(
            torch.stack(clean_swap_imgs, dim=0).to(self.device),
            torch.stack(protected_swap_imgs, dim=0).to(self.device),
        )

    @staticmethod
    def _find_numbered_image(directory: Path, image_index: int) -> Path | None:
        suffixes = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        candidates = [f"{image_index:06d}", f"{image_index - 1:06d}"]
        for stem in candidates:
            for suffix in suffixes:
                path = directory / f"{stem}{suffix}"
                if path.exists():
                    return path
        return None

    @staticmethod
    def _load_image_tensor(path: Path, size: int) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.BILINEAR)
        return transforms.ToTensor()(image)

    @staticmethod
    def _format_utility(utility: dict) -> str:
        return (
            f"mse_255={float(utility['mse']):.3f}, "
            f"psnr={float(utility['psnr']):.3f}, "
            f"ssim={float(utility['ssim']):.3f}, "
            f"lpips={float(utility['lpips']):.3f}"
        )

    def _format_utility_average(self, total: dict, count: int) -> str:
        if count <= 0:
            return "n/a"
        return self._format_utility({key: total[key] / count for key in total})
