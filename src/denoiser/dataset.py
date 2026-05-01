from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PairedImageDataset(Dataset):
    """
    Paired clean/perturb image dataset.

    Expected layout:
        root/
            clean/
                1.png
            perturb/
                1.png

    Files are paired by identical relative filename.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 256,
        clean_dir: str = "clean",
        perturb_dir: str = "perturb",
        cloak_dir: str = "cloak",
        target_dir: str = "target",
        include_cloak: bool = False,
        include_target: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.clean_dir = self.root_dir / clean_dir
        self.perturb_dir = self.root_dir / perturb_dir
        self.cloak_dir = self.root_dir / cloak_dir
        self.target_dir = self.root_dir / target_dir
        self.include_cloak = include_cloak and self.cloak_dir.exists()
        self.include_target = include_target and self.target_dir.exists()
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.pairs = self._get_pairs()

    def _get_pairs(self) -> list[tuple[Path, ...]]:
        clean_paths = [
            path
            for path in sorted(self.clean_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
        ]
        pairs = []
        for clean_path in clean_paths:
            rel_path = clean_path.relative_to(self.clean_dir)
            perturb_path = self.perturb_dir / rel_path
            if perturb_path.exists():
                pair = [perturb_path, clean_path]
                if self.include_cloak:
                    cloak_path = self.cloak_dir / rel_path
                    if not cloak_path.exists():
                        continue
                    pair.append(cloak_path)
                if self.include_target:
                    target_path = self.target_dir / rel_path
                    if not target_path.exists():
                        continue
                    pair.append(target_path)
                pairs.append(tuple(pair))

        if not pairs:
            raise ValueError(
                f"No paired images found under {self.clean_dir} and {self.perturb_dir}"
            )
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        perturb_path, clean_path = pair[:2]
        perturb = Image.open(perturb_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")
        if len(pair) == 3:
            extra = Image.open(pair[2]).convert("RGB")
            item = (self.transform(perturb), self.transform(clean), self.transform(extra))
        elif len(pair) == 4:
            cloak = Image.open(pair[2]).convert("RGB")
            target = Image.open(pair[3]).convert("RGB")
            item = (
                self.transform(perturb),
                self.transform(clean),
                self.transform(cloak),
                self.transform(target),
            )
        else:
            item = (self.transform(perturb), self.transform(clean))
        return item
