import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SampleDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.A, self.B = self._get_double_imgs_list()

    def _get_double_imgs_list(self):
        sample_dir = self.root_dir

        A = [
            sample_dir / f
            for f in ["imgs_A_1_7.png", "imgs_A_5_1.png", "imgs_A_7_4.png"]
        ]
        B = [
            sample_dir / f
            for f in ["imgs_B_1_7.png", "imgs_B_5_1.png", "imgs_B_7_4.png"]
        ]

        return A, B

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        img_A_path, img_B_path = self.A[idx], self.B[idx]

        img_A = self.transform(Image.open(img_A_path).convert("RGB"))
        img_B = self.transform(Image.open(img_B_path).convert("RGB"))

        return img_A, img_B


class VGGFace2Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root_dir = Path(config.dataset.metric_512_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.images = self._get_images_list()
        self.index_pairs = self._get_random_pairs()

    def _get_images_list(self) -> list[Path]:
        images = sorted([f for f in self.root_dir.iterdir() if f.is_file()])
        return images

    def _get_random_pairs(self) -> list[tuple[int, int]]:
        metric_pairs = self.config.dataset.metric_pairs
        image_count = len(self.images)
        index_pairs = []
        for _ in range(metric_pairs):
            i = random.randrange(image_count)
            j = random.randrange(image_count)
            while j == i:
                j = random.randrange(image_count)
            index_pairs.append((i, j))

        return index_pairs

    def __len__(self):
        return self.config.dataset.metric_pairs

    def __getitem__(self, idx):
        idx_a, idx_b = self.index_pairs[idx]
        img_A_path, img_B_path = self.images[idx_a], self.images[idx_b]

        img_A = self.transform(Image.open(img_A_path).convert("RGB"))
        img_B = self.transform(Image.open(img_B_path).convert("RGB"))

        return img_A, img_B


class MetricDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root_dir = Path(
            config.third_party.dataset.metric_224_dir
            if config.third_party.dataset.use_224
            else config.third_party.dataset.metric_512_dir
        )
        if config.third_party.name == "faceshifter":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif (
            config.third_party.name == "simswap" and config.third_party.dataset.use_224
        ):
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transforms.Compose(
                [transforms.Resize(256), transforms.ToTensor()]
            )

        self.images = self._get_images_list()
        self.index_pairs = self._get_random_pairs()

    def _get_images_list(self) -> list[Path]:
        images = sorted([f for f in self.root_dir.iterdir() if f.is_file()])
        return images

    def _get_random_pairs(self) -> list[tuple[int, int]]:
        metric_pairs = self.config.third_party.dataset.metric_pairs
        image_count = len(self.images)
        index_pairs = []
        for _ in range(metric_pairs):
            i = random.randrange(image_count)
            j = random.randrange(image_count)
            while j == i:
                j = random.randrange(image_count)
            index_pairs.append((i, j))

        return index_pairs

    def __len__(self):
        return self.config.third_party.dataset.metric_pairs

    def __getitem__(self, idx):
        idx_a, idx_b = self.index_pairs[idx]
        img_A_path, img_B_path = self.images[idx_a], self.images[idx_b]

        img_A = self.transform(Image.open(img_A_path).convert("RGB"))
        img_B = self.transform(Image.open(img_B_path).convert("RGB"))

        return img_A, img_B


class AdaptiveMetricDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root_dir = Path(
            config.third_party.dataset.metric_224_dir
            if config.third_party.dataset.use_224
            else config.third_party.dataset.metric_512_dir
        )

        if config.third_party.dataset.use_224:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transforms.Compose(
                [transforms.Resize(256), transforms.ToTensor()]
            )

        self.images = self._get_images_list()
        self.index_pairs = self._get_random_pairs()

    def _get_images_list(self) -> list[Path]:
        images = sorted([f for f in self.root_dir.iterdir() if f.is_file()])
        return images

    def _get_random_pairs(self) -> list[tuple[int, int]]:
        metric_pairs = self.config.third_party.dataset.metric_pairs
        image_count = len(self.images)
        index_pairs = []
        for _ in range(metric_pairs):
            i = j = k = 0
            while i == j or i == k or j == k:
                i = random.randrange(image_count)
                j = random.randrange(image_count)
                k = random.randrange(image_count)
            index_pairs.append((i, j, k))

        return index_pairs

    def __len__(self):
        return self.config.third_party.dataset.metric_pairs

    def __getitem__(self, idx):
        idx_a, idx_b, idx_c = self.index_pairs[idx]
        img_A_path, img_B_path, img_C_path = (
            self.images[idx_a],
            self.images[idx_b],
            self.images[idx_c],
        )

        img_A = self.transform(Image.open(img_A_path).convert("RGB"))
        img_B = self.transform(Image.open(img_B_path).convert("RGB"))
        img_C = self.transform(Image.open(img_C_path).convert("RGB"))

        return img_A, img_B, img_C


class FFHQSample(Dataset):
    def __init__(self, config):
        self.root_dir = Path(config.third_party.dataset.sample_dir)
        image_size = config.third_party.dataset.input_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.A, self.B = self._get_double_imgs_list()

    def _get_double_imgs_list(self):
        sample_dir = self.root_dir

        A = [sample_dir / f for f in ["09861.png"]]
        B = [sample_dir / f for f in ["67533.png"]]

        return A, B

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        img_A_path, img_B_path = self.A[idx], self.B[idx]

        img_A = self.transform(Image.open(img_A_path).convert("RGB"))
        img_B = self.transform(Image.open(img_B_path).convert("RGB"))

        return img_A, img_B


class DiffFaceFFHQSample(Dataset):
    def __init__(self, config):
        self.root_dir = Path(config.third_party.dataset.sample_dir)
        image_size = config.third_party.dataset.input_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.A, self.B = self._get_double_imgs_list()

    def _get_double_imgs_list(self):
        sample_dir = self.root_dir

        A = [sample_dir / f for f in ["09861.png"]]
        B = [sample_dir / f for f in ["67533.png"]]

        return A, B

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        img_A_path, img_B_path = self.A[idx], self.B[idx]

        img_A = self.transform(Image.open(img_A_path).convert("RGB"))
        img_B = self.transform(Image.open(img_B_path).convert("RGB"))

        return img_A, img_B


class FFHQDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root_dir = Path(config.third_party.dataset.metric_dir)

        self.metric_pairs = config.third_party.dataset.metric_pairs

        image_size = config.third_party.dataset.input_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.images = self._get_images_list()
        self.index_pairs = self._get_random_pairs()

    def _get_images_list(self) -> list[Path]:
        images = sorted([f for f in self.root_dir.iterdir() if f.is_file()])
        return images

    def _get_random_pairs(self) -> list[tuple[int, int]]:
        metric_pairs = self.config.third_party.dataset.metric_pairs
        image_count = len(self.images)
        index_pairs = []
        for _ in range(metric_pairs):
            i = random.randrange(image_count)
            j = random.randrange(image_count)
            while j == i:
                j = random.randrange(image_count)
            index_pairs.append((i, j))

        return index_pairs

    def __len__(self):
        return self.config.third_party.dataset.metric_pairs

    def __getitem__(self, idx):
        idx_a, idx_b = self.index_pairs[idx]
        img_A_path, img_B_path = self.images[idx_a], self.images[idx_b]

        img_A = self.transform(Image.open(img_A_path).convert("RGB"))
        img_B = self.transform(Image.open(img_B_path).convert("RGB"))

        return img_A, img_B


class DiffFaceFFHQDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root_dir = Path(config.third_party.dataset.metric_dir)

        self.metric_pairs = config.third_party.dataset.metric_pairs

        image_size = config.third_party.dataset.input_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.images = self._get_images_list()
        self.index_pairs = self._get_random_pairs()

    def _get_images_list(self) -> list[Path]:
        images = sorted([f for f in self.root_dir.iterdir() if f.is_file()])
        return images

    def _get_random_pairs(self) -> list[tuple[int, int]]:
        metric_pairs = self.config.third_party.dataset.metric_pairs
        image_count = len(self.images)
        index_pairs = []
        for _ in range(metric_pairs):
            i = random.randrange(image_count)
            j = random.randrange(image_count)
            while j == i:
                j = random.randrange(image_count)
            index_pairs.append((i, j))

        return index_pairs

    def __len__(self):
        return self.config.third_party.dataset.metric_pairs

    def __getitem__(self, idx):
        idx_a, idx_b = self.index_pairs[idx]
        img_A_path, img_B_path = self.images[idx_a], self.images[idx_b]

        img_A = self.transform(Image.open(img_A_path).convert("RGB"))
        img_B = self.transform(Image.open(img_B_path).convert("RGB"))

        return img_A, img_B


class AFDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root_dir = Path(config.third_party.dataset.metric_512_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize(config.third_party.origin.image_resolution),
                transforms.ToTensor(),
            ]
        )

        self.images_path = self._get_images_list()

    def _get_images_list(self) -> list[Path]:
        image_path = sorted(
            p
            for p in self.root_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

        return image_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = self.images_path[random.randint(0, len(self.images_path) - 1)]
        image = self.transform(Image.open(image_path).convert("RGB"))

        return image


class SepMarkDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root_dir = Path(config.third_party.dataset.metric_dir)
        self.transform = transforms.Compose(
            [transforms.Resize(256), transforms.ToTensor()]
        )

        self.images = self._get_images_list()
        self.index_pairs = self._get_random_pairs()

    def _get_images_list(self) -> list[Path]:
        images = sorted([f for f in self.root_dir.iterdir() if f.is_file()])
        return images

    def _get_random_pairs(self) -> list[tuple[int, int]]:
        metric_pairs = self.config.third_party.dataset.metric_pairs
        image_count = len(self.images)
        index_pairs = []
        for _ in range(metric_pairs):
            i = random.randrange(image_count)
            j = random.randrange(image_count)
            while j == i:
                j = random.randrange(image_count)
            index_pairs.append((i, j))

        return index_pairs

    def __len__(self):
        return self.config.third_party.dataset.metric_pairs

    def __getitem__(self, idx):
        idx_a, idx_b = self.index_pairs[idx]
        img_A_path, img_B_path = self.images[idx_a], self.images[idx_b]

        img_A = self.transform(Image.open(img_A_path).convert("RGB"))
        img_B = self.transform(Image.open(img_B_path).convert("RGB"))

        return img_A, img_B
