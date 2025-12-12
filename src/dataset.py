import random
import torch
import face_recognition
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
from tqdm import tqdm
from torchvision import transforms


class SampleDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.A, self.B = self._get_double_imgs_list()

    def _get_double_imgs_list(self):
        sample_dir = self.root_dir

        A = [sample_dir / f for f in ["zjl.jpg", "james.jpg", "source.png"]]
        B = [sample_dir / f for f in ["6.jpg", "6.jpg", "6.jpg"]]

        return A, B

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        img_A_path, img_B_path = self.A[idx], self.B[idx]

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
        self.metric_pairs = config.third_party.dataset.metric_pairs
        self.mtcnn = MTCNN(
            keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.A, self.B, self.C = self._get_triple_imgs_list()

    def _is_single_identity_img(self, img_path: Path) -> bool:
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)

        return len(face_locations) == 1

    def _filter_valid_images(self, all_imgs_path: list, count: int) -> list:
        valid_imgs_path = []
        remaining = count
        with tqdm(total=remaining, desc="Filtering valid images") as pbar:
            for img_path in all_imgs_path:
                if self._is_single_identity_img(img_path):
                    valid_imgs_path.append(img_path)
                    remaining -= 1
                    pbar.update(1)
                    pbar.set_postfix(remaining=remaining)

                if remaining <= 0:
                    break

        return valid_imgs_path

    def _get_triple_imgs_list(self):
        all_people = [f for f in self.root_dir.iterdir() if f.is_dir()]
        all_people = sorted(all_people)
        random.shuffle(all_people)

        A, B, C = [], [], []
        for idx, people in enumerate(all_people):
            if idx % 3 == 0:
                A.extend([f for f in people.iterdir() if f.is_file()])
            elif idx % 3 == 1:
                B.extend([f for f in people.iterdir() if f.is_file()])
            elif idx % 3 == 2:
                C.extend([f for f in people.iterdir() if f.is_file()])

        A, B, C = sorted(A), sorted(B), sorted(C)
        random.shuffle(A)
        random.shuffle(B)
        random.shuffle(C)

        valid_imgs_A_path = self._filter_valid_images(A, self.metric_pairs)
        valid_imgs_B_path = self._filter_valid_images(B, self.metric_pairs)
        valid_imgs_C_path = self._filter_valid_images(C, self.metric_pairs)
        min_count = min(
            len(valid_imgs_A_path),
            len(valid_imgs_B_path),
            len(valid_imgs_C_path),
            self.metric_pairs,
        )

        return (
            valid_imgs_A_path[:min_count],
            valid_imgs_B_path[:min_count],
            valid_imgs_C_path[:min_count],
        )

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        img_A_path, img_B_path, img_C_path = self.A[idx], self.B[idx], self.C[idx]

        img_A = self.transform(Image.open(img_A_path).convert("RGB"))
        img_B = self.transform(Image.open(img_B_path).convert("RGB"))
        img_C = self.transform(Image.open(img_C_path).convert("RGB"))

        return img_A, img_B, img_C


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
