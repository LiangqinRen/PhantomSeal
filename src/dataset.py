import random
import torch
import face_recognition
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
from torchvision import transforms


class SampleDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.A, self.B = self.__get_double_imgs_list()

    def __get_double_imgs_list(self):
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
        self.metric_pairs = config.third_party.dataset.metric_pairs
        self.mtcnn = MTCNN(
            keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.A, self.B = self.__get_double_imgs_list()

    def __is_single_identity_img(self, img_path: Path) -> bool:
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)

        return len(face_locations) == 1

    def __filter_valid_images(self, all_imgs_path: list, count: int) -> list:
        valid_imgs_path = []
        remaining = count
        with tqdm(total=remaining, desc="Filtering valid images") as pbar:
            for img_path in all_imgs_path:
                if self.__is_single_identity_img(img_path):
                    valid_imgs_path.append(img_path)
                    remaining -= 1
                    pbar.update(1)
                    pbar.set_postfix(remaining=remaining)

                if remaining <= 0:
                    break

        return valid_imgs_path

    def __get_double_imgs_list(self):
        all_people = [f for f in self.root_dir.iterdir() if f.is_dir()]
        all_people = sorted(all_people)
        random.shuffle(all_people)

        A, B = [], []
        for idx, people in enumerate(all_people):
            if idx % 2 == 0:
                A.extend([f for f in people.iterdir() if f.is_file()])
            elif idx % 2 == 1:
                B.extend([f for f in people.iterdir() if f.is_file()])

        A, B = sorted(A), sorted(B)
        random.shuffle(A)
        random.shuffle(B)

        min_count = min(len(A), len(B), self.metric_pairs)
        valid_imgs_A_path = self.__filter_valid_images(A, min_count)
        valid_imgs_B_path = self.__filter_valid_images(B, min_count)

        return valid_imgs_A_path, valid_imgs_B_path

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        img_A_path, img_B_path = self.A[idx], self.B[idx]

        img_A = self.transform(Image.open(img_A_path).convert("RGB"))
        img_B = self.transform(Image.open(img_B_path).convert("RGB"))

        return img_A, img_B
