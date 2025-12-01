import random
import torch
import face_recognition
import cv2
from numpy import ndarray
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
from tqdm import tqdm
from torchvision import transforms
from insightface.app import FaceAnalysis


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
        self.A, self.B = self._get_double_imgs_list()

        image_size = config.third_party.dataset.input_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def _get_double_imgs_list(self):
        images_path = [f for f in self.root_dir.iterdir() if f.is_file()]
        images_path = sorted(images_path)
        random.shuffle(images_path)

        A, B = [], []
        for idx, img_path in enumerate(images_path):
            if idx % 2 == 0:
                A.append(img_path)
            else:
                B.append(img_path)

        A, B = sorted(A), sorted(B)
        random.shuffle(A)
        random.shuffle(B)

        min_count = min(len(A), len(B), self.metric_pairs)

        return A[:min_count], B[:min_count]

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        img_A_path, img_B_path = self.A[idx], self.B[idx]

        img_A = self.transform(Image.open(img_A_path).convert("RGB"))
        img_B = self.transform(Image.open(img_B_path).convert("RGB"))

        return img_A, img_B


class FFPPDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root_dir = Path(config.third_party.dataset.FFPP_original_dir)

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.A, self.B = self._get_double_imgs_ndarray()

        image_size = config.third_party.dataset.input_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def _get_random_frame_from_video(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rand_index = random.randint(0, total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, rand_index)
        ret, frame = cap.read()
        cap.release()

        return frame

    def _crop_face_center_by_landmarks(
        self, img_bgr: ndarray, scale: float = 2.0, out_size: int = 256
    ):
        faces = self.app.get(img_bgr)
        if len(faces) == 0:
            raise ValueError("No face detected.")

        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        bbox = face.bbox.astype(int)
        kps = face.kps
        cx, cy = int(kps[:, 0].mean()), int(kps[:, 1].mean())

        face_w = bbox[2] - bbox[0]
        face_h = bbox[3] - bbox[1]
        face_size = int(max(face_w, face_h) * scale)

        half = face_size // 2
        H, W = img_bgr.shape[:2]
        x0, y0 = cx - half, cy - half
        x1, y1 = cx + half, cy + half

        pad_l = max(0, -x0)
        pad_t = max(0, -y0)
        pad_r = max(0, x1 - W)
        pad_b = max(0, y1 - H)
        if any([pad_l, pad_t, pad_r, pad_b]):
            img_bgr = cv2.copyMakeBorder(
                img_bgr, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REFLECT_101
            )
            x0 += pad_l
            x1 += pad_l
            y0 += pad_t
            y1 += pad_t

        crop = img_bgr[y0:y1, x0:x1].copy()
        resized = cv2.resize(crop, (out_size, out_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb

    def _get_double_imgs_ndarray(self):
        videos_path = [f for f in self.root_dir.iterdir() if f.is_file()]
        videos_path = sorted(videos_path)
        random.shuffle(videos_path)

        length = int(len(videos_path) / 2)
        length = 9
        A, B = sorted(videos_path[:length]), sorted(videos_path[length:])
        random.shuffle(A)
        random.shuffle(B)

        C = [
            self._crop_face_center_by_landmarks(self._get_random_frame_from_video(path))
            for path in A
        ]
        D = [
            self._crop_face_center_by_landmarks(self._get_random_frame_from_video(path))
            for path in B
        ]

        return C, D

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        img_A_path, img_B_path = self.A[idx], self.B[idx]

        img_A = self.transform(Image.fromarray(img_A_path))
        img_B = self.transform(Image.fromarray(img_B_path))

        return img_A, img_B
