import torch
import lpips
import math
import requests
import base64
import time
import random
import warnings
import urllib3
import traceback
import hmac
import hashlib
import json
import boto3
import face_recognition
import threading
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import deque
from omegaconf import OmegaConf
from http.client import HTTPSConnection
from datetime import datetime, timedelta, timezone
from facenet_pytorch import MTCNN, InceptionResnetV1
from skimage import metrics
from torch import Tensor
from PIL import Image
from io import BytesIO
from torchvision.transforms.functional import to_pil_image
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast
from torchvision.utils import save_image

# _ORIG_TORCH_LOAD = torch.load


# def _load_patch(*args, **kwargs):
#     if (
#         args
#         and isinstance(args[0], str)
#         and os.path.basename(args[0]) in ("Arcface_model_only.tar")
#     ):
#         kwargs.setdefault("weights_only", False)
#         kwargs.setdefault("map_location", "cpu")
#     return _ORIG_TORCH_LOAD(*args, **kwargs)


# torch.load = _load_patch


class RateLimiter:
    def __init__(self, rps: float):
        if rps <= 0:
            raise ValueError("RateLimiter rps must be positive!")

        self.rps = rps

        self._call_history = deque()
        self._lock = threading.Lock()

    def wait_if_needed(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._call_history and now - self._call_history[0] >= 1.0:
                    self._call_history.popleft()

                if len(self._call_history) < self.rps:
                    self._call_history.append(now)
                    return

                wait_time = 1.0 - (now - self._call_history[0])

            if wait_time > 0:
                time.sleep(min(wait_time, 0.1))


class Utility:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.lpips_distance = lpips.LPIPS(net="vgg", verbose=False).cuda()

    def calculate_utility(self, imgs1: Tensor, imgs2: Tensor) -> dict | None:
        if imgs1 is None or imgs2 is None:
            return None

        utilities = {"mse": [], "psnr": [], "ssim": [], "lpips": []}

        imgs1_ndarray = imgs1.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        imgs2_ndarray = imgs2.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        for i in range(min(imgs1.shape[0], imgs2.shape[0])):
            mse = metrics.mean_squared_error(imgs1_ndarray[i], imgs2_ndarray[i])
            utilities["mse"].append(mse)

            utilities["psnr"].append(
                metrics.peak_signal_noise_ratio(
                    imgs1_ndarray[i], imgs2_ndarray[i], data_range=255
                )
            )

            utilities["ssim"].append(
                metrics.structural_similarity(
                    imgs1_ndarray[i],
                    imgs2_ndarray[i],
                    channel_axis=2,
                    data_range=255,
                )
            )

            lpips_score = self.lpips_distance(imgs1[i], imgs2[i])
            utilities["lpips"].append(lpips_score.detach().cpu().numpy())

        for i in utilities:
            utilities[i] = np.mean(utilities[i])

        return utilities


class Effectiveness:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

        self.candi_funcs = self._init_functions()

        self.mtcnn = MTCNN(
            image_size=160,
            device="cuda",
            selection_method="largest",
            keep_all=False,
        )
        self.FaceVerification = InceptionResnetV1(
            classify=False, pretrained="vggface2"
        ).cuda()
        self.FaceVerification.eval()

        self.aws_client = boto3.client(
            "rekognition",
            aws_access_key_id=self.config.evaluate.aws.api_key,
            aws_secret_access_key=self.config.evaluate.aws.api_secret,
            region_name=self.config.evaluate.aws.api_region,
        )

        self.facepp_limiter = RateLimiter(config.evaluate.facepp.qps)

    def _init_functions(self) -> dict:
        candidate_functions = {}

        if self.config.evaluate.facenet_512.use:
            candidate_functions["facenet"] = self._get_facenet_matching
        if self.config.evaluate.face_recognition.use:
            candidate_functions["facerec"] = self._get_facerec_matching
        if self.config.evaluate.facepp.use:
            candidate_functions["face++"] = self._get_facepp_matching
        if self.config.evaluate.aws.use:
            candidate_functions["aws"] = self._get_aws_matching

        return candidate_functions

    def _convert_to_bgr112_ndarray(self, x: Tensor) -> np.ndarray:
        assert x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 3
        x = x.detach().float()

        xmax, xmin = float(x.max().item()), float(x.min().item())
        if xmax <= 1.5 and xmin >= -0.5:
            x = x * 255.0
        elif xmin >= -1.1 and xmax <= 1.1:
            x = (x * 127.5) + 127.5
        x = x.clamp(0, 255)

        x = F.interpolate(
            x, size=(112, 112), mode="bilinear", align_corners=False, antialias=True
        )

        img = x[0].permute(1, 2, 0).cpu().numpy()
        img = img[..., ::-1]
        img = np.ascontiguousarray(img.astype(np.uint8))

        return img

    def _convert_to_bgr112_tensor(self, x: Tensor) -> Tensor:
        assert x.dim() in (3, 4), f"unexpected shape: {x.shape}"
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.detach().clone().float()

        if x.max() <= 1.5:
            x = x * 255.0

        x = x[:, [2, 1, 0], :, :]

        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        x = (x - 127.5) / 128.0

        return x.contiguous()

    def get_images_distance(self, imgs1: Tensor, imgs2: Tensor) -> list[float]:
        distances = []

        imgs1_ndarray = imgs1.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        imgs2_ndarray = imgs2.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for i in range(imgs1_ndarray.shape[0]):
            try:
                img1_cropped = self.mtcnn(imgs1_ndarray[i])
                img2_cropped = self.mtcnn(imgs2_ndarray[i])
                if img1_cropped is None or img2_cropped is None:
                    distances.append(float("inf"))
                    continue

                img1_embeddings = (
                    self.FaceVerification(img1_cropped.unsqueeze(0).cuda())
                    .detach()
                    .cpu()
                )
                img2_embeddings = (
                    self.FaceVerification(img2_cropped.unsqueeze(0).cuda())
                    .detach()
                    .cpu()
                )

                distances.append((img1_embeddings - img2_embeddings).norm().item())
            except Exception as e:
                self.logger.warning(e)
                distances.append(math.nan)

        return distances

    def _get_facenet_matching(
        self, imgs1: Tensor, imgs2: Tensor
    ) -> tuple[float, float]:
        matching_count, valid_count = 0, 1e-10
        distances = self.get_images_distance(imgs1, imgs2)
        for distance in distances:
            if distance is math.nan:
                continue
            else:
                matching_count += distance <= self.config.evaluate.facenet_512.threshold
                valid_count += 1

        return matching_count, valid_count

    def _get_facerec_matching(
        self, imgs1: Tensor, imgs2: Tensor
    ) -> tuple[float, float]:
        matching_count, valid_count = 0, 1e-10
        for i in range(imgs1.shape[0]):
            try:
                img1, img2 = imgs1[i], imgs2[i]
                img1 = np.array(
                    Image.fromarray(
                        (img1.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(
                            np.uint8
                        )
                    )
                )
                img2 = np.array(
                    Image.fromarray(
                        (img2.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(
                            np.uint8
                        )
                    )
                )
                img1_encoding = face_recognition.face_encodings(img1, model="large")[0]
                img2_encoding = face_recognition.face_encodings(img2, model="large")[0]
                matching_count += face_recognition.compare_faces(
                    [img1_encoding], img2_encoding
                )[0]
                valid_count += 1
            except IndexError:
                valid_count += 1
            except Exception as e:
                self.logger.warning(e)

        return matching_count, valid_count

    def _get_facepp_matching_single(
        self, img1: Tensor, img2: Tensor, key: str, secret: str
    ):
        buffer1 = BytesIO()
        img1 = img1 * 255
        img_image = Image.fromarray(img1.cpu().permute(1, 2, 0).byte().numpy())
        img_image.save(buffer1, format="PNG")
        img1_base64 = base64.b64encode(buffer1.getvalue()).decode("utf-8")

        buffer2 = BytesIO()
        img2 = img2 * 255
        img_image = Image.fromarray(img2.cpu().permute(1, 2, 0).byte().numpy())
        img_image.save(buffer2, format="PNG")
        img2_base64 = base64.b64encode(buffer2.getvalue()).decode("utf-8")

        url = self.config.evaluate.facepp.compare_url
        payload = {
            "api_key": key,
            "api_secret": secret,
            "image_base64_1": img1_base64,
            "image_base64_2": img2_base64,
        }

        fail_count = 0
        while fail_count < 5:
            try:
                self.facepp_limiter.wait_if_needed()
                response = requests.post(url, data=payload)
                if response.status_code == 200:
                    response = response.json()
                    if "confidence" in response:
                        return (
                            (1, 1)
                            if response["confidence"] > response["thresholds"]["1e-5"]
                            else (0, 1)
                        )
                    elif ("faces1" in response and len(response["faces1"]) == 0) or (
                        "faces2" in response and len(response["faces2"]) == 0
                    ):
                        return (0, 1)
                    else:
                        self.logger.warning(response)
                        return (0, 1e-10)
                elif response.status_code == 400:
                    return (0, 1)
                elif response.status_code == 403:
                    fail_count += 1
                    self.logger.debug(
                        f"Face++ API rate limit reached, retrying... {response.status_code}, {response.text}"
                    )
                    time.sleep(1)
                else:
                    self.logger.error(response)
                    return (0, 1e-10)
            except BaseException as e:
                self.logger.error(e)
                return (0, 1e-10)

        return (0, 1e-10)

    def _get_facepp_matching(self, imgs1: Tensor, imgs2: Tensor) -> tuple[float, float]:
        api_key = self.config.evaluate.facepp.api_key
        api_secret = self.config.evaluate.facepp.api_secret
        thread_limit = self.config.evaluate.facepp.thread_limit
        with ThreadPoolExecutor(max_workers=thread_limit) as executor:
            futures = [
                executor.submit(
                    self._get_facepp_matching_single,
                    imgs1[i],
                    imgs2[i],
                    api_key,
                    api_secret,
                )
                for i in range(imgs1.shape[0])
            ]
            results = [future.result() for future in futures]

        success_count, total_count = 0, 1e-10
        for result in results:
            success_count += result[0]
            total_count += result[1]

        return (success_count, total_count)

    def _get_aws_matching(self, imgs1: Tensor, imgs2: Tensor) -> tuple[float, float]:
        matching_count, valid_count = 0, 1e-10
        for img1, img2 in zip(imgs1, imgs2):
            response = None
            try:
                img1 = Image.fromarray(
                    (img1.detach().cpu().permute(1, 2, 0).numpy() * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                buffer1 = BytesIO()
                img1.save(buffer1, format="png")
                img_bytes1 = buffer1.getvalue()

                img2 = Image.fromarray(
                    (img2.detach().cpu().permute(1, 2, 0).numpy() * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                buffer2 = BytesIO()
                img2.save(buffer2, format="png")
                img_bytes2 = buffer2.getvalue()

                response = self.aws_client.compare_faces(
                    SimilarityThreshold=80,
                    SourceImage={"Bytes": img_bytes1},
                    TargetImage={"Bytes": img_bytes2},
                )

                matching_count += len(response["FaceMatches"])
                valid_count += 1
            except Exception as e:
                if response is not None and "Error" in response:
                    error_code = response["Error"]["Code"]
                    if error_code == "InvalidParameterException":
                        valid_count += 1
                    else:
                        self.logger.error(e)
                        valid_count += 1e-10
                else:
                    self.logger.error(e)
                    valid_count += 1e-10

        return matching_count, valid_count

    def is_same_identity_via_facepp(self, img1: Tensor, img2: Tensor) -> list[bool]:
        api_keys = self.config.evaluate.facepp.api_key
        api_secrets = self.config.evaluate.facepp.api_secret

        index = random.randint(0, len(api_keys))
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._get_facepp_matching_single,
                    img1,
                    img2,
                    api_keys[index],
                    api_secrets[index],
                )
            ]
            matchings = [future.result() for future in futures]

        results = []
        for matching in matchings:
            results.append(matching[0] == 1)

        return results

    def get_image_distance(self, img1: np.ndarray, embeddings: list) -> list:
        img1_cropped = self.mtcnn(img1)

        if img1_cropped is None:
            return [math.nan] * len(embeddings)

        img1_embedding = self.FaceVerification(img1_cropped.unsqueeze(0).cuda())
        distances = []
        with torch.no_grad():
            for embedding in embeddings:
                distance = (embedding - img1_embedding).norm().item()
                distances.append(distance)

        return distances

    def get_success_swap_indices(
        self, src_imgs: Tensor, swap_result_imgs: Tensor
    ) -> Tensor:
        success_indices = []
        for i, (src_img, swap_result_img) in enumerate(zip(src_imgs, swap_result_imgs)):
            for (
                _,
                v,
            ) in self.candi_funcs.items():  # there should be only one function
                matching, valid = v(src_img.unsqueeze(0), swap_result_img.unsqueeze(0))
                if int(matching) == 1 and int(valid) == 1:
                    success_indices.append(i)

        return torch.tensor(success_indices, dtype=torch.long)

    def calculate_effectiveness(
        self,
        source_imgs: Tensor | None,
        pert_imgs: Tensor | None,
        swap_imgs: Tensor | None,
        pert_swap_imgs: Tensor | None,
        cloak_imgs: Tensor | None,
    ) -> dict:
        effectivenesses = {}
        for k, v in self.candi_funcs.items():
            effectivenesses[k] = {}
            if (
                source_imgs is not None
                and pert_imgs is not None
                and self.config.evaluate.effectiveness.perturb
            ):
                effectivenesses[k]["pert"] = v(source_imgs, pert_imgs)

            if (
                source_imgs is not None
                and swap_imgs is not None
                and self.config.evaluate.effectiveness.ASRo
            ):
                effectivenesses[k]["swap"] = v(source_imgs, swap_imgs)

            if (
                source_imgs is not None
                and pert_swap_imgs is not None
                and self.config.evaluate.effectiveness.ASRp
            ):
                effectivenesses[k]["pert_swap"] = v(source_imgs, pert_swap_imgs)

            if (
                pert_swap_imgs is not None
                and cloak_imgs is not None
                and self.config.evaluate.effectiveness.TSR
            ):
                effectivenesses[k]["cloak"] = v(pert_swap_imgs, cloak_imgs)

        return effectivenesses


class AIEditing:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

    def face_beauty_via_ailabtools(self, imgs: Tensor) -> tuple[Tensor, int]:
        url = self.config.evaluate.ai_lab_tools.face_beauty_url
        headers = {"ailabapi-api-key": self.config.evaluate.ai_lab_tools.api_key}
        data = {
            "sharp": "0.5",
            "smooth": "0.5",
            "white": "0.5",
        }
        transform = transforms.ToTensor()

        img_list = []
        beauty_success_count = 0
        for i in range(imgs.size(0)):
            img = to_pil_image(imgs[i])

            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            files = {"image": buffer}

            fail_count = 0
            while fail_count < 3:
                response = requests.post(url, headers=headers, files=files, data=data)
                try:
                    if response.status_code == 200:
                        response = response.json()
                        urllib3.disable_warnings(
                            urllib3.exceptions.InsecureRequestWarning
                        )
                        response = requests.get(
                            response["data"]["image_url"], verify=False
                        )

                        img = Image.open(BytesIO(response.content)).convert("RGB")
                        img_list.append(transform(img))

                        beauty_success_count += 1
                        break
                    else:
                        fail_count += 1
                        self.logger.error(
                            "ailabtools status=%s reason=%s url=%s body=%s",
                            response.status_code,
                            response.reason,
                            response.url,
                            response.text[:1000],
                        )
                except Exception as e:
                    fail_count += 1
                    self.logger.error(response)
                    self.logger.error(e)

            if fail_count == 3:
                img_list.append(imgs[i])

        beauty_imgs = torch.stack(img_list, dim=0).cuda()
        return beauty_imgs, beauty_success_count

    def cartoon_via_ailabtools(self, imgs: Tensor) -> tuple[Tensor, int]:
        url = self.config.evaluate.ai_lab_tools.cartoon_url
        headers = {"ailabapi-api-key": self.config.evaluate.ai_lab_tools.api_key}
        data = {"type": "jpcartoon"}
        transform = transforms.ToTensor()

        img_list = []
        cartoon_success_count = 0
        for i in range(imgs.size(0)):
            img = to_pil_image(imgs[i])

            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            files = {"image": buffer}

            fail_count = 0
            while fail_count < 1:
                response = requests.post(url, headers=headers, files=files, data=data)
                try:
                    if response.status_code == 200:
                        response = response.json()
                        response = requests.get(response["data"]["image_url"])
                        img = Image.open(BytesIO(response.content)).convert("RGB")
                        img_list.append(transform(img))

                        cartoon_success_count += 1
                        break
                    else:
                        fail_count += 1
                        self.logger.error(response)
                except Exception as e:
                    fail_count += 1
                    self.logger.error(response)

            if fail_count == 1:
                img_list.append(imgs[i])

        cartoon_imgs = torch.stack(img_list, dim=0).cuda()
        return cartoon_imgs, cartoon_success_count

    def face_beauty_via_tencentcloud(self, imgs: Tensor) -> tuple[Tensor, int]:
        def sign(key, msg):
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        secret_id = self.config.evaluate.tencent_cloud.secret_id
        secret_key = self.config.evaluate.tencent_cloud.secret_key
        token = ""

        img_list = []
        beauty_success_count = 0
        transform = transforms.ToTensor()
        for i in range(imgs.size(0)):
            to_pil = transforms.ToPILImage()
            pil_img = to_pil(imgs[i].cpu())

            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            buffer.seek(0)
            base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

            service = "fmu"
            host = "fmu.tencentcloudapi.com"
            region = "ap-beijing"
            version = "2019-12-13"
            action = "BeautifyPic"
            payload = {"Image": base64_data}

            algorithm = "TC3-HMAC-SHA256"
            timestamp = int(time.time())
            beijing_tz = timezone(timedelta(hours=8))
            beijing_time = datetime.fromtimestamp(timestamp, tz=beijing_tz)
            date = beijing_time.strftime("%Y-%m-%d")

            http_request_method = "POST"
            canonical_uri = "/"
            canonical_querystring = ""
            ct = "application/json; charset=utf-8"
            canonical_headers = "content-type:%s\nhost:%s\nx-tc-action:%s\n" % (
                ct,
                host,
                action.lower(),
            )
            signed_headers = "content-type;host;x-tc-action"
            hashed_request_payload = hashlib.sha256(
                json.dumps(payload).encode("utf-8")
            ).hexdigest()
            canonical_request = (
                http_request_method
                + "\n"
                + canonical_uri
                + "\n"
                + canonical_querystring
                + "\n"
                + canonical_headers
                + "\n"
                + signed_headers
                + "\n"
                + hashed_request_payload
            )

            credential_scope = date + "/" + service + "/" + "tc3_request"
            hashed_canonical_request = hashlib.sha256(
                canonical_request.encode("utf-8")
            ).hexdigest()
            string_to_sign = (
                algorithm
                + "\n"
                + str(timestamp)
                + "\n"
                + credential_scope
                + "\n"
                + hashed_canonical_request
            )

            secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
            secret_service = sign(secret_date, service)
            secret_signing = sign(secret_service, "tc3_request")
            signature = hmac.new(
                secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
            ).hexdigest()

            authorization = (
                algorithm
                + " "
                + "Credential="
                + secret_id
                + "/"
                + credential_scope
                + ", "
                + "SignedHeaders="
                + signed_headers
                + ", "
                + "Signature="
                + signature
            )

            headers = {
                "Authorization": authorization,
                "Content-Type": "application/json; charset=utf-8",
                "Host": host,
                "X-TC-Action": action,
                "X-TC-Timestamp": timestamp,
                "X-TC-Version": version,
            }
            if region:
                headers["X-TC-Region"] = region
            if token:
                headers["X-TC-Token"] = token

            fail_count = 0
            while fail_count < 3:
                try:
                    req = HTTPSConnection(host)
                    req.request(
                        "POST",
                        "/",
                        headers=headers,
                        body=json.dumps(payload).encode("utf-8"),
                    )
                    resp = req.getresponse()
                    text_data = resp.read()
                    json_data = json.loads(text_data)
                    image_bytes = base64.b64decode(json_data["Response"]["ResultImage"])
                    img = Image.open(BytesIO(image_bytes)).convert("RGB")
                    img_list.append(transform(img).cuda())
                    beauty_success_count += 1
                    break
                except Exception as e:
                    fail_count += 1
                    self.logger.error(e)
                    traceback.print_exc()

            if fail_count == 3:
                img_list.append(imgs[i].cuda())

        beauty_imgs = torch.stack(img_list, dim=0).cuda()
        return beauty_imgs, beauty_success_count


class Cloak:
    def __init__(self, logger, config, effectiveness):
        self.logger = logger
        self.config = config
        self.effectiveness = effectiveness

        self.cloak_dir = Path(self.config.third_party.dataset.cloak_dir)

        self.cloak_imgs = self._get_cloak_imgs()
        self.cloak_cache = self._cache_cloak_embeddings()

    def _cache_cloak_embeddings(self) -> dict:
        mtcnn = MTCNN(
            image_size=160,
            device="cuda",
            selection_method="largest",
            keep_all=False,
        )
        FaceVerification = InceptionResnetV1(
            classify=False, pretrained="vggface2"
        ).cuda()
        FaceVerification.eval()

        cloak_cache = {}
        with torch.no_grad():
            for k, v in self.cloak_imgs.items():
                imgs_ndarray = v.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                embeddings = []
                for i, img in enumerate(imgs_ndarray):
                    img_cropped = mtcnn(img)
                    if img_cropped is None:
                        self.logger.fatal(f"Cannot detect the face from {i}th {k}")
                    embedding = FaceVerification(img_cropped.unsqueeze(0).cuda())
                    embeddings.append(embedding)
                cloak_cache[k] = embeddings

            return cloak_cache

    def _hash_tensor(self, img: Tensor):
        return hash(tuple(img.view(-1).tolist()))

    def _get_cloak_imgs_path(self) -> dict:
        cloak_count = self.config.third_party.dataset.cloak_count

        male_imgs_path_list = [
            p
            for p in (self.cloak_dir / f"male_{cloak_count}").rglob("*")
            if p.is_file()
        ]
        female_imgs_path_list = [
            p
            for p in (self.cloak_dir / f"female_{cloak_count}").rglob("*")
            if p.is_file()
        ]
        mix_imgs_path_list = [
            p for p in (self.cloak_dir / f"mix_{cloak_count}").rglob("*") if p.is_file()
        ]

        return {
            "male": male_imgs_path_list,
            "female": female_imgs_path_list,
            "mix": mix_imgs_path_list,
        }

    def _load_imgs(self, imgs_path: list[Path]) -> Tensor:
        transform = (
            transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
            if OmegaConf.select(self.config, "third_party.dataset.use_224") is not None
            and self.config.third_party.dataset.use_224
            else transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
        )
        imgs_list = [
            cast(Tensor, transform(Image.open(path).convert("RGB")))
            for path in imgs_path
        ]

        imgs = torch.stack(imgs_list)

        return imgs.cuda()

    def _get_cloak_imgs(self) -> dict:
        cloak_imgs_path = self._get_cloak_imgs_path()

        return {
            "male": self._load_imgs(cloak_imgs_path["male"]),
            "female": self._load_imgs(cloak_imgs_path["female"]),
            "mix": self._load_imgs(cloak_imgs_path["mix"]),
        }

    def _check_imgs_gender_single(self, img: Tensor, key: str, secret: str) -> dict:
        result = {self._hash_tensor(img): "fail"}

        buffered = BytesIO()
        img_image = img * 255
        img_image = Image.fromarray(img_image.cpu().permute(1, 2, 0).byte().numpy())
        img_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        url = "https://api-us.faceplusplus.com/facepp/v3/detect"
        payload = {
            "api_key": key,
            "api_secret": secret,
            "image_base64": img_base64,
            "return_attributes": "gender",
        }

        fail_count = 0
        while fail_count < 10:
            try:
                response = requests.post(url, data=payload)
                if response.status_code == 200:
                    response_json = response.json()
                    if len(response_json["faces"]) > 1:
                        return result

                    gender = response_json["faces"][0]["attributes"]["gender"]["value"]
                    result[self._hash_tensor(img)] = gender.lower()
                    break
                elif response.status_code == 400:
                    try:
                        self.logger.info(response.json().get("time_used"))
                    except Exception:
                        self.logger.info("face++ returned status 400")
                    return result
                elif response.status_code == 403:
                    fail_count += 0.25
                    time.sleep(0.3)
                else:
                    fail_count += 1
                    self.logger.error(response)
            except BaseException as e:
                fail_count += 1
                self.logger.error(e)

        return result

    def _check_imgs_gender(self, imgs: Tensor):
        api_keys = self.config.evaluate.facepp.api_key
        api_secrets = self.config.evaluate.facepp.api_secret
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._check_imgs_gender_single,
                    imgs[i],
                    api_keys[i % len(api_keys)],
                    api_secrets[i % len(api_secrets)],
                )
                for i in range(imgs.shape[0])
            ]
            results = [future.result() for future in futures]

        imgs_gender = {}
        for result in results:
            imgs_gender.update(result)

        return imgs_gender

    def find_best_cloaks(self, imgs: Tensor) -> Tensor:
        imgs_gender = {}
        if not self.config.third_party.dataset.cloak_mix:
            imgs_gender = self._check_imgs_gender(imgs)

        imgs_ndarray = imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        best_cloaks = []
        for i in range(imgs.shape[0]):
            if self.config.third_party.dataset.cloak_mix:
                candidates = self.cloak_imgs["mix"]
                cache = self.cloak_cache["mix"]
            else:
                candidates = (
                    self.cloak_imgs["female"]
                    if imgs_gender[self._hash_tensor(imgs[i])] == "male"
                    else self.cloak_imgs["male"]
                )
                cache = (
                    self.cloak_cache["female"]
                    if imgs_gender[self._hash_tensor(imgs[i])] == "male"
                    else self.cloak_cache["male"]
                )

            # img_to_match = imgs[i].unsqueeze(0)
            # img_to_match = img_to_match.repeat(candidates.shape[0], 1, 1, 1)
            # same_identity = self.effectiveness.is_same_identity_via_facepp(
            #     img_to_match, candidates
            # )
            same_identity = [False] * candidates.shape[0]

            results = self.effectiveness.get_image_distance(imgs_ndarray[i], cache)
            distances = []
            for j, distance in enumerate(results):
                if (
                    distance is math.nan
                    or distance <= self.config.third_party.dataset.cloak_min_distance
                    or same_identity[j]
                ):
                    continue
                distances.append((distance, j))

            sorted_distances = sorted(distances)
            if len(sorted_distances) > self.config.third_party.dataset.cloak_index:
                best_cloak_idx = sorted_distances[
                    self.config.third_party.dataset.cloak_index
                ][1]
                best_cloaks.append(candidates[best_cloak_idx])
            else:
                best_cloaks.append(candidates[sorted_distances[-1][1]])

        return torch.stack(best_cloaks, dim=0)


class DistanceCloakSelector:
    def __init__(self, logger, config, effectiveness):
        self.logger = logger
        self.config = config
        self.effectiveness = effectiveness

        self.cloak_dir = Path(self.config.third_party.dataset.cloak_dir)

        self.cloak_imgs = self._get_cloak_imgs()
        self.cloak_embeddings = self._cache_cloak_embeddings()

    def _get_cloak_imgs_path(self) -> dict:
        male_imgs_path_list = [
            p for p in (self.cloak_dir / "male_full").rglob("*") if p.is_file()
        ]
        female_imgs_path_list = [
            p for p in (self.cloak_dir / "female_full").rglob("*") if p.is_file()
        ]
        mix_imgs_path_list = [
            p for p in (self.cloak_dir / "mix_full").rglob("*") if p.is_file()
        ]

        if self.config.third_party.dataset.cloak_mix:
            return {"mix": mix_imgs_path_list}
        else:
            return {"male": male_imgs_path_list, "female": female_imgs_path_list}

    def _load_imgs(self, imgs_path: list[Path]) -> Tensor:
        transform = (
            transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
            if OmegaConf.select(self.config, "third_party.dataset.use_224") is not None
            and self.config.third_party.dataset.use_224
            else transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
        )
        imgs_list = [
            cast(Tensor, transform(Image.open(path).convert("RGB")))
            for path in imgs_path
        ]

        imgs = torch.stack(imgs_list)

        return imgs.cuda()

    def _get_cloak_imgs(self) -> dict:
        cloak_imgs_path = self._get_cloak_imgs_path()
        return {k: self._load_imgs(v) for k, v in cloak_imgs_path.items()}

    def _cache_cloak_embeddings(self) -> dict:
        cloak_embeddings = {}
        for k, v in self.cloak_imgs.items():
            imgs_ndarray = v.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            embeddings = []
            for i, img in enumerate(imgs_ndarray):
                img_cropped = self.effectiveness.mtcnn(img)
                if img_cropped is None:
                    self.logger.fatal(f"Cannot detect the face from {i}-th cloak")
                    save_image(v, self.config.image_dir / "cloak_without_face.png")

                embedding = self.effectiveness.FaceVerification(
                    img_cropped.unsqueeze(0).cuda()
                )
                embedding = embedding.detach().cpu()
                embeddings.append(embedding)

            cloak_embeddings[k] = embeddings

        return cloak_embeddings

    def _hash_tensor(self, img: Tensor):
        return hash(tuple(img.view(-1).tolist()))

    def _check_imgs_gender_single(self, img: Tensor, key: str, secret: str) -> dict:
        result = {self._hash_tensor(img): "fail"}

        buffered = BytesIO()
        img_image = img * 255
        img_image = Image.fromarray(img_image.cpu().permute(1, 2, 0).byte().numpy())
        img_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        url = "https://api-us.faceplusplus.com/facepp/v3/detect"
        payload = {
            "api_key": key,
            "api_secret": secret,
            "image_base64": img_base64,
            "return_attributes": "gender",
        }

        fail_count = 0
        while fail_count < 10:
            try:
                response = requests.post(url, data=payload)
                if response.status_code == 200:
                    response = response.json()
                    if len(response["faces"]) > 1:
                        return result

                    gender = response["faces"][0]["attributes"]["gender"]["value"]
                    result[self._hash_tensor(img)] = gender.lower()
                    break
                elif response.status_code == 400:
                    return result
                elif response.status_code == 403:
                    fail_count += 0.25
                    time.sleep(0.3)
                else:
                    fail_count += 1
                    self.logger.error(response)
            except BaseException as e:
                fail_count += 1
                self.logger.error(e)

        return result

    def _check_imgs_gender(self, imgs: Tensor):
        api_keys = self.config.evaluate.facepp.api_key
        api_secrets = self.config.evaluate.facepp.api_secret
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._check_imgs_gender_single,
                    imgs[i],
                    api_keys[i % len(api_keys)],
                    api_secrets[i % len(api_secrets)],
                )
                for i in range(imgs.shape[0])
            ]
            results = [future.result() for future in futures]

        imgs_gender = {}
        for result in results:
            imgs_gender.update(result)

        return imgs_gender

    def find_best_cloaks(self, imgs: Tensor) -> Tensor:
        imgs_gender = {}
        if not self.config.third_party.dataset.cloak_mix:
            imgs_gender = self._check_imgs_gender(imgs)

        imgs_ndarray = imgs.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        best_cloak_imgs = []
        for i in range(imgs.shape[0]):
            if self.config.third_party.dataset.cloak_mix:
                candidates = self.cloak_imgs["mix"]
                embeddings = self.cloak_embeddings["mix"]
            else:
                candidates = (
                    self.cloak_imgs["female"]
                    if imgs_gender[self._hash_tensor(imgs[i])] == "male"  # type: ignore
                    else self.cloak_imgs["male"]
                )
                embeddings = (
                    self.cloak_embeddings["female"]
                    if imgs_gender[self._hash_tensor(imgs[i])] == "male"  # type: ignore
                    else self.cloak_embeddings["male"]
                )

            distances = self.effectiveness.get_image_distance(
                imgs_ndarray[i], [emb.cuda() for emb in embeddings]
            )
            cluster_distances = []
            for j, distance in enumerate(distances):
                if (
                    distance is math.nan
                    or distance <= self.config.third_party.dataset.cloak_min_distance
                ):
                    continue

                cluster_distances.append((distance, j))

            sorted_cluster_distances = sorted(cluster_distances)
            min_distance = float("inf")
            min_index = 0
            for distance, idx in sorted_cluster_distances:
                if (
                    abs(distance - self.config.third_party.dataset.cloak_distance)
                    < min_distance
                ):
                    min_distance = abs(
                        distance - self.config.third_party.dataset.cloak_distance
                    )
                    min_index = idx

            best_cloak_imgs.append(candidates[min_index])

        return torch.stack(best_cloak_imgs, dim=0)


class ScoreCalculator:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

    def calculate_score(
        self,
        iter_source_metric: dict,
        iter_context_metric: dict | None,
        metric: dict | None,
    ) -> dict:
        scores = {key: {"iter": 0, "total": 0} for key in iter_source_metric.keys()}
        identity_weight = self.config.evaluate.score.identity
        context_weight = self.config.evaluate.score.context
        trace_weight = self.config.evaluate.score.trace
        for key in scores.keys():
            iter_source_swap = (
                iter_source_metric[key]["pert_swap"][0]
                / iter_source_metric[key]["pert_swap"][1]
            )
            iter_trace = (
                iter_source_metric[key]["cloak"][0]
                / iter_source_metric[key]["cloak"][1]
            )

            if iter_context_metric is not None:
                iter_context_swap = (
                    iter_context_metric[key]["pert_swap"][0]
                    / iter_context_metric[key]["pert_swap"][1]
                )
            else:
                iter_context_swap = 1

            scores[key]["iter"] = (
                identity_weight * (1 - iter_source_swap)
                + context_weight * (1 - iter_context_swap)
                + trace_weight * iter_trace
            )

            if metric is not None:
                total_source_swap = (
                    metric["pert_source_effectiveness"][key]["pert_swap"][0]
                    / metric["pert_source_effectiveness"][key]["pert_swap"][1]
                )
                total_trace = (
                    metric["pert_source_effectiveness"][key]["cloak"][0]
                    / metric["pert_source_effectiveness"][key]["cloak"][1]
                )

                if iter_context_metric is not None:
                    total_context_swap = (
                        metric["pert_target_effectiveness"][key]["pert_swap"][0]
                        / metric["pert_target_effectiveness"][key]["pert_swap"][1]
                    )
                else:
                    total_context_swap = 1

                scores[key]["total"] = (
                    identity_weight * (1 - total_source_swap)
                    + context_weight * (1 - total_context_swap)
                    + trace_weight * total_trace
                )

        return scores
