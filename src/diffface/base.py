from optimization.image_editor import VGGDataset
from third_party.DiffFace.models.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from third_party.DiffFace.optimization.constants import (
    ASSETS_DIR_NAME,
    RANKED_RESULTS_DIR,
)
from third_party.DiffFace.optimization.augmentations import (
    ImageAugmentations,
    StructureAugmentations,
)
from third_party.DiffFace.utils.metrics_accumulator import MetricsAccumulator
from third_party.DiffFace.utils.module import SpecificNorm
from third_party.DiffFace.models.parsing import BiSeNet
from third_party.DiffFace.models.gaze_estimation.gaze_estimator import (
    Gaze_estimator,
)
from third_party.DiffFace.utils.module import SpecificNorm, cosin_metric
from third_party.DiffFace.utils.eye_crop import get_eye_coords
from third_party.DiffFace.models.gaze_estimation.gaze_estimator import Gaze_estimator
from src.utils import cd

import torch
import lpips
import warnings
import face_alignment
import os
import sys
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from importlib import import_module
from pathlib import Path
from torch.serialization import SourceChangeWarning, add_safe_globals
from torch.nn.parallel.data_parallel import DataParallel
from torchvision import transforms
from torch.nn.functional import mse_loss, l1_loss
from torchvision.utils import save_image

warnings.filterwarnings("ignore", category=SourceChangeWarning)

ROOT = Path(__file__).resolve().parents[1]
DIFFFACE_ROOT = ROOT.parent / "third_party" / "DiffFace"
sys.path.insert(0, str(DIFFFACE_ROOT))
ResNet = import_module("models.models").ResNet
add_safe_globals([DataParallel, ResNet])
_ORIG_TORCH_LOAD = torch.load


def _load_patch(*args, **kwargs):
    if (
        args
        and isinstance(args[0], str)
        and os.path.basename(args[0]) in ("Arcface_model_only.tar", "GazeEstimator.pt")
    ):
        kwargs.setdefault("weights_only", False)
        kwargs.setdefault("map_location", "cpu")
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _load_patch


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config

        self.project_root = Path(config.third_party.project_root)

        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": False,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": str(config.third_party.origin.timestep_respacing),
                "image_size": 256,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )

        self.device = "cuda:0"

        with cd(Path("third_party") / "DiffFace"):
            self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.load_state_dict(
            torch.load(
                f"{self.project_root}/checkpoints/Model.pt",
                map_location="cpu",
            )
        )
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()

        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

        self.image_structureAugmentations = StructureAugmentations(
            224, config.third_party.origin.aug_num // 2
        )
        self.image_augmentations = ImageAugmentations(
            112, config.third_party.origin.aug_num
        )
        self.metrics_accumulator = MetricsAccumulator()

        netArc_checkpoint = torch.load(
            f"{self.project_root}/checkpoints/Arcface_model_only.tar"
        )
        netArc = netArc_checkpoint["model"].module
        self.netArc = netArc.to(self.device).eval()

        self.spNorm = SpecificNorm()
        self.netSeg = BiSeNet(n_classes=19).to(self.device)
        self.netSeg.load_state_dict(
            torch.load(
                f"{self.project_root}/checkpoints/FaceParser.pth",
                map_location="cpu",
                weights_only=False,
            )
        )
        self.netSeg.eval()

        with cd(Path("third_party") / "DiffFace"):
            self.netGaze = Gaze_estimator().to(self.device)
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, flip_input=False
            )

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()
        return unscaled_timestep

    def makeMask(self, origin_mask):
        numpy = origin_mask.squeeze(0).detach().cpu().numpy().argmax(0)
        numpy = numpy.copy().astype(np.uint8)

        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        ids = [1, 2, 3, 4, 5, 10, 11, 12, 13]

        mask = np.zeros([256, 256])
        for id in ids:
            index = np.where(numpy == id)
            mask[index] = 1

        return np.expand_dims(mask, axis=0)

    def id_loss(self, x_in, targ, embedder):

        id_loss = torch.tensor(0)

        masked_input = x_in * self.mask
        # masked_input = x_in

        masked_input = F.interpolate(masked_input, (112, 112))
        targ = F.interpolate(targ, (112, 112))

        masked_input = self.image_augmentations(masked_input)

        src_id = embedder(masked_input)
        src_id = F.normalize(src_id, p=2, dim=1)

        targ_id = embedder(targ)
        targ_id = F.normalize(targ_id, p=2, dim=1)

        dists = 1 - cosin_metric(src_id, targ_id)

        # We want to sum over the averages
        for i in range(1):
            id_loss = id_loss + dists[i::1].mean()

        return id_loss

    def swap_face(self, src_img: Tensor, tgt_img: Tensor) -> Tensor:
        def cond_fn(x, t, img_id, y=None):
            with torch.enable_grad():
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)
                out = self.diffusion.p_mean_variance(
                    self.model, x, t, img_id, clip_denoised=False, model_kwargs={"y": y}
                )

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)

                loss = torch.tensor(0)

                # ID loss
                targ = src_img
                arc_src = (x_in + 1) / 2
                arc_src = transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )(arc_src)
                arc_targ = (targ + 1) / 2
                arc_targ = transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )(arc_targ)
                id_loss = self.id_loss(arc_src, arc_targ, self.netArc) * 6000

                loss = loss + id_loss
                self.metrics_accumulator.update_metric("id_loss", id_loss.item())

                # Segmentation loss
                src_mask = (x_in + 1) / 2
                # src_mask = transforms.Resize((512, 512))(src_mask)
                src_mask = transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                )(src_mask)
                targ_mask = (tgt_img + 1) / 2
                # targ_mask = transforms.Resize((512, 512))(targ_mask)
                targ_mask = transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                )(targ_mask)

                src_seg = self.netSeg(self.spNorm(src_mask))[0]
                # src_seg = transforms.Resize((256, 256))(src_seg)
                targ_seg = self.netSeg(self.spNorm(targ_mask))[0]
                # targ_seg = transforms.Resize((256, 256))(targ_seg)

                seg_loss = torch.tensor(0).to(self.device).float()

                # Attributes = [0, 'background', 1 'skin', 2 'r_brow', 3 'l_brow', 4 'r_eye', 5 'l_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
                ids = [1, 2, 3, 4, 5, 10, 11, 12, 13]

                for id in ids:
                    seg_loss += l1_loss(src_seg[0, id, :, :], targ_seg[0, id, :, :])
                    # seg_loss += mse_loss(src_seg[0,id,:,:], targ_seg[0,id,:,:])

                loss = loss + seg_loss * 200
                self.metrics_accumulator.update_metric("seg_loss", seg_loss.item())

                # Gaze loss
                if t < 50 and t > 10:
                    src_eye = x_in * 0.5 + 0.5
                    targ_eye = tgt_img
                    targ_eye = targ_eye * 0.5 + 0.5
                    llx, lly, lrx, lry, rlx, rly, rrx, rry = get_eye_coords(
                        self.fa, targ_eye
                    )

                    if llx is not None:
                        targ_left_eye = targ_eye[:, :, lly:lry, llx:lrx]
                        src_left_eye = src_eye[:, :, lly:lry, llx:lrx]
                        targ_right_eye = targ_eye[:, :, rly:rry, rlx:rrx]
                        src_right_eye = src_eye[:, :, rly:rry, rlx:rrx]
                        targ_left_eye = torch.mean(targ_left_eye, dim=1, keepdim=True)
                        src_left_eye = torch.mean(src_left_eye, dim=1, keepdim=True)
                        targ_right_eye = torch.mean(targ_right_eye, dim=1, keepdim=True)
                        src_right_eye = torch.mean(src_right_eye, dim=1, keepdim=True)
                        targ_left_gaze = self.netGaze(targ_left_eye.squeeze(0))
                        src_left_gaze = self.netGaze(src_left_eye.squeeze(0))
                        targ_right_gaze = self.netGaze(targ_right_eye.squeeze(0))
                        src_right_gaze = self.netGaze(src_right_eye.squeeze(0))
                        left_gaze_loss = l1_loss(targ_left_gaze, src_left_gaze)
                        right_gaze_loss = l1_loss(targ_right_gaze, src_right_gaze)
                        gaze_loss = (left_gaze_loss + right_gaze_loss) * 200

                        loss = loss + gaze_loss.sum()
                        self.metrics_accumulator.update_metric(
                            "gaze_loss", gaze_loss.item()
                        )
                    else:
                        print("no eye detected")

                # Background loss
                masked_background = x_in

                loss = loss + mse_loss(masked_background, tgt_img) * 50
                self.metrics_accumulator.update_metric(
                    "l2_loss", mse_loss(masked_background, tgt_img).item()
                )
                self.metrics_accumulator.update_metric(
                    "bg_loss",
                    mse_loss(masked_background, tgt_img * (1 - self.mask)).item(),
                )

                return -torch.autograd.grad(loss, x)[0]

        @torch.no_grad()
        def postprocess_fn(out, t):
            if self.mask is not None:
                background_stage_t = self.diffusion.q_sample(tgt_img, t[0])
                background_stage_t = torch.tile(
                    background_stage_t,
                    dims=(1, 1, 1, 1),
                )

                softmask = self.mask * (
                    min(
                        1,
                        (75 - (t.data + 1))
                        / (75.0 - self.config.third_party.origin.masking_threshold),
                    )
                )

                out["sample"] = out["sample"] * softmask + background_stage_t * (
                    1 - softmask
                )

            return out

        # Attributes = [0, 'background', 1 'skin', 2 'r_brow', 3 'l_brow', 4 'r_eye', 5 'l_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        color_list = [
            [0, 0, 0],
            [255, 0, 0],
            [0, 204, 204],
            [0, 0, 204],
            [255, 153, 51],
            [204, 0, 204],
            [0, 0, 0],
            [204, 0, 0],
            [102, 51, 0],
            [0, 0, 0],
            [76, 153, 0],
            [102, 204, 0],
            [255, 255, 0],
            [0, 0, 153],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        targ_mask = tgt_img.detach().clone()
        # targ_mask = transforms.Resize((512, 512))(targ_mask)
        targ_mask = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(
            targ_mask
        )
        targ_mask = self.netSeg(self.spNorm(targ_mask))[0]
        # targ_mask = transforms.Resize((256, 256))(targ_mask)
        parsing = targ_mask.squeeze(0).detach().cpu().numpy().argmax(0)
        targ_base = np.zeros((256, 256, 3))
        for idx, color in enumerate(color_list):
            targ_base[parsing == idx] = color
        targ_base /= 255.0
        mask = self.makeMask(targ_mask)
        self.mask = torch.from_numpy(mask).unsqueeze(0).to(self.device).float()
        self.mask.requires_grad_()

        src_img = src_img * 2.0 - 1.0
        tgt_img = tgt_img * 2.0 - 1.0

        # self.args.iterations_num: 8
        # for iteration_number in range(1):  # self.args.iterations_num: 8
        #     print(f"Start iterations {iteration_number}")
        sample_func = (
            self.diffusion.ddim_sample_loop_progressive
            if self.config.third_party.origin.model == "ddim"
            else self.diffusion.p_sample_loop_progressive
        )

        img = (src_img + 1) / 2
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        img_id = self.get_imgs_identity(img)

        samples = sample_func(
            self.model,
            (
                1,
                3,
                self.model_config["image_size"],
                self.model_config["image_size"],
            ),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=False,
            skip_timesteps=self.config.third_party.origin.skip_timesteps,
            init_image=tgt_img,
            postprocess_fn=postprocess_fn,
            randomize_class=True,
            img_id=img_id,
        )
        total_steps = (
            self.diffusion.num_timesteps
            - self.config.third_party.origin.skip_timesteps
            - 1
        )

        final_pred_tensor = None
        for j, sample in enumerate(samples):
            if j == total_steps:
                pred = sample["pred_xstart"][0]
                if self.mask is not None:
                    pred = tgt_img[0] * (1 - self.mask[0]) + pred * self.mask[0]
                final_pred_tensor = pred.add(1).div(2).clamp(0, 1)

        return final_pred_tensor.unsqueeze(0)  # type: ignore

    def get_imgs_identity(self, img: torch.Tensor) -> torch.Tensor:
        img_id = F.interpolate(img, (112, 112))
        img_id = self.netArc(img_id)
        img_id = F.normalize(img_id, p=2, dim=1)
        return img_id
