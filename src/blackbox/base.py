from src.common_utils import cd, use_project
from src.simswap.options import build_simswap_test_options
from src.evaluate import Utility, Effectiveness, DistanceCloakSelector

import torch
import inspect
import warnings
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from types import MethodType
from omegaconf import OmegaConf, ListConfig
from pathlib import Path
from torchvision import transforms
from torch.serialization import add_safe_globals, SourceChangeWarning
from torch.nn.functional import mse_loss, l1_loss


class Base:
    def __init__(self, logger, config):
        super(Base, self).__init__()
        self.logger = logger
        self.config = config
        self.protection_method = self._get_protection_method()

        warnings.filterwarnings("ignore", category=SourceChangeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message=r".*The parameter 'pretrained' is deprecated since 0\.13.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Arguments other than a weight enum or `None` for 'weights' are deprecated since 0\.13.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*`rcond` parameter will change to the default of machine precision.*",
            category=FutureWarning,
            module=r".*matlab_cp2tform",
        )

        self.device = torch.device("cuda")

        # simswap
        self._simswap_normalize = transforms.Compose(
            [
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        simswap_root = Path(self.config.third_party.simswap_dir)
        with use_project([simswap_root]), cd(simswap_root):
            from models.models import create_model
            from models import arcface_models

            self.test_options = build_simswap_test_options(config)

            add_safe_globals(
                [
                    nn.Conv2d,
                    nn.Linear,
                    nn.BatchNorm2d,
                    nn.BatchNorm1d,
                    nn.ReLU,
                    nn.PReLU,
                    nn.Sigmoid,
                    nn.Dropout,
                    nn.Sequential,
                    nn.MaxPool2d,
                    nn.AdaptiveAvgPool2d,
                ]
            )

            for _, obj in inspect.getmembers(arcface_models):
                if inspect.isfunction(obj):
                    add_safe_globals([obj])
                elif inspect.isclass(obj):
                    from typing import Callable, Any, cast

                    add_safe_globals([cast(Callable[..., Any], obj)])

            self.target = create_model(self.test_options)
            self.target.cuda().eval()

            setattr(
                self.target.netG, "encoder", MethodType(self.encoder, self.target.netG)
            )

        # hififace
        hififace_root = Path(self.config.third_party.hififace_dir)
        with use_project([hififace_root]), cd(hififace_root):
            from hififace_pl import HifiFace

            config_path = Path(config.third_party.origin.hififace.config_path)
            if not config_path.is_absolute():
                config_path = hififace_root / config_path
            checkpoint_path = Path(config.third_party.origin.hififace.checkpoint_path)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(config.root_dir) / checkpoint_path

            origin_config = OmegaConf.load(config_path)

            self.net = HifiFace(origin_config)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.net.load_state_dict(checkpoint["state_dict"])
            self.net = self.net.eval().to(self.device)

        faceshifter_root = Path(self.config.third_party.faceshifter_dir)
        with use_project([faceshifter_root, faceshifter_root / "face_modules"]), cd(
            faceshifter_root
        ):
            from face_modules.model import Backbone
            from facenet_pytorch import InceptionResnetV1

            self.arcface = Backbone(50, 0.6, "ir_se").cuda()
            self.arcface.load_state_dict(
                torch.load(
                    config.third_party.origin.faceshifter.model_path,
                    weights_only=True,
                ),
                strict=False,
            )
            self.arcface = self.arcface.eval().cuda()
            for param in self.arcface.parameters():
                param.requires_grad_(False)

            self.facenet = InceptionResnetV1(
                classify=False,
                pretrained="vggface2",
            ).cuda()
            self.facenet = self.facenet.eval().cuda()
            for param in self.facenet.parameters():
                param.requires_grad_(False)

        diffface_root = Path(self.config.third_party.diffface_dir)
        with use_project([diffface_root]), cd(diffface_root):
            from models.parsing import BiSeNet
            from utils.module import SpecificNorm

            self.spNorm = SpecificNorm()
            self.netSeg = BiSeNet(n_classes=19).cuda()
            self.netSeg.load_state_dict(
                torch.load(
                    self.config.third_party.origin.face_parser_path,
                    map_location="cpu",
                    weights_only=False,
                )
            )
            self.netSeg = self.netSeg.eval().cuda()
            for param in self.netSeg.parameters():
                param.requires_grad_(False)

        self.defense_targets: dict[str, object] = {}

        def build_defense_target(config_name: str, model_class):
            return model_class(self.logger, self._resolve_third_party_config(config_name))

        for target_name in self.get_eval_target_names():
            if target_name == "simswap":
                from src.simswap.base import Base as SimSwap

                self.defense_targets["simswap"] = build_defense_target(
                    "simswap", SimSwap
                )
            elif target_name == "faceshifter":
                from src.faceshifter.base import Base as FaceShifter

                self.defense_targets["faceshifter"] = build_defense_target(
                    "faceshifter", FaceShifter
                )
            elif target_name == "diffface":
                from src.diffface.base import Base as DiffFace

                self.defense_targets["diffface"] = build_defense_target(
                    "diffface", DiffFace
                )
            elif target_name == "diffswap":
                from src.diffswap.base import Base as DiffSwap

                self.defense_targets["diffswap"] = build_defense_target(
                    "diffswap", DiffSwap
                )
            elif target_name == "uniface":
                from src.uniface.base import Base as UniFace

                self.defense_targets["uniface"] = build_defense_target(
                    "uniface", UniFace
                )
            elif target_name == "e4s":
                from src.e4s.base import Base as E4S

                self.defense_targets["e4s"] = build_defense_target("e4s", E4S)
            elif target_name == "infoswap":
                from src.infoswap.base import Base as InfoSwap

                self.defense_targets["infoswap"] = build_defense_target(
                    "infoswap", InfoSwap
                )
            elif target_name == "hififace":
                from src.hififace.base import Base as HifiFace

                self.defense_targets["hififace"] = build_defense_target(
                    "hififace", HifiFace
                )
            else:
                raise ValueError(f"Unsupported defense target: {target_name}")

        self._normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # common
        self.utility = Utility(logger, config)
        self.effectiveness = Effectiveness(logger, config)
        self.cloak = None
        if self.protection_method == "phantomseal":
            self.cloak = DistanceCloakSelector(logger, config, self.effectiveness)

        self.nullswap_generator = None
        if self.protection_method == "nullswap":
            self.nullswap_generator = self._build_nullswap_generator()

    def _get_protection_method(self) -> str:
        protection_config = getattr(self.config, "protection", None)
        method = str(getattr(protection_config, "method", "phantomseal")).lower()
        if method not in {"phantomseal", "nullswap"}:
            raise ValueError(f"Unsupported blackbox protection method: {method}")
        return method

    def _resolve_third_party_config(self, config_name: str):
        target_config = OmegaConf.create(
            OmegaConf.to_container(self.config, resolve=False)
        )
        target_config.third_party = OmegaConf.load(
            Path(self.config.root_dir) / f"config/third_party/{config_name}.yaml"
        )
        return OmegaConf.create(OmegaConf.to_container(target_config, resolve=True))

    def _build_nullswap_generator(self):
        from src.nullswap.model import NullSwap

        nullswap_config = self._resolve_third_party_config("nullswap")
        model_config = nullswap_config.third_party.model

        generator = NullSwap(
            image_channels=model_config.image_channels,
            epsilon=model_config.epsilon,
            id_base_channels=model_config.id_base_channels,
            id_bottleneck_channels=model_config.id_bottleneck_channels,
            id_num_blocks=model_config.id_num_blocks,
            feature_conv_channels=tuple(model_config.feature_conv_channels),
            feature_bottleneck_channels=model_config.feature_bottleneck_channels,
            feature_num_blocks=model_config.feature_num_blocks,
            perturb_refine_channels=model_config.perturb_refine_channels,
            perturb_bottleneck_channels=model_config.perturb_bottleneck_channels,
            perturb_num_blocks=model_config.perturb_num_blocks,
            perturb_feature_size=tuple(model_config.perturb_feature_size),
            cloak_hidden_channels=model_config.cloak_hidden_channels,
            cloak_bottleneck_channels=model_config.cloak_bottleneck_channels,
            cloak_num_blocks=model_config.cloak_num_blocks,
            reduction=model_config.reduction,
            alpha_init=model_config.alpha_init,
            beta_init=model_config.beta_init,
        ).to(self.device)

        checkpoint_path = Path(nullswap_config.third_party.defense.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"NullSwap checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        generator_state_dict = checkpoint.get("generator", checkpoint)
        generator.load_state_dict(generator_state_dict)
        generator.eval()
        for param in generator.parameters():
            param.requires_grad_(False)

        return generator

    def protect_with_nullswap(self, imgs: Tensor) -> Tensor:
        if self.nullswap_generator is None:
            raise RuntimeError("NullSwap generator is not initialized")

        noise_scale = float(
            getattr(getattr(self.config, "protection", None), "nullswap_noise_scale", 1.0)
        )
        if noise_scale < 0:
            raise ValueError("protection.nullswap_noise_scale must be non-negative")

        with torch.no_grad():
            outputs = self.nullswap_generator(imgs)
            nullswap_cloak = outputs["cloak"]

        perturbation = nullswap_cloak - imgs
        return torch.clamp(imgs + perturbation * noise_scale, 0.0, 1.0)

    @staticmethod
    def encoder(this, input):
        x = input

        x = this.first_layer(x)
        x = this.down1(x)
        x = this.down2(x)
        x = this.down3(x)
        if this.deep:
            x = this.down4(x)

        return x

    def get_arcface_embedding(self, imgs: Tensor) -> Tensor:
        return F.normalize(
            self.arcface(
                F.interpolate(
                    imgs[:, :, 19:237, 19:237],
                    (112, 112),
                    mode="bilinear",
                    align_corners=True,
                )
            ),
            dim=-1,
            p=2,
        )

    def get_facenet_embedding(self, imgs: Tensor) -> Tensor:
        imgs = F.interpolate(imgs, size=(160, 160), mode="bilinear", align_corners=False)
        imgs = (imgs - 0.5) / 0.5
        return F.normalize(self.facenet(imgs), dim=-1, p=2)

    def get_face_parse_logits(self, imgs: Tensor) -> Tensor:
        parser_in = F.interpolate(imgs, size=(512, 512), mode="bilinear", align_corners=False)
        parser_in = self._normalize(parser_in)
        logits = self.netSeg(self.spNorm(parser_in))[0]
        return F.interpolate(
            logits,
            size=imgs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def get_face_region_mask(self, parse_logits: Tensor) -> Tensor:
        parsing = parse_logits.argmax(dim=1, keepdim=True)
        mask = torch.zeros_like(parsing, dtype=torch.float32)
        for face_id in (1, 2, 3, 4, 5, 10, 11, 12, 13):
            mask = torch.where(
                parsing == face_id,
                torch.ones_like(mask),
                mask,
            )
        return mask

    def get_feature_region_mask(self, parse_logits: Tensor) -> Tensor:
        parsing = parse_logits.argmax(dim=1, keepdim=True)
        mask = torch.zeros_like(parsing, dtype=torch.float32)
        for face_id in (2, 3, 4, 5, 10, 11, 12, 13):
            mask = torch.where(
                parsing == face_id,
                torch.ones_like(mask),
                mask,
            )
        return mask

    def get_eval_target_names(self) -> list[str]:
        targets = getattr(self.config.third_party.defense, "targets", None)
        if targets is None:
            return [self.config.third_party.defense.target]
        if isinstance(targets, ListConfig):
            return list(targets)
        if isinstance(targets, (list, tuple)):
            return list(targets)
        return [targets]

    def swap_face(self, src_img: Tensor, tgt_img: Tensor, target_name: str | None = None) -> Tensor:
        target_name = target_name or self.config.third_party.defense.target
        if target_name == "diffface":
            assert src_img.shape[0] == tgt_img.shape[0]

            source_swap = []
            src_img = src_img.cpu()
            tgt_img = tgt_img.cpu()

            for i in range(src_img.size(0)):
                a = src_img[i : i + 1].contiguous().to(self.device, non_blocking=True)
                b = tgt_img[i : i + 1].contiguous().to(self.device, non_blocking=True)
                out = self.defense_targets[target_name].swap_face(a, b)
                source_swap.append(out.detach().cpu())
                del a, b, out

            return torch.cat(source_swap, dim=0).cuda().float()
        if target_name == "simswap":
            return self.defense_targets[target_name].swap_face(src_img, tgt_img).float()
        if target_name == "faceshifter":
            assert src_img.shape[0] == tgt_img.shape[0]
            results = []
            src_img = src_img.cpu()
            tgt_img = tgt_img.cpu()
            for i in range(src_img.size(0)):
                a = src_img[i : i + 1].contiguous().to(self.device, non_blocking=True)
                b = tgt_img[i : i + 1].contiguous().to(self.device, non_blocking=True)
                out = self.defense_targets[target_name].swap_face(a * 2 - 1, b * 2 - 1)
                results.append(out.detach().cpu())
                del a, b, out
            return torch.cat(results, dim=0).cuda().float()
        if target_name == "hififace":
            return self.defense_targets[target_name].swap_face(src_img, tgt_img).float()
        if target_name == "diffswap":
            diffswap_size = int(self.defense_targets[target_name].model_input_size)
            src_img = F.interpolate(
                src_img,
                size=(diffswap_size, diffswap_size),
                mode="bilinear",
                align_corners=False,
            )
            tgt_img = F.interpolate(
                tgt_img,
                size=(diffswap_size, diffswap_size),
                mode="bilinear",
                align_corners=False,
            )
            out = self.defense_targets[target_name].swap_face(src_img, tgt_img)
            return F.interpolate(
                out,
                size=(
                    self.config.third_party.dataset.image_size,
                    self.config.third_party.dataset.image_size,
                ),
                mode="bilinear",
                align_corners=False,
            ).float()
        if target_name == "uniface":
            pass
        elif target_name == "infoswap":
            src_img = F.interpolate(
                src_img, size=(512, 512), mode="bilinear", align_corners=False
            )
            tgt_img = F.interpolate(
                tgt_img, size=(512, 512), mode="bilinear", align_corners=False
            )
        elif target_name == "e4s":
            src_img = F.interpolate(
                src_img, size=(1024, 1024), mode="bilinear", align_corners=False
            )
            tgt_img = F.interpolate(
                tgt_img, size=(1024, 1024), mode="bilinear", align_corners=False
            )

        src_img = src_img * 2 - 1
        tgt_img = tgt_img * 2 - 1
        out = self.defense_targets[target_name].swap_face(src_img, tgt_img)
        if target_name != "uniface":
            out = ((out + 1) / 2).clamp(0, 1)
        return F.interpolate(
            out,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).float()

    def _normalize(self, image: Tensor) -> Tensor:
        return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
