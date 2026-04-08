import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic ConvBlock used in NullSwap:
        Conv -> BN -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeConvBlock(nn.Module):
    """
    DeConvBlock described in the paper:
        Upsample -> BN -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_dim = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SEResBottleneck(nn.Module):
    """
    ResNet bottleneck + SE attention.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        stride: int = 1,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        out_channels = bottleneck_channels * self.expansion

        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction=reduction)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)
        out = out + identity
        out = self.relu(out)
        return out


class IDExtractor(nn.Module):
    """
    Identity Extraction Module:
        1 ConvBlock + 4 SEResBottlenecks
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        bottleneck_channels: int = 64,
        num_blocks: int = 4,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        self.stem = ConvBlock(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        blocks = []
        in_ch = base_channels
        for _ in range(num_blocks):
            blocks.append(
                SEResBottleneck(
                    in_channels=in_ch,
                    bottleneck_channels=bottleneck_channels,
                    stride=1,
                    reduction=reduction,
                )
            )
            in_ch = bottleneck_channels * SEResBottleneck.expansion

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.maxpool(x)
        z_id = self.blocks(x)
        return z_id


class FeatureBlock(nn.Module):
    """
    Feature Block:
        3 ConvBlocks + 5 SEResBottlenecks
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: tuple[int, int, int] = (32, 64, 64),
        bottleneck_channels: int = 64,
        num_blocks: int = 5,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        assert len(conv_channels) == 3, "FeatureBlock requires exactly 3 ConvBlocks."

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=conv_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = ConvBlock(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = ConvBlock(
            in_channels=conv_channels[1],
            out_channels=conv_channels[2],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        blocks = []
        in_ch = conv_channels[2]
        for _ in range(num_blocks):
            blocks.append(
                SEResBottleneck(
                    in_channels=in_ch,
                    bottleneck_channels=bottleneck_channels,
                    stride=1,
                    reduction=reduction,
                )
            )
            in_ch = bottleneck_channels * SEResBottleneck.expansion

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        z_s = self.blocks(x)
        return z_s


class AdaptiveNoise(nn.Module):
    """
    Adaptive random noise injection for perturbation features.

    A practical implementation corresponding to the paper's idea:
        noise = beta * (alpha * randn + eta)

    where:
        alpha: learnable scalar
        beta : learnable scalar
        eta  : learnable noise tensor
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        alpha_init: float = 3.0,
        beta_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
        self.eta = nn.Parameter(torch.zeros(1, channels, height, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rand_noise = torch.randn_like(x)
        noise = self.beta * (self.alpha * rand_noise + self.eta)
        return x + noise


class PerturbationBlock(nn.Module):
    """
    Perturbation Block in NullSwap:
        1 ConvBlock for refinement
        + 3 SEResBottlenecks
        + adaptive random noise

    Input:
        z_id: [B, C, H, W]

    Output:
        z_p: perturbation-related feature
    """

    def __init__(
        self,
        in_channels: int,
        refine_channels: int = 64,
        bottleneck_channels: int = 64,
        num_blocks: int = 3,
        reduction: int = 16,
        feature_size: tuple[int, int] = (64, 64),
        alpha_init: float = 3.0,
        beta_init: float = 0.5,
    ) -> None:
        super().__init__()

        self.refine = ConvBlock(
            in_channels=in_channels,
            out_channels=refine_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        blocks = []
        in_ch = refine_channels
        for _ in range(num_blocks):
            blocks.append(
                SEResBottleneck(
                    in_channels=in_ch,
                    bottleneck_channels=bottleneck_channels,
                    stride=1,
                    reduction=reduction,
                )
            )
            in_ch = bottleneck_channels * SEResBottleneck.expansion

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = in_ch

        h, w = feature_size
        self.noise = AdaptiveNoise(
            channels=self.out_channels,
            height=h,
            width=w,
            alpha_init=alpha_init,
            beta_init=beta_init,
        )

    def forward(self, z_id: torch.Tensor) -> torch.Tensor:
        x = self.refine(z_id)
        x = self.blocks(x)
        z_p = self.noise(x)
        return z_p


class CloakingBlock(nn.Module):
    """
    Practical reconstruction module for a Figure-2-style NullSwap pipeline.

    Since the paper figure does not fully specify intermediate tensor sizes,
    this block explicitly aligns perturbation features to shallow image features
    before fusion, then predicts a bounded residual perturbation.
    """

    def __init__(
        self,
        feature_channels: int,
        perturb_channels: int,
        hidden_channels: int = 128,
        bottleneck_channels: int = 64,
        num_blocks: int = 3,
        reduction: int = 16,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        fused_channels = feature_channels + perturb_channels

        self.perturb_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.feature_reconstruct = SEResBottleneck(
            in_channels=fused_channels,
            bottleneck_channels=bottleneck_channels,
            stride=1,
            reduction=reduction,
        )
        feature_reconstruct_channels = bottleneck_channels * SEResBottleneck.expansion
        self.deconv1 = DeConvBlock(feature_reconstruct_channels, hidden_channels)
        self.conv = ConvBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.deconv2 = DeConvBlock(hidden_channels, hidden_channels)
        self.image_fuse = ConvBlock(
            in_channels=hidden_channels + out_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.image_reconstruct = nn.Sequential(
            ConvBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
            nn.Tanh(),
        )

    def forward(
        self,
        z_s: torch.Tensor,
        z_p: torch.Tensor,
        imgs: torch.Tensor,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z_p.shape[-2:] != z_s.shape[-2:]:
            z_p = F.interpolate(
                z_p,
                size=z_s.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused = torch.cat([z_s, self.perturb_scale * z_p], dim=1)
        fused = self.feature_reconstruct(fused)
        fused = self.deconv1(fused)
        fused = self.conv(fused)
        fused = self.deconv2(fused)

        if fused.shape[-2:] != imgs.shape[-2:]:
            fused = F.interpolate(
                fused,
                size=imgs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused = torch.cat([fused, imgs], dim=1)
        fused = self.image_fuse(fused)
        delta = self.image_reconstruct(fused) * epsilon
        cloak = torch.clamp(imgs + delta, 0.0, 1.0)
        return cloak, delta


class NullSwap(nn.Module):
    """
    A best-effort, runnable reconstruction of NullSwap Figure 2.

    Data flow:
        image -> IDExtractor -> PerturbationBlock
        image -> FeatureBlock
        (perturbation feature, shallow feature) -> CloakingBlock -> protected image
    """

    def __init__(
        self,
        image_channels: int = 3,
        epsilon: float = 8.0 / 255.0,
        id_base_channels: int = 64,
        id_bottleneck_channels: int = 64,
        id_num_blocks: int = 4,
        feature_conv_channels: tuple[int, int, int] = (32, 64, 64),
        feature_bottleneck_channels: int = 64,
        feature_num_blocks: int = 5,
        perturb_refine_channels: int = 64,
        perturb_bottleneck_channels: int = 64,
        perturb_num_blocks: int = 3,
        perturb_feature_size: tuple[int, int] = (128, 128),
        cloak_hidden_channels: int = 128,
        cloak_bottleneck_channels: int = 64,
        cloak_num_blocks: int = 3,
        discriminator_base_channels: int = 32,
        reduction: int = 16,
        alpha_init: float = 3.0,
        beta_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon

        self.id_extractor = IDExtractor(
            in_channels=image_channels,
            base_channels=id_base_channels,
            bottleneck_channels=id_bottleneck_channels,
            num_blocks=id_num_blocks,
            reduction=reduction,
        )
        self.feature_block = FeatureBlock(
            in_channels=image_channels,
            conv_channels=feature_conv_channels,
            bottleneck_channels=feature_bottleneck_channels,
            num_blocks=feature_num_blocks,
            reduction=reduction,
        )
        self.perturbation_block = PerturbationBlock(
            in_channels=self.id_extractor.out_channels,
            refine_channels=perturb_refine_channels,
            bottleneck_channels=perturb_bottleneck_channels,
            num_blocks=perturb_num_blocks,
            reduction=reduction,
            feature_size=perturb_feature_size,
            alpha_init=alpha_init,
            beta_init=beta_init,
        )
        self.cloaking_block = CloakingBlock(
            feature_channels=self.feature_block.out_channels,
            perturb_channels=self.perturbation_block.out_channels,
            hidden_channels=cloak_hidden_channels,
            bottleneck_channels=cloak_bottleneck_channels,
            num_blocks=cloak_num_blocks,
            reduction=reduction,
            out_channels=image_channels,
        )

    def forward(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        z_id = self.id_extractor(imgs)
        z_p = self.perturbation_block(z_id)
        z_s = self.feature_block(imgs)
        cloak, delta = self.cloaking_block(z_s, z_p, imgs, epsilon=self.epsilon)

        return {
            "cloak": cloak,
            "delta": delta,
            "z_id": z_id,
            "z_p": z_p,
            "z_s": z_s,
        }


class NullSwapDiscriminator(nn.Module):
    """
    Lightweight discriminator following the paper's description of ConvBlocks.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, base_channels, stride=2),
            ConvBlock(base_channels, base_channels * 2, stride=2),
            ConvBlock(base_channels * 2, base_channels * 4, stride=2),
            ConvBlock(base_channels * 4, base_channels * 8, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(base_channels * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return self.head(feat)


if __name__ == "__main__":
    image = torch.randn(10, 3, 256, 256)

    model = NullSwap()
    outputs = model(image)

    print("Input shape:", image.shape)
    for key, value in outputs.items():
        print(f"{key} shape:", value.shape)
