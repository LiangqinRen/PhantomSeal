import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(torch.cat([skip, x], dim=1))


class SimpleDenoiserUNet(nn.Module):
    """
    Lightweight residual U-Net for 256x256 RGB perturb -> clean restoration.

    The network predicts a residual correction and adds it to the input. This is
    a good default when the perturb image is already close to the clean image.
    Inputs and outputs are expected in [-1, 1].
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        depth: int = 2,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.depth = depth

        channels = [base_channels * (2**i) for i in range(depth + 1)]

        self.inc = ConvBlock(in_channels, channels[0])
        self.downs = nn.ModuleList(
            [
                DownBlock(channels[i], channels[i + 1])
                for i in range(depth)
            ]
        )
        self.ups = nn.ModuleList(
            [
                UpBlock(channels[i + 1], channels[i], channels[i])
                for i in reversed(range(depth))
            ]
        )
        self.out = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inc(x)
        skips = [y]
        for down in self.downs:
            y = down(y)
            skips.append(y)

        for up, skip in zip(self.ups, reversed(skips[:-1])):
            y = up(y, skip)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(
                y,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        y = self.out(y)

        if self.residual:
            y = x + y
        return torch.clamp(y, -1.0, 1.0)
