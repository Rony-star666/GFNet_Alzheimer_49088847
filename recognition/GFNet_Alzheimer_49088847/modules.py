import torch
import torch.nn as nn
import torch.fft


class GlobalFilter(nn.Module):
    def __init__(self, height, width, channels):
        super().__init__()
        freq_w = width // 2 + 1
        self.complex_weight = nn.Parameter(
            torch.randn(height, freq_w, channels, 2) * 0.02
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)                # (B,H,W,C)
        x_freq = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)  # (H, freq_W, C)
        x_freq = x_freq * weight
        x_ifft = torch.fft.irfft2(x_freq, s=(H, W), dim=(1, 2), norm="ortho")
        x_out = x_ifft.permute(0, 3, 1, 2)       # (B,C,H,W)
        return x_out


class GFBlock(nn.Module):
    def __init__(self, height, width, channels, mlp_ratio=4.0, dropout=0.3):
        super().__init__()
        self.gf = GlobalFilter(height, width, channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B,C,H,W)
        identity = x
        y = x.permute(0, 2, 3, 1)               # (B,H,W,C)
        y = self.norm1(y).permute(0, 3, 1, 2)   # (B,C,H,W)
        y = self.gf(y)
        x = y + identity

        y2 = x.permute(0, 2, 3, 1)
        y2 = self.norm2(y2)
        y2 = self.mlp(y2)
        y2 = y2.permute(0, 3, 1, 2)
        return x + y2


class GFNetBinary(nn.Module):
    def __init__(self, height, width, in_channels=1, num_classes=2,
                 depth=3, channels=48, dropout=0.3):
        super().__init__()
        self.height = height
        self.width = width

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),    # GroupNorm 对小 batch 更稳定
            nn.GELU(),
            nn.Dropout2d(dropout)
        )

        self.blocks = nn.ModuleList([
            GFBlock(height, width, channels, dropout=dropout) for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes)
        )

    def forward(self, x):
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def build_model(in_channels=1, num_classes=2, height=224, width=224):
    return GFNetBinary(height=height, width=width,
                       in_channels=in_channels, num_classes=num_classes,
                       depth=3, channels=48, dropout=0.3)
