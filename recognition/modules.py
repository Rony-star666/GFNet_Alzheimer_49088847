import torch
import torch.nn as nn

class GlobalFilter(nn.Module):
    def __init__(self, height, width, channels):
        super().__init__()
        freq_w = width // 2 + 1
        self.complex_weight = nn.Parameter(
            torch.randn(height, freq_w, channels, 2) * 0.02
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x_freq = torch.fft.rfft2(x, dim=(1,2), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_freq = x_freq * weight
        x_ifft = torch.fft.irfft2(x_freq, s=(H, W), dim=(1,2), norm="ortho")
        x_out = x_ifft.permute(0, 3, 1, 2)
        return x_out
class GFBlock(nn.Module):
    def __init__(self, height, width, channels, mlp_ratio=4.0):
        super().__init__()
        self.gf = GlobalFilter(height, width, channels)
        hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )
        # 改成对最后一维 (C) 归一化
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, H, W)
        identity = x
        x = x.permute(0, 2, 3, 1)  # -> (B, H, W, C)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # -> (B, C, H, W)
        x = self.gf(x)
        x = x + identity

        y = x.permute(0, 2, 3, 1)  # -> (B, H, W, C)
        y = self.norm2(y)
        y = self.mlp(y)
        y = y.permute(0, 3, 1, 2)
        return x + y