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
class GFNetBinary(nn.Module):
    def __init__(self, height, width, in_channels=1, num_classes=2, depth=4, channels=64):
        super().__init__()
        self.height = height
        self.width = width
  
        self.proj = nn.Conv2d(in_channels, channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            GFBlock(height, width, channels) for _ in range(depth)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes)
        )

    def forward(self, x):
        # x: (B, in_channels, H, W)
        x = self.proj(x)  # -> (B, channels, H, W)
        for b in self.blocks:
            x = b(x)
        out = self.head(x)
        return out
def build_model(in_channels=1, num_classes=2, height=224, width=224):
    return GFNetBinary(height=height, width=width,
                       in_channels=in_channels, num_classes=num_classes)