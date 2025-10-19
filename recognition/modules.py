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