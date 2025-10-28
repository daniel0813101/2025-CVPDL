import yaml
from pathlib import Path
from typing import Optional

import math
import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f  # reuse Ultralytics building blocks


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        mx = self.max_pool(x).view(b, c)
        att = self.mlp(avg) + self.mlp(mx)
        att = self.sigmoid(att).view(b, c, 1, 1)
        return x * att


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg, mx], dim=1)
        att = self.sigmoid(self.conv(a))
        return x * att


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))


class C2f_CBAM(C2f):
    """
    Drop-in replacement for C2f that appends a CBAM on the fused output.
    Args same as C2f: (c1, c2, n=1, shortcut=False, g=1, e=0.5)
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=16, spatial_kernel=7):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.cbam = CBAM(c2, reduction=reduction, spatial_kernel=spatial_kernel)

    def forward(self, x):
        y = super().forward(x)
        return self.cbam(y)


class CBAMLayer(nn.Module):
    """Standalone CBAM wrapper for insertion after arbitrary feature blocks."""

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.cbam = CBAM(channels, reduction=reduction, spatial_kernel=spatial_kernel)

    def forward(self, x):
        return self.cbam(x)


def load_scaled_yaml(cfg_path: Path, *, scale_hint: Optional[str] = None) -> dict:
    """
    Load a YOLO configuration YAML and apply scale overrides based on the filename suffix.

    Ultralytics handles this internally for built-in configs, but custom YAMLs need manual scaling.
    For example, calling `yolo11m_CBAM.yaml` will apply the 'm' scale defined inside the base YAML.
    """

    cfg_path = Path(cfg_path)
    with cfg_path.open('r') as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError(f'YAML config at {cfg_path} is not a dictionary.')

    if scale_hint:
        scale_char = scale_hint
    else:
        prefix = cfg_path.stem.split('_', 1)[0]
        scale_char = prefix[-1] if prefix else ''
        if scale_char and scale_char not in {'n', 's', 'm', 'l', 'x'}:
            scale_char = ''

    if scale_char and 'scales' in cfg and isinstance(cfg['scales'], dict):
        if scale_char in cfg['scales']:
            depth, width, max_channels = cfg['scales'][scale_char]
            cfg['depth_multiple'] = depth
            cfg['width_multiple'] = width
            cfg['max_channels'] = max_channels

    width_mult = float(cfg.get('width_multiple', 1.0))
    max_channels = float(cfg.get('max_channels', float('inf')))

    def _scale_channels(ch_val: float) -> int:
        ch_val = min(ch_val, max_channels)
        return max(1, int(math.ceil((ch_val * width_mult) / 8.0) * 8))

    if width_mult != 1.0 or scale_char:
        for section in ('backbone', 'head'):
            for layer in cfg.get(section, []):
                if len(layer) >= 4 and layer[2] == 'CBAMLayer' and layer[3]:
                    layer[3][0] = _scale_channels(layer[3][0])

    return cfg
