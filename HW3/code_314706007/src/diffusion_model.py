import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2
) -> torch.Tensor:
    """
    Linear schedule from the DDPM paper. Returns a (timesteps,) tensor.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal timestep embedding.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:  # zero pad if odd dim
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block using multi-head attention on flattened spatial tokens.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        x = x.view(b, c, h * w).transpose(1, 2)  # (b, hw, c)
        attn_out, _ = self.attn(x, x, x)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        return attn_out + residual


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Lightweight UNet backbone tailored for 28x28 RGB MNIST images.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        dropout: float = 0.1,
        use_attention_at: Iterable[int] | None = (7,),
    ):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Down blocks
        down_layers: List[nn.Module] = []
        in_ch = base_channels
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                down_layers.append(
                    ResidualBlock(
                        in_ch, out_ch, time_emb_dim=time_emb_dim, dropout=dropout
                    )
                )
                if use_attention_at and (28 // (2 ** level)) in use_attention_at:
                    down_layers.append(AttentionBlock(out_ch))
                in_ch = out_ch
            if level != len(channel_mults) - 1:
                down_layers.append(Downsample(in_ch))
        self.down = nn.ModuleList(down_layers)

        # Middle
        self.mid = nn.ModuleList(
            [
                ResidualBlock(in_ch, in_ch, time_emb_dim=time_emb_dim, dropout=dropout),
                AttentionBlock(in_ch),
                ResidualBlock(in_ch, in_ch, time_emb_dim=time_emb_dim, dropout=dropout),
            ]
        )

        # Up blocks
        up_layers: List[nn.Module] = []
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                up_layers.append(
                    ResidualBlock(
                        in_ch + out_ch,
                        out_ch,
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                    )
                )
                in_ch = out_ch
                if use_attention_at and (28 // (2 ** level)) in use_attention_at:
                    up_layers.append(AttentionBlock(in_ch))
            if level != 0:
                up_layers.append(Upsample(in_ch))
        self.up = nn.ModuleList(up_layers)

        self.final = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)
        x = self.init_conv(x)
        hs: List[torch.Tensor] = [x]  # store skip connections from residual blocks

        # Down path: push skips after residual blocks
        for layer in self.down:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
                hs.append(x)
            elif isinstance(layer, AttentionBlock):
                x = layer(x)
            else:  # Downsample
                x = layer(x)

        # Middle
        for layer in self.mid:
            x = layer(x, t_emb) if isinstance(layer, ResidualBlock) else layer(x)

        # Up path
        for layer in self.up:
            if isinstance(layer, ResidualBlock):
                skip = hs.pop()
                if skip.shape[2:] != x.shape[2:]:
                    raise RuntimeError(
                        f"Skip spatial {skip.shape[2:]} does not match current {x.shape[2:]}"
                    )
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb)
            else:
                x = layer(x)

        return self.final(x)


class GaussianDiffusion(nn.Module):
    """
    DDPM training objective and sampling utilities.
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 28,
        channels: int = 3,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps

        betas = linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Diffuse the data at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def p_losses(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted and true noise.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Single reverse diffusion step.
        """
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None, None]

        model_mean = sqrt_recip_alpha * (
            x - betas_t / sqrt_one_minus * self.model(x, t)
        )

        if (t == 0).all():
            return model_mean

        posterior_var = self.posterior_variance[t][:, None, None, None]
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple[int, int, int, int], device) -> torch.Tensor:
        img = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.p_sample_loop(
            (batch_size, self.channels, self.image_size, self.image_size), device=device
        )
