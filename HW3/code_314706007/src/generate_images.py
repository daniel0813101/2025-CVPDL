import argparse
import math
from pathlib import Path

import torch
from torchvision import utils

from diffusion_model import GaussianDiffusion, UNet


def build_model(image_size: int, channels: int) -> GaussianDiffusion:
    unet = UNet(
        in_channels=channels,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        time_emb_dim=256,
        dropout=0.1,
        use_attention_at=(7,),
    )
    diffusion = GaussianDiffusion(
        model=unet,
        image_size=image_size,
        channels=channels,
        timesteps=1000,
    )
    return diffusion


def load_checkpoint(diffusion: GaussianDiffusion, ckpt_path: Path, use_ema: bool) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    target_state = checkpoint.get("ema") if use_ema else checkpoint.get("diffusion")
    if target_state is None:
        raise ValueError("Requested weights not found in checkpoint")
    diffusion.load_state_dict(target_state)


@torch.no_grad()
def generate_images(
    diffusion: GaussianDiffusion,
    device: torch.device,
    out_dir: Path,
    num_images: int,
    batch_size: int,
    start_index: int = 1,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    total_batches = math.ceil(num_images / batch_size)
    idx = start_index
    diffusion.eval()
    for _ in range(total_batches):
        cur_batch = min(batch_size, num_images - (idx - start_index))
        samples = diffusion.sample(cur_batch, device=device)
        samples = (samples.clamp(-1, 1) + 1) * 0.5  # [0,1]
        for j in range(cur_batch):
            utils.save_image(samples[j], out_dir / f"{idx:05d}.png")
            idx += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MNIST-like images using a trained diffusion model"
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to checkpoint file (expects diffusion/ema state dicts)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./generated"),
        help="Output directory for generated PNGs",
    )
    parser.add_argument("--num_images", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA weights if available in checkpoint",
    )
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--channels", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    diffusion = build_model(args.image_size, args.channels).to(device)
    load_checkpoint(diffusion, args.ckpt, use_ema=args.use_ema)
    generate_images(
        diffusion=diffusion,
        device=device,
        out_dir=args.out_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
    )
    print(f"Generated {args.num_images} images to {args.out_dir}")


if __name__ == "__main__":
    main()
