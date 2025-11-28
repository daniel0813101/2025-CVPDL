## Diffusion-Based MNIST Generation

This project trains a lightweight diffusion model (UNet backbone + Gaussian Diffusion) to generate MNIST-like RGB digits. Training happens in `src/image_generation_written_digits.ipynb` and images can be produced later through `src/generate_images.py`.

### 1. Environment Setup
- **Python**: 3.10+ is recommended (CUDA builds of PyTorch require matching toolkit/drivers).
- **Create a virtualenv**
  ```bash
  cd 2025-CVPDL/HW3/code_314706007
  python -m venv .venv
  source .venv/bin/activate
  ```
- **Install dependencies**
  ```bash
  pip install --upgrade pip
  pip install -r requirement.txt
  ```
  If you have a CUDA-capable GPU, install the matching `torch`/`torchvision` wheels from https://pytorch.org/get-started/locally/ before running training for far better performance.

### 2. Dataset Location
- The notebook expects pre-rendered RGB MNIST PNGs under `HW3/dataset/mnist` relative to the repo root (`DATA_ROOT = Path('../../dataset/mnist')` inside the notebook).
- Each PNG is assumed to be 28×28 with three channels and arbitrarily named (e.g., `00001.png`). If you store the dataset somewhere else, update the `DATA_ROOT` constant in the notebook to point at your custom directory.
- Folder structure example:
  ```
  2025-CVPDL/
  └── HW3/
      ├── code_314706007/
      └── dataset/
          └── mnist/
              ├── 00001.png
              ├── 00002.png
              └── ...
  ```

### 3. Run Training
1. Launch Jupyter (VS Code, Jupyter Lab, etc.) rooted at `code_314706007`.
2. Open `src/image_generation_written_digits.ipynb`.
3. Execute the cells in order:
   - Imports, hyperparameters, and dataloader setup.
   - Model instantiation (prints the UNet/diffusion architecture).
   - Training loop cell to start optimization (logs written under `src/logs/<timestamp>`).
4. Optional diagnostics:
   - The notebook records `train_log.csv` per run; after training finishes, execute the final plotting cell to render and save `train_loss_curve.png` beside the log.
5. Outputs:
   - Samples, sampling-progress grids, and checkpoints live in `src/logs/<timestamp>/`.
   - Checkpoints include both raw and EMA weights, optimizer, GradScaler state, and metadata (`run_id`, `epoch`, `global_step`).

### 4. Generate Images from a Checkpoint
Run the helper script once training produced a checkpoint:
```bash
cd 2025-CVPDL/HW3/code_314706007/src
python generate_images.py \
  --ckpt logs/<run_id>/checkpoint_epoch_050.pt \
  --out_dir ../generated \
  --num_images 500 \
  --batch_size 64 \
  --use_ema
```
- `--ckpt`: path to the checkpoint created by the notebook.
- `--use_ema`: recommended for smoother samples (uses EMA weights if present).
- `--out_dir`: base directory where a timestamped folder is created (e.g., `../generated/20251128-193000/`).
- Override `--image_size` or `--channels` only if you retrained the diffusion model with different values.

### 5. Tips
- Keep an eye on GPU memory; adjust `BATCH_SIZE`, `base_channels`, or `channel_mults` in the notebook to fit your hardware.
- When resuming training, load a previous checkpoint within the notebook by pointing to `RUN_DIR / checkpoint_epoch_xxx.pt` before executing the training loop cell.
