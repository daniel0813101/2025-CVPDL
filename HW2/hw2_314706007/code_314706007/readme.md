# Long-Tailed Object Detection with Attention

This document explains how to reproduce the workflow implemented in
`src/longtail_object_detection_attention.ipynb`. The notebook trains a YOLOv11
detector augmented with CBAM attention layers and long-tail rebalancing tricks,
then exports a submission CSV for the Taica CVPDL HW2 dataset.

## TL;DR
- Make sure dataset is in correct path (same level as this root folder: hw2_314706007/)
- Select Kernel
- pip install -r reuirement.txt
- Run all
- adjust batch_size, image_size to fit in 12GB vram

## 1. Prerequisites

- Python 3.10+ and a CUDA-capable GPU with at least 16 GB memory.
- Clone or unpack this repository so that the `src/` directory, `cbam.py`, and
  `yolo11_CBAM.yaml` remain in place.
- Install dependencies:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirement.txt
  ```

  The key packages are `ultralytics>=8.3.209` for YOLOv11 and `torch==2.8.0`.

## 2. Dataset Layout

1. Download the CVPDL HW2 dataset and place it under
   `dataset/taica-cvpdl-2025-hw-2/CVPDL_hw2/CVPDL_hw2` relative to the notebook.
   The folder must contain `train/images`, `train/labels`, and `test/images`.
2. If your structure differs, edit the `DATA_ROOT` constant in the first code
   cell of the notebook so it points to the dataset root.
3. Optional: create a lightweight symlink instead of copying the dataset when
   working on different machines.

## 3. Notebook Walkthrough

Launch Jupyter Lab/Notebook in the repository root and open
`src/longtail_object_detection_attention.ipynb`. Execute the notebook top to
bottom after addressing every `# TODO` marker.

### Step 1 – Experiment bootstrap

- The first cell defines experiment metadata (run directories, seeds, split
  ratio) and helper utilities for reading YOLO-style labels.
- Required user edits:
  - `TrainConfig.model_yaml`: set to the architecture you want to train, e.g.
    `yolo11_CBAM.yaml`.
  - `TrainConfig.batch_size`: tune according to available GPU memory.
  - `TrainConfig.epochs`: increase if the model has not converged.

The code creates an experiment root at
`src/artifacts/yolov11_attention/<timestamp>/` where all subsequent assets are
stored.

### Step 2 – Class distribution analysis

- The notebook parses every training label file, aggregates per-class box and
  image counts, and prints summary statistics.
- This informs the long-tail imbalance and prepares metadata used for sampling.

### Step 3 – Attention modules registration

- `cbam.py` defines CBAM attention layers that plug into YOLOv11.
- The notebook registers these modules with Ultralytics by assigning them to
  `ultralytics.nn.modules` and `ultralytics.nn.tasks`.
- Confirm that `yolo11_CBAM.yaml` references the registered layers (e.g.
  `C2f_CBAM`). Adjust the YAML if you introduce a different depth/width scale.

### Step 4 – Data rebalancing helpers

- Utility functions compute per-class frequencies and apply oversampling and
  undersampling to fight the long tail.
- `prepare_training_dataset` copies images/labels into a fresh YOLO dataset
  structure (`images/train`, `labels/train`, etc.), applies rebalancing, and
  emits a YAML configuration describing the generated dataset.
- Adjust the oversampling target or disable undersampling in the function call
  if your experiments require different rebalancing strengths.

### Step 5 – Training configuration

- `TrainConfig` wraps Ultralytics arguments such as `imgsz`, `optimizer`, and
  augmentation knobs. Inspect the dataclass for defaults you may want to adjust
  (e.g. `lr0`, `lrf`, `weight_decay`).
- `ALPHA_SCHEDULE` is used to anneal the classification focal-loss gamma during
  training. Modify `start`, `end`, and `transition_epochs` to experiment with
  alternative schedules.

### Step 6 – Launching the pipeline

Executing the final cell calls `run_single_stage_pipeline(...)`, which performs:

1. Dataset preparation with rebalancing (`resample=True`, `oversample_tail=True`,
   `undersample_head=True`).
2. Ultralytics YOLO training using the attention-enhanced model definition.
3. Validation evaluation and automatic checkpointing of the best model.
4. Test-set inference with the best weights and creation of a submission CSV.

Monitor the console output for Ultralytics logs. Training metrics are saved to
`results.csv` within the run directory, and the best weights land in
`weights/best.pt`.

### Step 7 – Reviewing outputs

After the pipeline finishes, the notebook prints paths similar to:

- Experiment directory:
  `src/artifacts/yolov11_attention/<timestamp>/`
- Training run:
  `.../run/train/`
- Best weights: `.../weights/best.pt`
- Submission CSV: `.../infer_submission.csv`

The notebook also reads the final row from `results.csv` to display validation
metrics such as mAP and F1.

## 4. Tips and Troubleshooting

- **GPU memory**: reduce `TrainConfig.batch_size` or `imgsz` if you encounter
  CUDA OOM errors.
- **Augmentation workers**: set `TrainConfig.workers` to match your CPU cores.
  The notebook defaults to `0` for deterministic behavior; increasing it speeds
  up data loading.
- **Model definition**: ensure `yolo11_CBAM.yaml` lives in `src/` (or provide an
  absolute path) so Ultralytics can find it.
- **Reproducibility**: the global `SEED` is fixed to 11. Adjust if you need
  different random splits.

Following these steps should fully reproduce the attention-augmented long-tail
object detection experiment described in the notebook.
