"""Fine-tune YOLOv8n on the 52-class playing-card dataset.

Follows the recipe in docs/reports/CardSense_DL_Project_Report.pdf (section 4):
COCO-pretrained YOLOv8n, 640 px, AdamW (weight decay 5e-4), lr 0.01 with cosine decay and a 3-epoch
warmup, effective batch 32, 50 epochs with early stopping (patience 10), mosaic on for the first 40 epochs.

Card-aware augmentations from the report:
  * glare overlay      random bright ellipses (p = 0.4, major axis ~ N(80, 30) px, boost 0.3-0.7): implemented below
  * suit-preserving colour jitter  hue shift is disabled outright (hsv_h = 0), so hearts never drift towards clubs
  * rank-aware rotation  the report allows +-180 degrees for most ranks and only +-25 for 6s and 9s. Ultralytics
    applies one rotation range to every image, so this script uses +-25 degrees for all ranks: safe for 6/9,
    but less rotation variety than the report for other ranks. See the README for how this affects results.

    python training/train_yolo.py --data path/to/data.yaml
"""
import argparse
import math
import random
from pathlib import Path

import torch
from ultralytics import YOLO


def add_glare(images: torch.Tensor, p: float = 0.4) -> torch.Tensor:
    """Alpha-blend random bright ellipses onto a batch of images in [0, 1] (shape B x 3 x H x W)."""
    b, _, h, w = images.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=images.device), torch.arange(w, device=images.device), indexing="ij")
    for i in range(b):
        if random.random() > p:
            continue
        cx, cy = random.uniform(0, w), random.uniform(0, h)
        major = max(10.0, random.gauss(80, 30))
        minor = major * random.uniform(0.3, 1.0)
        theta = random.uniform(0, math.pi)
        x, y = xx - cx, yy - cy
        xr = x * math.cos(theta) + y * math.sin(theta)
        yr = -x * math.sin(theta) + y * math.cos(theta)
        mask = (((xr / major) ** 2 + (yr / minor) ** 2) <= 1).float()
        boost = random.uniform(0.3, 0.7)
        images[i] = images[i] * (1 - boost * mask) + boost * mask
    return images


def glare_callback(trainer):
    if trainer.model.training:
        trainer.batch["img"] = add_glare(trainer.batch["img"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml of the YOLO-format dataset")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="cardsense")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    model = YOLO(args.weights)
    model.add_callback("on_preprocess_batch", glare_callback)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=args.seed,
        optimizer="AdamW", lr0=0.01, weight_decay=5e-4, cos_lr=True, warmup_epochs=3, patience=10,
        close_mosaic=10,                 # mosaic on for the first 40 of 50 epochs
        degrees=25, hsv_h=0.0, hsv_s=0.5, hsv_v=0.4, fliplr=0.5, scale=0.5, translate=0.1,
        project=args.project, name=args.name, exist_ok=True,
    )
    metrics = model.val(split="test")  # the held-out test split, evaluated once at the end
    print(f"test mAP@0.5 = {metrics.box.map50:.4f}   mAP@0.5:0.95 = {metrics.box.map:.4f}")
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"best weights: {best}")


if __name__ == "__main__":
    main()
