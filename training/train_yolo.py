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

    python training/train_yolo.py --data data/cards/data.yaml      # after training/prepare_dataset.py

Writes runs/<name>/weights/best.pt plus runs/<name>/test_metrics.json (held-out test split, evaluated once).
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


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


class GlareTrainer(DetectionTrainer):
    """DetectionTrainer that adds the glare augmentation to every training batch.

    Ultralytics has no per-batch callback that can modify images (an unknown event name is accepted silently
    and never fires), so the hook is the trainer's own preprocess_batch, which runs only in the training loop.
    """

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)  # moves to device, scales to [0, 1]
        batch["img"] = add_glare(batch["img"])
        return batch


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
    ap.add_argument("--device", default=None, help="'0' for the first GPU, 'cpu'; default: GPU if present")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0, help="train on this fraction of the images (smoke tests)")
    args = ap.parse_args()

    random.seed(args.seed)
    model = YOLO(args.weights)
    model.train(
        trainer=GlareTrainer, data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=args.seed,
        optimizer="AdamW", lr0=0.01, weight_decay=5e-4, cos_lr=True, warmup_epochs=3, patience=10,
        close_mosaic=10,                 # mosaic on for the first 40 of 50 epochs
        degrees=25, hsv_h=0.0, hsv_s=0.5, hsv_v=0.4, fliplr=0.5, scale=0.5, translate=0.1,
        project=args.project, name=args.name, exist_ok=True, deterministic=True,
        device=args.device, workers=args.workers, fraction=args.fraction,
    )
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    # Evaluate the best checkpoint on the held-out test split, once, after training has finished.
    best_model = YOLO(str(best))
    metrics = best_model.val(data=args.data, split="test", imgsz=args.imgsz, device=args.device, plots=True,
                             project=args.project, name=args.name + "_test", exist_ok=True)
    result = {
        "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "per_class_map50": {best_model.names[int(c)]: float(v) for c, v in zip(metrics.box.ap_class_index, metrics.box.ap50)},
        "epochs_requested": args.epochs, "imgsz": args.imgsz, "seed": args.seed,
        "ultralytics": __import__("ultralytics").__version__, "torch": torch.__version__,
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"test mAP@0.5 = {result['map50']:.4f}   mAP@0.5:0.95 = {result['map50_95']:.4f}")
    print(f"best weights: {best}")


if __name__ == "__main__":
    main()
