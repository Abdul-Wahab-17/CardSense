"""Kaggle GPU kernel: prepare the dataset, train YOLOv8n and write the weights and metrics to /kaggle/working/out.

Pushed by run_on_kaggle.sh. It runs fully offline: Kaggle kernels have no internet until the account is phone
verified, so the code, the ultralytics wheels and the COCO-pretrained yolov8n.pt come from a private input
dataset ("cardsense-offline") that run_on_kaggle.sh builds, next to the public card dataset.
"""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")


def run(*cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def find(name):
    """First file or directory called `name` under /kaggle/input (follows symlinks)."""
    seen = []
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        seen.extend(os.path.join(root, n) for n in dirs + files)
        if name in dirs + files:
            return Path(root) / name
    raise SystemExit(f"{name} not found under /kaggle/input; contents: {seen[:60]}")


bundle = find("git_ref.txt").parent
run(sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *sorted(glob.glob(str(bundle / "wheels" / "*.whl"))))
font_dir = Path.home() / ".config" / "Ultralytics"  # ultralytics would try to download this font
font_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(bundle / "Arial.ttf", font_dir / "Arial.ttf")

repo = bundle / "repo"

card_data = find("README.roboflow.txt").parent  # the public dataset's root (data.yaml + train/valid/test)
run(sys.executable, repo / "training" / "prepare_dataset.py", "--dir", card_data, "--out", WORK / "cards")
run(sys.executable, repo / "training" / "train_yolo.py", "--data", WORK / "cards" / "data.yaml",
    "--weights", bundle / "yolov8n.pt", "--project", WORK / "runs", "--device", "0", "--workers", "4", cwd=repo)

out = WORK / "out"
out.mkdir(exist_ok=True)
run_dir = WORK / "runs" / "cardsense"
for name in ("weights/best.pt", "results.csv", "results.png", "confusion_matrix_normalized.png", "PR_curve.png",
             "test_metrics.json", "args.yaml"):
    shutil.copy(run_dir / name, out / Path(name).name)
shutil.copy(bundle / "git_ref.txt", out / "git_ref.txt")
shutil.rmtree(WORK / "cards")  # keep the kernel output small
print("done:", sorted(p.name for p in out.iterdir()))
