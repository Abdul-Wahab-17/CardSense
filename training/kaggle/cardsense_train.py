"""Kaggle GPU kernel: prepare the dataset, train YOLOv8n and write the weights and metrics to /kaggle/working/out.

Pushed by training/kaggle/run_on_kaggle.sh. Needs internet (to clone this repo and fetch yolov8n.pt) and the
dataset attached as an input; both are set in kernel-metadata.json.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("CARDSENSE_REPO", "https://github.com/Abdul-Wahab-17/CardSense.git")
REF = os.environ.get("CARDSENSE_REF", "main")
WORK = Path("/kaggle/working")


def run(*cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


run(sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.103", "pyyaml")
run("git", "clone", "--depth", "1", "--branch", REF, REPO, WORK / "CardSense")
repo = WORK / "CardSense"
run(sys.executable, repo / "training" / "prepare_dataset.py", "--dir", "/kaggle/input", "--out", WORK / "cards")
run(sys.executable, repo / "training" / "train_yolo.py", "--data", WORK / "cards" / "data.yaml",
    "--project", WORK / "runs", "--device", "0", "--workers", "4", cwd=repo)

out = WORK / "out"
out.mkdir(exist_ok=True)
run_dir = WORK / "runs" / "cardsense"
for name in ("weights/best.pt", "results.csv", "results.png", "confusion_matrix_normalized.png", "PR_curve.png",
             "test_metrics.json", "args.yaml"):
    shutil.copy(run_dir / name, out / Path(name).name)
(out / "git_ref.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True))
shutil.rmtree(WORK / "cards")  # keep the kernel output small
shutil.rmtree(WORK / "CardSense")
print("done:", sorted(p.name for p in out.iterdir()))
