"""Kaggle GPU kernel: prepare the dataset, train YOLOv8n and write the weights and metrics to /kaggle/working/out.

Pushed by the supervisor (supervise_kaggle.py). It runs fully offline: Kaggle kernels have no internet until the
account is phone verified, so the code, the ultralytics wheels and the COCO-pretrained yolov8n.pt come from a private
input dataset ("cardsense-offline") built by the supervisor, next to the public card dataset.

Whatever happens, /kaggle/working/out ends up small and readable: status.json (ok / error + log tail + epochs done),
the last checkpoint (so an interrupted run can resume) and, on success, best.pt and the metrics.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "out"
RUN = WORK / "runs" / "cardsense"
LOG = WORK / "kernel_log.txt"
status = {"ok": False, "started": time.strftime("%Y-%m-%d %H:%M:%S")}


def run(*cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    with open(LOG, "ab") as lf:
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            lf.write(line)
        if p.wait():
            raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}")


def find(name):
    """First file or directory called `name` under /kaggle/input (follows symlinks)."""
    seen = []
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        seen.extend(os.path.join(root, n) for n in dirs + files)
        if name in dirs + files:
            return Path(root) / name
    raise SystemExit(f"{name} not found under /kaggle/input; contents: {seen[:60]}")


def epochs_done():
    try:
        return max(0, len((RUN / "results.csv").read_text().strip().splitlines()) - 1)
    except FileNotFoundError:
        return 0


OUT.mkdir(parents=True, exist_ok=True)
try:
    bundle = find("git_ref.txt").parent
    run(sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
        *sorted(glob.glob(str(bundle / "wheels" / "*.whl"))))
    font_dir = Path.home() / ".config" / "Ultralytics"  # ultralytics would try to download this font
    font_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(bundle / "Arial.ttf", font_dir / "Arial.ttf")

    repo = bundle / "repo"
    card_data = find("README.roboflow.txt").parent  # the public dataset's root (data.yaml + train/valid/test)
    run(sys.executable, repo / "training" / "prepare_dataset.py", "--dir", card_data, "--out", WORK / "cards")

    train = [sys.executable, repo / "training" / "train_yolo.py", "--data", WORK / "cards" / "data.yaml",
             "--weights", bundle / "yolov8n.pt", "--project", WORK / "runs", "--device", "0", "--workers", "4"]
    resume = bundle / "resume" / "last.pt"
    if resume.exists():  # the supervisor uploaded a checkpoint from an interrupted attempt
        (RUN / "weights").mkdir(parents=True, exist_ok=True)
        shutil.copy(resume, RUN / "weights" / "last.pt")
        train += ["--resume-from", RUN / "weights" / "last.pt"]
        status["resumed_from_epoch"] = "checkpoint"
    run(*train, cwd=repo)

    for name in ("weights/best.pt", "results.csv", "results.png", "confusion_matrix_normalized.png", "PR_curve.png",
                 "test_metrics.json", "args.yaml"):
        shutil.copy(RUN / name, OUT / Path(name).name)
    shutil.copy(bundle / "git_ref.txt", OUT / "git_ref.txt")
    status["ok"] = True
except BaseException:  # noqa: B036 - also record SystemExit/KeyboardInterrupt (Kaggle timeouts)
    status["error"] = traceback.format_exc()[-3000:]
    raise
finally:
    status.update(epochs_done=epochs_done(), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        status["log_tail"] = "".join(LOG.read_text(errors="replace").splitlines(True)[-60:])
    except FileNotFoundError:
        pass
    last = RUN / "weights" / "last.pt"
    if last.exists():  # lets the supervisor resume instead of starting over
        shutil.copy(last, OUT / "last.pt")
    (OUT / "status.json").write_text(json.dumps(status, indent=2))
    shutil.rmtree(WORK / "cards", ignore_errors=True)  # the 800 MB working copy must never be part of the output
    shutil.rmtree(WORK / "runs", ignore_errors=True)
    print("status:", status["ok"], "epochs done:", status["epochs_done"])
