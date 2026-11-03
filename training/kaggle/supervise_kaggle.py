#!/usr/bin/env python3
"""Train the CardSense detector on a free Kaggle GPU, unattended, and download best.pt + metrics.

    export KAGGLE_API_TOKEN=...        # kaggle.com -> Settings -> API -> Create New Token
    pip install kaggle
    python training/kaggle/supervise_kaggle.py [--ref HEAD]

What it does, without anyone watching:
  1. builds a private "cardsense-offline" dataset (this repo at --ref, ultralytics wheels, yolov8n.pt, a font):
     Kaggle kernels have no internet until the account is phone verified;
  2. pushes the GPU kernel and polls it every 5 minutes;
  3. every Kaggle call is retried with exponential backoff and honours rate limits (HTTP 429);
  4. if the kernel dies part-way (timeout, GPU error), uploads its last checkpoint and resumes (max 3 resumes);
     a deterministic code error stops with the log tail in training/results/failed/ instead of retrying forever;
  5. downloads only the small result files (never the whole working directory).
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "training" / "results"
MODELS = REPO / "models"
TRANSIENT = re.compile(r"429|rate.?limit|too many requests|timed? ?out|temporary failure|name resolution|connection|"
                       r"50[234]|ssl|incompleteread|cuda|dataloader|killed|timeout", re.I)
DATASET = "cardsense-offline"
KERNEL = "cardsense-yolov8n"
POLL_SECONDS = 300


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def kaggle(*args, retries=8, check=True):
    """Run the kaggle CLI, retrying transient failures (rate limits, network) with exponential backoff."""
    delay = 30
    for attempt in range(1, retries + 1):
        p = subprocess.run(["kaggle", *map(str, args)], capture_output=True, text=True)
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode == 0 and "rate limited" not in out.lower():
            return out
        if attempt == retries or not TRANSIENT.search(out):
            if check:
                raise RuntimeError(f"kaggle {' '.join(map(str, args))} failed:\n{out[-800:]}")
            return out
        log(f"kaggle {args[0]} {args[1] if len(args) > 1 else ''}: transient ({out.strip()[-80:]!r}), retry in {delay}s")
        time.sleep(delay)
        delay = min(delay * 2, 1800)


def username():
    m = re.search(r"username:\s*(\S+)", kaggle("config", "view"))
    if not m:
        sys.exit("kaggle is not authenticated: export KAGGLE_API_TOKEN")
    return m.group(1)


def build_bundle(ref, user, dest: Path, resume: Path | None):
    (dest / "wheels").mkdir(parents=True)
    (dest / "repo").mkdir()
    archive = subprocess.run(["git", "-C", REPO, "archive", "--format=tar", ref], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", dest / "repo"], input=archive, check=True)
    (dest / "git_ref.txt").write_text(subprocess.check_output(["git", "-C", REPO, "rev-parse", ref], text=True))
    version = re.search(r"^ultralytics==(\S+)", (REPO / "requirements-worker.txt").read_text(), re.M).group(1)
    subprocess.run([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "-d", dest / "wheels",
                    f"ultralytics=={version}", "ultralytics-thop", "py-cpuinfo"], check=True)
    for url, name in (("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt", "yolov8n.pt"),
                      ("https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf", "Arial.ttf")):
        subprocess.run(["curl", "-fsSL", "--retry", "5", "--retry-delay", "10", "-o", dest / name, url], check=True)
    if resume:
        (dest / "resume").mkdir()
        shutil.copy(resume, dest / "resume" / "last.pt")
    (dest / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET, "id": f"{user}/{DATASET}", "licenses": [{"name": "other"}]}))


def upload_bundle(user, bundle, note):
    exists = "ready" in kaggle("datasets", "status", f"{user}/{DATASET}", check=False).lower()
    if exists:
        kaggle("datasets", "version", "-p", bundle, "-m", note, "-r", "tar")
    else:
        kaggle("datasets", "create", "-p", bundle, "-r", "tar")
    for _ in range(60):  # wait until Kaggle has processed it
        if "ready" in kaggle("datasets", "status", f"{user}/{DATASET}", check=False).lower():
            return
        time.sleep(20)
    raise RuntimeError("dataset did not become ready in 20 minutes")


def push_kernel(user):
    d = Path(tempfile.mkdtemp())
    shutil.copy(HERE / "cardsense_train.py", d)
    meta = json.loads((HERE / "kernel-metadata.json").read_text().replace("KAGGLE_USERNAME", user))
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    kaggle("kernels", "push", "-p", d)
    shutil.rmtree(d)


def wait_for_kernel(user):
    slug = f"{user}/{KERNEL}"
    while True:
        out = kaggle("kernels", "status", slug, check=False)
        state = re.search(r"KernelWorkerStatus\.(\w+)", out)
        state = state.group(1) if state else "UNKNOWN"
        log(f"kernel {state}")
        if state in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELED"):
            return state
        time.sleep(POLL_SECONDS)


def fetch(user, dest: Path, pattern):
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    kaggle("kernels", "output", f"{user}/{KERNEL}", "-p", dest, "--file-pattern", pattern)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--max-resumes", type=int, default=3)
    a = ap.parse_args()
    user = username()
    resume = None
    for attempt in range(a.max_resumes + 1):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            log(f"attempt {attempt + 1}: building bundle for {a.ref}" + (" with a checkpoint" if resume else ""))
            build_bundle(a.ref, user, bundle, resume)
            upload_bundle(user, bundle, f"bundle for {a.ref} (attempt {attempt + 1})")
        push_kernel(user)
        state = wait_for_kernel(user)
        out = RESULTS / "_fetch"
        fetch(user, out, r"(status\.json|last\.pt|best\.pt|results\.csv|results\.png|confusion_matrix_normalized\.png|"
                         r"PR_curve\.png|test_metrics\.json|args\.yaml|git_ref\.txt)$")
        status_file = out / "status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if status.get("ok") and (out / "best.pt").exists():
            RESULTS.mkdir(parents=True, exist_ok=True)
            MODELS.mkdir(exist_ok=True)
            shutil.move(out / "best.pt", MODELS / "best.pt")
            for f in out.iterdir():
                if f.name != "last.pt":
                    shutil.move(f, RESULTS / f.name)
            shutil.rmtree(out, ignore_errors=True)
            log("done: models/best.pt and training/results/ written")
            return 0
        tail = (status.get("error", "") + status.get("log_tail", "")) or f"kernel ended {state} without a status file"
        has_ckpt = (out / "last.pt").exists() and status.get("epochs_done", 0) > 0
        if attempt < a.max_resumes and has_ckpt and TRANSIENT.search(tail):
            log(f"interrupted after {status.get('epochs_done')} epochs; resuming from the checkpoint")
            resume = RESULTS / "last.pt"
            RESULTS.mkdir(parents=True, exist_ok=True)
            shutil.copy(out / "last.pt", resume)
            continue
        failed = RESULTS / "failed"
        shutil.rmtree(failed, ignore_errors=True)
        shutil.move(out, failed)
        (failed / "diagnosis.txt").write_text(tail)
        log(f"FAILED (not retryable): see {failed}/diagnosis.txt\n{tail[-600:]}")
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
