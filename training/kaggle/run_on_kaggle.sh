#!/usr/bin/env bash
# Train the detector on a free Kaggle GPU and download best.pt + metrics into models/ and training/results/.
#
#   export KAGGLE_API_TOKEN=...          # kaggle.com -> Settings -> API -> Create New Token
#   pip install kaggle
#   training/kaggle/run_on_kaggle.sh [git-ref]      # default ref: HEAD of the current checkout
#
# Takes about 1-2 hours on a T4. The kernel runs offline (Kaggle disables internet until the account is phone
# verified), so this script first uploads a small private dataset with the code at <git-ref>, the ultralytics
# wheels and the COCO-pretrained yolov8n.pt. The card dataset is attached from Kaggle (andy8744/...).
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(git rev-parse --show-toplevel)"
REF="${1:-HEAD}"
USER_NAME="$(kaggle config view | awk '/username/ {print $3}')"
[ -n "$USER_NAME" ] || { echo "kaggle is not authenticated (set KAGGLE_API_TOKEN)"; exit 1; }

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
BUNDLE="$WORKDIR/bundle"; KERNEL="$WORKDIR/kernel"
mkdir -p "$BUNDLE/wheels" "$KERNEL"

echo "building the offline bundle for $REF"
git -C "$REPO_ROOT" archive --format=tar.gz -o "$BUNDLE/repo.tar.gz" "$REF"
git -C "$REPO_ROOT" rev-parse "$REF" > "$BUNDLE/git_ref.txt"
ULTRALYTICS="$(awk -F'==' '/^ultralytics==/ {print $2}' "$REPO_ROOT/requirements-worker.txt")"
python -m pip download -q --no-deps -d "$BUNDLE/wheels" "ultralytics==$ULTRALYTICS" ultralytics-thop py-cpuinfo
curl -fsSL -o "$BUNDLE/yolov8n.pt" https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
curl -fsSL -o "$BUNDLE/Arial.ttf" https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf
sed "s#KAGGLE_USERNAME#$USER_NAME#" dataset-metadata.json > "$BUNDLE/dataset-metadata.json"
if kaggle datasets status "$USER_NAME/cardsense-offline" >/dev/null 2>&1; then
  kaggle datasets version -p "$BUNDLE" -m "bundle for $REF" -r zip
else
  kaggle datasets create -p "$BUNDLE" -r zip
fi
echo "waiting for the dataset to be ready"
until kaggle datasets status "$USER_NAME/cardsense-offline" 2>&1 | grep -qi ready; do sleep 15; done

cp cardsense_train.py "$KERNEL/"
sed "s#KAGGLE_USERNAME#$USER_NAME#g" kernel-metadata.json > "$KERNEL/kernel-metadata.json"
SLUG="$USER_NAME/cardsense-yolov8n"
kaggle kernels push -p "$KERNEL"
echo "pushed; waiting for it to finish (https://www.kaggle.com/code/$SLUG)"
while true; do
  STATUS="$(kaggle kernels status "$SLUG" 2>&1 || true)"
  echo "$(date +%H:%M) $STATUS"
  case "$STATUS" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCEL*) mkdir -p ../results/failed; kaggle kernels output "$SLUG" -p ../results/failed || true
                      echo "kernel failed; log in training/results/failed"; exit 1 ;;
  esac
  sleep 120
done

mkdir -p ../results ../../models
kaggle kernels output "$SLUG" -p ../results
mv ../results/best.pt ../../models/best.pt
echo "weights: models/best.pt   metrics: training/results/test_metrics.json"
