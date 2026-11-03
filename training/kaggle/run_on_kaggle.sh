#!/usr/bin/env bash
# Train the detector on a free Kaggle GPU and download best.pt + metrics into models/ and training/results/.
#
#   export KAGGLE_API_TOKEN=...          # kaggle.com -> Settings -> API -> Create New Token
#   pip install kaggle
#   training/kaggle/run_on_kaggle.sh [git-ref]      # default ref: main
#
# Takes about 1-2 hours on a T4. Needs a Kaggle account with phone verification (required for GPU and internet).
set -euo pipefail
cd "$(dirname "$0")"
REF="${1:-main}"
USER_NAME="$(kaggle config view | awk '/username/ {print $3}')"
[ -n "$USER_NAME" ] || { echo "kaggle is not authenticated (set KAGGLE_API_TOKEN)"; exit 1; }

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
cp cardsense_train.py "$WORKDIR/"
# The ref is baked into the pushed script, so a run is pinned to one commit/branch.
sed -i "s#os.environ.get(\"CARDSENSE_REF\", \"main\")#os.environ.get(\"CARDSENSE_REF\", \"$REF\")#" "$WORKDIR/cardsense_train.py"
sed "s#KAGGLE_USERNAME#$USER_NAME#" kernel-metadata.json > "$WORKDIR/kernel-metadata.json"

SLUG="$USER_NAME/cardsense-yolov8n"
kaggle kernels push -p "$WORKDIR"
echo "pushed; waiting for it to finish (https://www.kaggle.com/code/$SLUG)"
while true; do
  STATUS="$(kaggle kernels status "$SLUG" 2>&1 || true)"
  echo "$(date +%H:%M) $STATUS"
  case "$STATUS" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCEL*) kaggle kernels output "$SLUG" -p ../results/failed || true; echo "kernel failed; log in training/results/failed"; exit 1 ;;
  esac
  sleep 120
done

mkdir -p ../results ../../models
kaggle kernels output "$SLUG" -p ../results
mv ../results/best.pt ../../models/best.pt
echo "weights: models/best.pt   metrics: training/results/test_metrics.json"
