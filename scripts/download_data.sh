#!/usr/bin/env bash
# Download the Alibaba Taobao UserBehavior dataset into data/raw/.
#
# Official source: Tianchi dataset 649
#   https://tianchi.aliyun.com/dataset/649
# Downloaded here from the Kaggle mirror `marwa80/userbehavior` (about 950 MB zipped,
# about 3.7 GB unzipped as UserBehavior.csv, no header row).
#
# Requires Kaggle credentials at ~/.kaggle/kaggle.json and the `kaggle` dev dependency
# (installed by `uv sync`). Run from anywhere:
#   bash scripts/download_data.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
DATASET="marwa80/userbehavior"
ZIP="$RAW/userbehavior.zip"

mkdir -p "$RAW"

if [ -f "$RAW/UserBehavior.csv" ]; then
  echo "data/raw/UserBehavior.csv already exists, skipping download."
  exit 0
fi

echo "Downloading $DATASET to $RAW ..."
cd "$ROOT"
uv run kaggle datasets download -d "$DATASET" -p "$RAW"

echo "Unzipping ..."
unzip -o "$ZIP" -d "$RAW"
# Some mirrors nest a second zip inside the first; unpack it if present.
for inner in "$RAW"/*.zip; do
  [ "$inner" = "$ZIP" ] && continue
  [ -f "$inner" ] || continue
  echo "Unzipping nested archive $inner ..."
  unzip -o "$inner" -d "$RAW"
  rm -f "$inner"
done
rm -f "$ZIP"

echo "Done. Contents of $RAW:"
ls -la "$RAW"
