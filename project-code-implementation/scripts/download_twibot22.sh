#!/usr/bin/env bash
#
# download_twibot22.sh — fetch the official TwiBot-22 benchmark dataset
# =====================================================================
# The raw TwiBot-22 corpus is NOT bundled in this repository (the unpacked
# sample alone is > 2 GB). This script downloads it from the OFFICIAL
# Google Drive mirror hosted by the TwiBot-22 authors and unpacks it into
#   data/twibot22_raw/
#
# Usage:
#   bash scripts/download_twibot22.sh            # full dataset (large)
#   bash scripts/download_twibot22.sh --tiny     # dry-run check (prints next steps only)
#
# Requirements:
#   - gdown  (pip install gdown)
#   - ~12 GB free disk space (the full TwiBot-22 archive is several GB)
#
# License / compliance:
#   TwiBot-22 is released under CC BY-NC-ND 4.0 and may only be used for
#   academic research. You must also respect the "Content redistribution"
#   section of the Twitter Developer Agreement and Policy.
#   See https://github.com/LuoUndergradXJTU/TwiBot-22 for details.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${RAW_DIR:-$ROOT_DIR/data/twibot22_raw}"
ARCHIVE="$RAW_DIR/twibot22.zip"

# Official Google Drive file id, exactly as published in the TwiBot-22 repo:
#   https://github.com/LuoUndergradXJTU/TwiBot-22
GDRIVE_ID="${GDRIVE_ID:-1YwiOUwtl8pCd2GD97Q_WEzwEUtSPoxFs}"

echo "======================================================================="
echo "  TwiBot-22 downloader"
echo "  target : $RAW_DIR"
echo "  source : Google Drive (official TwiBot-22 mirror)"
echo "  license: CC BY-NC-ND 4.0 (academic research only)"
echo "======================================================================="

if [[ "${1:-}" == "--tiny" ]]; then
  echo
  echo "[tiny] Skipping download. Here is the full recipe:"
  echo
  cat <<'EOF'
  # 1. Fetch the archive (~several GB)
  bash scripts/download_twibot22.sh

  # 2. (Optional) quick smoke test on 2,000 users instead of the full 10k
  python -m twibot22_sampler.cli sample --preset smoke \
    --data-root   data/twibot22_raw \
    --work-root   data/work_smoke \
    --output-root data/samples/smoke_v1 \
    --seed 42

  # 3. Full 10k sample (matches every number on the README)
  python -m twibot22_sampler.cli sample --preset main \
    --data-root   data/twibot22_raw \
    --work-root   data/work_main \
    --output-root data/samples/final_v1 \
    --seed 42
EOF
  echo
  echo "[tiny] done. (Run without --tiny to actually download.)"
  exit 0
fi

# --- 0. sanity -----------------------------------------------------------
command -v gdown >/dev/null 2>&1 || {
  echo "gdown not found. Install it with:  pip install gdown" >&2
  exit 1
}

mkdir -p "$RAW_DIR"

# --- 1. download (resumable) ---------------------------------------------
echo "[1/3] Downloading TwiBot-22 from Google Drive (this can take a while)..."
gdown --id "$GDRIVE_ID" -O "$ARCHIVE" --continue

if [[ ! -s "$ARCHIVE" ]]; then
  echo "Download failed or produced an empty file: $ARCHIVE" >&2
  exit 1
fi
echo "[1/3] done: $(du -h "$ARCHIVE" | cut -f1)"

# --- 2. unpack ------------------------------------------------------------
echo "[2/3] Unpacking into $RAW_DIR ..."
cd "$RAW_DIR"
unzip -o "$ARCHIVE" -d . || python -m zipfile -e "$ARCHIVE" .

# --- 3. report ------------------------------------------------------------
echo "[3/3] Done. Contents of $RAW_DIR :"
find . -maxdepth 2 -type f | head -20
echo
echo "Next steps (run from project-code-implementation/):"
echo
cat <<EOF
  # Quick smoke test on 2,000 users (fast, verifies the whole chain)
  python -m twibot22_sampler.cli sample --preset smoke \\
    --data-root   $RAW_DIR \\
    --work-root   data/work_smoke \\
    --output-root data/samples/smoke_v1 \\
    --seed 42

  # Full 10k sample (reproduces every README headline number)
  python -m twibot22_sampler.cli sample --preset main \\
    --data-root   $RAW_DIR \\
    --work-root   data/work_main \\
    --output-root data/samples/final_v1 \\
    --seed 42
EOF
