#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SAMPLE_ROOT="${SAMPLE_ROOT:-$ROOT_DIR/data/samples/final_v1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-twibot-sampler}"
CONDA_SH="${CONDA_SH:-/opt/homebrew/anaconda3/etc/profile.d/conda.sh}"

# Full-user defaults chosen to keep the formal run stable on a laptop.
POST_TYPE_MODE="${POST_TYPE_MODE:-hybrid}"
POST_TYPE_PER_USER_LIMIT="${POST_TYPE_PER_USER_LIMIT:-20}"
TRIPLET_PER_USER_LIMIT="${TRIPLET_PER_USER_LIMIT:-12}"

POST_TYPE_CONCURRENCY="${POST_TYPE_CONCURRENCY:-${OPENAI_CONCURRENCY:-4}}"
TRIPLET_CONCURRENCY="${TRIPLET_CONCURRENCY:-${OPENAI_CONCURRENCY:-4}}"
POST_TYPE_RPM="${POST_TYPE_RPM:-${OPENAI_REQUESTS_PER_MINUTE:-120}}"
TRIPLET_RPM="${TRIPLET_RPM:-${OPENAI_REQUESTS_PER_MINUTE:-120}}"

if [[ ! -d "$SAMPLE_ROOT" ]]; then
  echo "Sample root not found: $SAMPLE_ROOT" >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  echo "Missing $ROOT_DIR/.env" >&2
  echo "Copy .env.example to .env and fill in your API settings first." >&2
  exit 1
fi

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV_NAME"
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

echo "[1/2] Running full post-type classification on $SAMPLE_ROOT"
python -m twibot22_sampler.cli classify-post-types \
  --sample-root "$SAMPLE_ROOT" \
  --mode "$POST_TYPE_MODE" \
  --per-user-limit "$POST_TYPE_PER_USER_LIMIT" \
  --concurrency "$POST_TYPE_CONCURRENCY" \
  --requests-per-minute "$POST_TYPE_RPM" \
  --overwrite

echo "[2/2] Running full triplet extraction on $SAMPLE_ROOT"
python -m twibot22_sampler.cli extract-triplets \
  --sample-root "$SAMPLE_ROOT" \
  --per-user-limit "$TRIPLET_PER_USER_LIMIT" \
  --concurrency "$TRIPLET_CONCURRENCY" \
  --requests-per-minute "$TRIPLET_RPM" \
  --overwrite

echo "Done."
echo "Post types manifest: $SAMPLE_ROOT/derived/post_types/run_manifest.json"
echo "Triplets manifest: $SAMPLE_ROOT/derived/triplets/run_manifest.json"
