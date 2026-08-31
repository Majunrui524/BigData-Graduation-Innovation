#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SAMPLE_ROOT="${SAMPLE_ROOT:-$ROOT_DIR/data/samples/final_v1}"
RUN_ROOT="${RUN_ROOT:-$SAMPLE_ROOT/analysis/run_10k_late}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-twibot-sampler}"
CONDA_SH="${CONDA_SH:-/opt/homebrew/anaconda3/etc/profile.d/conda.sh}"

EMBED_MODEL="${EMBED_MODEL:-text-embedding-3-small}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-64}"
EMBED_RPM="${EMBED_RPM:-120}"

GRAPH_K="${GRAPH_K:-5}"
GRAPH_CANDIDATE_K="${GRAPH_CANDIDATE_K:-100}"
GRAPH_MIN_SIMILARITY="${GRAPH_MIN_SIMILARITY:-0.0}"
GRAPH_BACKEND="${GRAPH_BACKEND:-numpy}"
GRAPH_CHUNK_SIZE="${GRAPH_CHUNK_SIZE:-256}"

COMMUNITY_ALGO="${COMMUNITY_ALGO:-structural_entropy}"
COMMUNITY_MIN_SIZE="${COMMUNITY_MIN_SIZE:-1}"
COMMUNITY_THRESHOLD="${COMMUNITY_THRESHOLD:-0.2}"

RERANKER_LR="${RERANKER_LR:-0.05}"
RERANKER_EPOCHS="${RERANKER_EPOCHS:-300}"
RERANKER_L2="${RERANKER_L2:-0.001}"

if [[ ! -d "$SAMPLE_ROOT" ]]; then
  echo "Sample root not found: $SAMPLE_ROOT" >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  echo "Missing $ROOT_DIR/.env" >&2
  echo "Copy .env.example to .env and fill in your API settings first." >&2
  exit 1
fi

if [[ ! -f "$SAMPLE_ROOT/derived/post_types/run_manifest.json" ]]; then
  echo "Missing post-type outputs under $SAMPLE_ROOT/derived/post_types" >&2
  echo "Run classify-post-types on final_v1 first." >&2
  exit 1
fi

if [[ ! -f "$SAMPLE_ROOT/derived/triplets/run_manifest.json" ]]; then
  echo "Missing triplet outputs under $SAMPLE_ROOT/derived/triplets" >&2
  echo "Run extract-triplets on final_v1 first." >&2
  exit 1
fi

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV_NAME"
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

mkdir -p "$RUN_ROOT"

echo "[1/7] Building user feature table"
python -m twibot22_sampler.cli build-user-features \
  --sample-root "$SAMPLE_ROOT"

echo "[2/7] Building user vectors"
python -m twibot22_sampler.cli build-user-vectors \
  --sample-root "$SAMPLE_ROOT" \
  --embedding-model "$EMBED_MODEL" \
  --batch-size "$EMBED_BATCH_SIZE" \
  --requests-per-minute "$EMBED_RPM" \
  --request-timeout-s 90 \
  --max-retries 6

echo "[3/7] Building temporal profiles"
python -m twibot22_sampler.cli build-temporal-profiles \
  --sample-root "$SAMPLE_ROOT"

echo "[4/7] Building late-fusion graph"
python -m twibot22_sampler.cli build-user-graph \
  --sample-root "$SAMPLE_ROOT" \
  --fusion-mode late \
  --backend "$GRAPH_BACKEND" \
  --chunk-size "$GRAPH_CHUNK_SIZE" \
  --k "$GRAPH_K" \
  --candidate-k "$GRAPH_CANDIDATE_K" \
  --min-similarity "$GRAPH_MIN_SIMILARITY" \
  --output-root "$RUN_ROOT/graph"

echo "[5/7] Detecting communities"
python -m twibot22_sampler.cli detect-communities \
  --sample-root "$SAMPLE_ROOT" \
  --graph-root "$RUN_ROOT/graph" \
  --output-root "$RUN_ROOT/communities" \
  --algorithm "$COMMUNITY_ALGO" \
  --min-community-size "$COMMUNITY_MIN_SIZE"

echo "[6/7] Evaluating communities"
python -m twibot22_sampler.cli evaluate-communities \
  --sample-root "$SAMPLE_ROOT" \
  --communities-root "$RUN_ROOT/communities" \
  --output-root "$RUN_ROOT/evaluation" \
  --threshold "$COMMUNITY_THRESHOLD"

echo "[7/7] Training reranker and analyzing changes"
python -m twibot22_sampler.cli train-community-reranker \
  --sample-root "$SAMPLE_ROOT" \
  --best-root "$RUN_ROOT" \
  --output-root "$RUN_ROOT/reranker" \
  --learning-rate "$RERANKER_LR" \
  --max-epochs "$RERANKER_EPOCHS" \
  --l2 "$RERANKER_L2" \
  --threshold-values 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5

python -m twibot22_sampler.cli analyze-community-reranker \
  --sample-root "$SAMPLE_ROOT" \
  --best-root "$RUN_ROOT" \
  --reranker-root "$RUN_ROOT/reranker" \
  --output-root "$RUN_ROOT/reranker_analysis" \
  --focus-split test \
  --top-k 100

echo
echo "Done."
echo "Run root: $RUN_ROOT"
echo "Graph manifest: $RUN_ROOT/graph/graph_manifest.json"
echo "Community manifest: $RUN_ROOT/communities/community_manifest.json"
echo "Community eval: $RUN_ROOT/evaluation/community_eval_manifest.json"
echo "Reranker manifest: $RUN_ROOT/reranker/reranker_manifest.json"
echo "Reranker analysis: $RUN_ROOT/reranker_analysis/reranker_comparison_manifest.json"
