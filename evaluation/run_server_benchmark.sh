#!/usr/bin/env bash
# Run one benchmark phase: start backends, warm them up, replay queries, stop backends.
#
# usage: run_server_benchmark.sh LABEL COLLECTION "ENV=VAL ENV2=VAL2" -- <run_api_benchmark.py args...>
#   LABEL       results go to $RESULTS_DIR/LABEL.jsonl, backend logs to $RESULTS_DIR/LABEL-backend-<port>.log
#   COLLECTION  Qdrant collection indexed with index_v3c_corpus.py
#   "ENV=VAL"   backend-only settings for this phase (e.g. "TYPE1_VLM_RERANK_ENABLED=false RRF_LEXICAL_WEIGHT=0"); "" for none
# Environment: PORTS (default "8001 8002"), RESULTS_DIR (default results), VLM_ENV_FILE (sourced when set; holds the
# OPENAI_* settings so keys never appear in argv), QDRANT_HOST/QDRANT_PORT.
set -euo pipefail

LABEL="$1"; COLLECTION="$2"; PHASE_ENV="$3"; shift 3
[ "${1:-}" = "--" ] && shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$REPO/results}"
PORTS=(${PORTS:-8001 8002})
mkdir -p "$RESULTS_DIR"

if [ -n "${VLM_ENV_FILE:-}" ]; then set -a; . "$VLM_ENV_FILE"; set +a; fi
export VISUAL_COLLECTION_NAME="$COLLECTION" SHOT_COLLECTION_NAME="${COLLECTION}_shot"
export QDRANT_HOST="${QDRANT_HOST:-127.0.0.1}" QDRANT_PORT="${QDRANT_PORT:-6333}" HF_HUB_DISABLE_TELEMETRY=1
for assignment in $PHASE_ENV; do export "$assignment"; done

PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT

URLS=()
for port in "${PORTS[@]}"; do
  (cd "$REPO/webapp/backend" && exec "$REPO/.venv/bin/python" -m uvicorn main:app --host 127.0.0.1 --port "$port" \
      >"$RESULTS_DIR/$LABEL-backend-$port.log" 2>&1) &
  PIDS+=($!)
  URLS+=("http://127.0.0.1:$port")
done
for url in "${URLS[@]}"; do
  for _ in $(seq 1 60); do curl -sf "$url/api/status" >/dev/null && break; sleep 2; done
  curl -sf "$url/api/status" >/dev/null || { echo "backend $url did not start" >&2; exit 1; }
done

# Lazy model loading would otherwise land in the first measured query of every backend.
WARM=()
for url in "${URLS[@]}"; do
  curl -s -o /dev/null --max-time 300 -X POST "$url/api/search" -H 'Content-Type: application/json' \
      -d '{"type":1,"query":"warm up","reset_session":true}' &
  WARM+=($!)
done
for pid in "${WARM[@]}"; do wait "$pid" || true; done

BASE_URLS="$(IFS=,; echo "${URLS[*]}")"
"$REPO/.venv/bin/python" "$REPO/evaluation/run_api_benchmark.py" --base-urls "$BASE_URLS" \
    --out "$RESULTS_DIR/$LABEL.jsonl" --videos-dir "$REPO/datasets" "$@"
