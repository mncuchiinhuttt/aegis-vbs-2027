# AEGIS - Adaptive, Evidence-Grounded Interactive Search (VBS 2027)

Research code for **AEGIS**, our interactive video retrieval system for the [Video Browser Showdown (VBS) 2027](https://videobrowsershowdown.org/call-for-papers/) at MMM 2027. Project page: <https://aegis.mncuchiinhuttt.dev/>. Authors and citation: [`CITATION.cff`](CITATION.cff). See [`VBS_GUIDE.md`](VBS_GUIDE.md) for a competition reference (task types, scoring, DRES, datasets).

> **Status - read first.** This is competition research code, not a validated product. The recorded benchmark in [`evaluation/README.md`](evaluation/README.md) is an offline replay on a 1,765-video V3C subset whose text index holds only video titles, descriptions and tags; it is not an official VBS score. The DRES integration has not been exercised against a live DRES instance. The web console and API have **no authentication** (they can submit to DRES and start jobs) and listen on localhost by default; do not expose them to an untrusted network.

The pipeline started from our earlier HCMC AI Challenge 2026 codebase (a batch-submission setting) and was reworked for VBS, which is **live/interactive**: an operator drives searches under a per-task clock, with iterative feedback, query-by-example and temporal browsing. Indexing and retrieval share one fixed WeMM primary embedding space; the independent VLM/OCR/ASR modules support evidence extraction and operator interaction. See [WebApp Dashboard](#5-webapp-dashboard--interactive-session).

The system processes raw datasets (video, images, and audio), generates multimodal embeddings, indexes them to a Qdrant database, and retrieves matching frames/sequences for three query types: Textual-KIS (Type 1), VQA (Type 2), and Temporal-Alignment (Type 3).

---

## Directory Structure

```
aegis-vbs-2027/
├── README.md              # Global workspace documentation
├── VBS_GUIDE.md           # VBS 2027 competition reference (tasks, scoring, DRES, datasets)
├── .env.template          # Root-level secrets template (HF_TOKEN, DRES_* credentials)
├── .gitignore             # Root git ignore (excludes /weights/, /datasets/, webapp/backend/logs/)
├── download_assets.py     # Script to automate downloading weights from Hugging Face
├── host_vllm.sh           # Self-hosts the local VLM via vLLM for batch inference (GPU only) - shared by preprocessing/ and inference-code/
├── evaluation/            # Standalone evaluation & benchmarking module
│   ├── README.md          # Evaluation module documentation & instructions
│   ├── eval_queries.json  # Fictional legacy schema demo; not real annotations
│   ├── requirements.txt   # Optional deps (ragas, datasets) for generation-quality metrics
│   └── run_eval.py        # Standalone benchmark runner (Latency, Recall@K, MRR, optional Ragas)
├── models/                # [SHARED PYTHON MODELS] 
│   ├── base_vlm.py        # Abstract VLM interface
│   ├── qwen_vlm.py        # Optional local VLM backend, independent of embeddings
│   ├── openai_vlm.py      # Replaceable API VLM adapter
│   ├── embedding.py       # WeMMEmbedding4BEmbedder and M2DClapEmbedder exports
│   ├── wemm_embedding.py  # Pinned native Tencent WeMM text/image inference
│   ├── embedding_identity.py # Model/revision/dimension/pooling fingerprints
│   ├── clip_embedder.py   # Lightweight CLIP embedder for keyframe scene-variance estimation
│   ├── asr.py             # faster-whisper (Whisper large-v3-turbo) speech-to-text transcriber
│   ├── object_detector.py # YOLOE-26 open-vocabulary object detector
│   ├── region_proposer.py  # SAM3 zero-shot region proposal (Object Detection + OCR pre-filter)
│   ├── super_resolution.py # Real-ESRGAN x4 conditional upscaling for small OCR crops
│   └── fallback_vlm.py     # Lightweight SmolVLM2 fallback for low-confidence OCR crops
├── preprocessing/         # Dataset indexing pipeline (inherited from our AIC 2026 pipeline, adapted for V3C/VBS)
│   ├── config.py          # Preprocessing settings, API URLs, and thresholds
│   ├── .env               # API Keys and model configurations (ignored)
│   ├── main.py            # Orchestrator to scan data, extract captions/embeddings
│   ├── requirements.txt   # Dependencies for preprocessing
│   ├── setup.sh           # Environment setup shell script
│   ├── host_qdrant.sh     # Starts Qdrant (via Docker or standalone binary download)
│   └── docker-compose.yml # Docker Compose config for Qdrant
├── inference-code/        # Retrieval and query engine
│   ├── config.py          # Search parameters, thresholds, and Qdrant settings (defaults tuned for VBS live latency)
│   ├── .env               # API Keys and model configurations (ignored)
│   ├── main.py            # CLI query parser for Type 1, 2, 3 retrieval
│   ├── search/hybrid_search.py # HybridSearcher - dense/sparse fusion, Rocchio feedback, temporal window match
│   └── requirements.txt   # Dependencies for inference
├── webapp/                # Interactive operator dashboard (VBS session UI + DRES integration)
│   ├── backend/           # FastAPI backend - see "WebApp Dashboard" section for endpoint list
│   │   ├── main.py            # API endpoints (search, feedback, DRES proxy, logging)
│   │   ├── dres_client.py     # Thin REST wrapper for the DRES evaluation server
│   │   ├── interaction_log.py # Local JSONL + best-effort DRES interaction/query logging
│   │   └── requirements.txt   # Backend dependencies (fastapi, requests, etc.)
│   └── frontend/          # React + Vite operator UI
│       └── src/components/
│           ├── ResultCard.tsx        # Result card: feedback, query-by-example, in-video search, DRES submit
│           └── BrowseVideoDialog.tsx # Full-video keyframe browser dialog
├── weights/               # [IGNORED] Downloaded model weights (.pth, .bin)
└── datasets/              # [IGNORED] Place raw videos, images, and audios here
```

Also in the repository: `scripts/fetch_v3c_subset.py` (V3C subset downloader), `tests/` (CPU-only contract tests), `queries/` (sample queries for the batch CLI) and `evaluation/data/` (benchmark inputs), plus [`NOTICE.md`](NOTICE.md) for third-party terms.

---

## Shared Models Architecture

To avoid duplicate codebase wrappers, all model configurations and execution logic are stored in the root `models/` directory.

- **VLM backends** (`VLM_OPTION`): a replaceable local backend or API adapter (`OPENAI_BASE_URL`/`OPENAI_VLM_MODEL_NAME`). This is independent from primary retrieval embeddings. `generate_batch()` can issue concurrent requests to a compatible serving backend; its actual throughput requires measurement.
- **Primary embeddings**: `WeMMEmbedding4BEmbedder` uses `tencent/WeMM-Embedding-4B`, pinned to revision `41a4fafdd26252ddff05c025413af7e61e26f08e`, through the official `model.embedding()` API. Native output is 2560d; only the checkpoint's supported MRL dimensions are accepted. There is no Qwen3-VL-Embedding or cloud-embedding fallback. `M2DClapEmbedder` remains the independent 768d ambient-audio encoder.
- **Object Detection**: `ObjectDetector` wraps YOLOE-26 (open-vocabulary, text-prompted, NMS-free end-to-end) to locate target objects, with tiled inference for small objects (`detect_tiled`), example-crop-based visual prompting (`detect_visual_prompt`), and SAM3-gated region-restricted tiling (`detect_in_regions`) used by the preprocessing pipeline (see below).
- **SAM3-gated Detection & OCR pre-filter**: `RegionProposer` (`models/region_proposer.py`) wraps `facebook/sam3` (Promptable Concept Segmentation, zero-shot, **gated on Hugging Face** - accept the license at https://huggingface.co/facebook/sam3 and set `HF_TOKEN` before downloading/running it) to propose candidate regions from general concept prompts ("human"/"vehicle"/"small distinct object" for Object Detection, "text or sign region" for OCR) before SAHI-style tiling (512x512, 0.2 overlap) and the actual detector/recognizer run - a keyframe's detection/OCR step is skipped entirely if SAM3 finds no matching region.
- **OCR**: `preprocessing/video/ocr.py` uses PP-OCRv6 for each detected text-box crop; crops shorter than 16px are upscaled first via `SuperResolutionUpscaler` (`models/super_resolution.py`, Real-ESRGAN x4), and low-confidence crops fall back to a dedicated lightweight VLM (`SmolVLM2FallbackVLM`, `models/fallback_vlm.py`) rather than the main captioning VLM.
- **Adaptive Keyframe Sampling**: `models/clip_embedder.py`'s lightweight CLIP model estimates how visually static/dynamic a scene is, sizing a per-scene keyframe budget (1-8) before the real embedding model runs farthest-point sampling within it - see `preprocessing/video/scene_detector.py`.
- **ASR**: `WhisperASR` (`models/asr.py`, faster-whisper/CTranslate2 running Whisper large-v3-turbo, 99-language multilingual - VBS's V3C dataset isn't Vietnamese-centric like AIC's, so a general multilingual model replaces the AIC pipeline's Vietnamese-specialized PhoWhisper) transcribes speech with word-level timestamps; silero VAD skips non-speech regions, and `preprocessing/audio/asr_segment_filter.py` drops any remaining low-confidence/hallucinated segments (OpenAI Whisper's own reference thresholds on `avg_logprob`/`no_speech_prob`/`compression_ratio`) before they reach embedding + indexing.

The scripts dynamically append the workspace root to `sys.path` to import `models.*` from anywhere.

### WeMM index contract

Set `WEMM_MODEL_REVISION`, `VISUAL_COLLECTION_NAME` and `SHOT_COLLECTION_NAME` consistently for indexing and retrieval. Their defaults select the pinned WeMM revision, `visual_keyframes_v1` and `vbs_shot_index`. `EMBEDDING_MRL_DIM=` uses native 2560d; supported alternatives are 64, 128, 256, 512 and 1024. The historical 2048d software prefix is not an accepted native-MRL setting.

Every primary visual/speech/shot point carries an `embedding_fingerprint` derived from model, revision, dimension and pooling recipe. Unknown, mixed or incompatible populated collections are rejected without deleting points, even if dimensions happen to match. For a new embedding space, choose **new visual and shot collection names** and re-embed from media. Do not relabel old vectors with a new fingerprint or treat a snapshot copy as an embedding conversion. VLM settings and WeMM's underlying backbone names are not alternate primary-encoder routes.

The BF16 checkpoint alone is about 10.35 GB; allow additional memory for inference and other models. For an encoder-only setup the full downloader is not needed: `uv run --group inference hf download tencent/WeMM-Embedding-4B --revision 41a4fafdd26252ddff05c025413af7e61e26f08e --local-dir weights/WeMM-Embedding-4B`. A recorded GPU run (V100 32 GB, fp16) and its limits are in [`evaluation/README.md`](evaluation/README.md#recorded-run-2026-10-28-one-v100-32-gb-1765-v3c-videos); the production VQA replay instructions are in [`inference-code/README.md`](inference-code/README.md#server-preparation-and-production-vqa-replay).

---

## Getting Started

### 0. Install the Python environment with uv (recommended)

The project uses one shared Python 3.12 environment at the repository root.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once, then sync only the dependencies needed for the workflow:

```bash
# Dashboard/backend only (fastest first run)
uv sync

# Retrieval/inference dependencies
uv sync --group inference

# Full indexing + OCR/SAM3 environment
uv sync --group preprocessing

# Optional upload/evaluation tools
uv sync --group upload
uv sync --group evaluation
```

Use `uv run ...` for every Python command; it automatically uses the locked
`.venv` and does not require manual activation. `uv.lock` is committed so
machines running the same project resolve the same dependency versions.

### Tests and frontend checks

```bash
uv sync --frozen --group inference --group dev
uv run --frozen --group inference --group dev python -m pytest -q tests
npm --prefix webapp/frontend ci
npm --prefix webapp/frontend run lint
npm --prefix webapp/frontend run build
```

These checks use no model weights, hosted VLM credentials or V3C videos. They validate contracts and buildability,
not retrieval quality, answer correctness or live DRES acceptance.

### 1. Download Model Checkpoints

Before running the download script, accept the SAM3 license at https://huggingface.co/facebook/sam3 (it's a gated repo) and set `HF_TOKEN` in your `.env` - the SAM3 download will fail without it.

The full downloader retrieves WeMM, Whisper, CLAP, the detector, SAM3, fallback VLM and super-resolution assets. Full preprocessing requires SAM3 access and NTT's separately licensed M2D runtime. Read [`NOTICE.md`](NOTICE.md) before accepting those terms; this release does not redistribute NTT's runtime. Retrieval-only users can instead use the WeMM-only command in the [index contract](#wemm-index-contract), without NTT/SAM3.

```bash
uv run --group inference python download_assets.py --accept-m2d-license
```

### 2. Host the Database (Qdrant)

Start the Qdrant server instance. The script will automatically try using Docker if installed, or download and run the native standalone Qdrant binary in the background:

```bash
cd preprocessing
chmod +x host_qdrant.sh
./host_qdrant.sh
```

Access the Qdrant Dashboard at: [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

### 2b. Host the VLM via vLLM (optional, GPU required)

For batch inference throughput when self-hosting the local VLM (instead of one-request-at-a-time HuggingFace `transformers` calls), serve it through vLLM's OpenAI-compatible server - shared by both `preprocessing/` and `inference-code/`:

```bash
chmod +x host_vllm.sh
./host_vllm.sh
```

Requires an NVIDIA (CUDA) or AMD (ROCm) GPU - it does not run on Apple Silicon/macOS or CPU-only machines. Once the server is up, point either module's `.env` at it instead of loading a local HF model:

```bash
VLM_OPTION=openai
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_VLM_MODEL_NAME=<same model served by host_vllm.sh>
VLM_BATCH_CONCURRENCY=16   # raise this to actually use vLLM's continuous batching
OPENAI_VLM_MAX_COMPLETION_TOKENS=4096  # bounded reasoning/output budget
OPENAI_VLM_TIMEOUT_SEC=45             # fail a hung live request promptly
```

### 2c. Migrating Indexed Data to a New Server

Everything Qdrant indexes (embeddings + payloads from `preprocessing/main.py`) lives on disk under `preprocessing/qdrant_storage/` - both `host_qdrant.sh`'s Docker path (`docker-compose.yml`'s `./qdrant_storage:/qdrant/storage` volume) and its standalone-binary fallback (`QDRANT__STORAGE__STORAGE_PATH=./qdrant_storage`) write there. Model weights (`weights/`) are **not** part of this - they're just re-downloaded via `download_assets.py` on the new server, no need to copy them.

**Option A - copy the storage directory directly (simplest, same Qdrant version only):**

```bash
# On the old server: stop Qdrant first so files aren't mid-write
cd preprocessing
docker compose down   # or: pkill qdrant   (if running the standalone binary)

# Copy the whole storage dir to the new server (any transfer tool works)
rsync -avz qdrant_storage/ new-server:/path/to/aegis-vbs-2027/preprocessing/qdrant_storage/

# On the new server: start Qdrant as usual - it picks the copied data up automatically
cd preprocessing && ./host_qdrant.sh
```

Only copy storage directories between servers running the **same Qdrant version**. Both `docker-compose.yml` and `host_qdrant.sh` pin `v1.10.1`, the version used for the recorded run. Both Docker port publishing and standalone HTTP/gRPC bindings are localhost-only by default; use an authenticated deployment if remote database access is needed.

**Option B - Qdrant's snapshot API (recommended, per-collection, no downtime on the source; restore to the same server version first):**

```bash
# Snapshot each configured collection; this example uses the default visual collection.
curl -X POST http://localhost:6333/collections/visual_keyframes_v1/snapshots

# List snapshots to get the exact filename just created
curl http://localhost:6333/collections/visual_keyframes_v1/snapshots

# Download it
curl -o visual_keyframes_v1.snapshot \
  http://localhost:6333/collections/visual_keyframes_v1/snapshots/<snapshot_name>

# Copy visual_keyframes_v1.snapshot, then restore to the chosen target collection.
# (new server's Qdrant must be running first)
curl -X POST http://localhost:6333/collections/visual_keyframes_v1/snapshots/upload \
  -F "snapshot=@visual_keyframes_v1.snapshot"
```

Repeat for the actual `VISUAL_COLLECTION_NAME`, `SHOT_COLLECTION_NAME` and `audio_env_index` as needed. A snapshot transfers existing vectors unchanged; it does not establish their compatibility with a new WeMM recipe. Preserve the old collections and use fresh names when re-embedding. See [Qdrant snapshot documentation](https://qdrant.tech/documentation/concepts/snapshots/) for server migration details.

---

## 3. Preprocessing & Indexing

1. Sync the preprocessing environment:
   ```bash
   uv sync --group preprocessing
   ```
2. Configure `.env` in `preprocessing/` (add API keys and choose model configurations).
3. Place your raw files in the global `datasets/` folder.
4. Run the pipeline:
   ```bash
   uv run --group preprocessing python preprocessing/main.py --data_dir datasets
   ```

### 3a. Reducing Index Size with Matryoshka (MRL) Truncation

`WeMMEmbedding4BEmbedder` defaults to the checkpoint's native **2560d** output. Set the same `EMBEDDING_MRL_DIM` for indexing and querying only to a value advertised by the pinned model: 64, 128, 256, 512, 1024 or 2560. The wrapper truncates and re-normalizes supported prefixes; it rejects unsupported dimensions instead of silently selecting a different encoder or dimension.

Measure the quality/storage trade-off on independently labelled server-side queries; no accuracy or speed improvement is guaranteed. A dimension, revision or pooling change creates a different embedding fingerprint. Select fresh `VISUAL_COLLECTION_NAME` **and** `SHOT_COLLECTION_NAME` values and re-embed before comparing results. Existing vectors remain unchanged; do not override the guard by retagging them. MRL settings do not apply to the M2D-CLAP audio space.

---

## 4. Query Retrieval (Inference)

1. Stay at the repository root and sync inference dependencies:
   ```bash
   uv sync --frozen --group inference
   ```
2. Configure `.env` in `inference-code/` (point to Qdrant host and define query models).
3. Run search queries from CLI:

#### Type 1: Textual-KIS (Retrieves matching video name and timestamp)

```bash
uv run --group inference python inference-code/main.py --type 1 --query "một người đang lái xe máy đi qua ngã tư dưới trời mưa"
```

#### Type 2: VQA (retrieval plus evidence-aware VLM; optional detector crops or full-frame mode)

```bash
uv run --group inference python inference-code/main.py --type 2 --query "người mặc áo đỏ đang dắt xe đạp màu xanh ở giây thứ mấy?"
```

#### Type 3: Temporal-Alignment (Reranks sequence of events chronologically)

```bash
uv run --group inference python inference-code/main.py --type 3 --query "đầu tiên có người chạy bộ qua đường, tiếp đến chiếc ô tô đen đi qua"
```

### Production VQA replay status

CPU-only unit and contract checks are not a model evaluation. `evaluation/eval_queries.json` is fictional legacy schema data, not an annotated VQA set. The recorded run in [`evaluation/README.md`](evaluation/README.md) measured only abstention, evidence availability and latency on 40 VQA questions; no answer-accuracy or hallucination rate exists because there are no answer labels. See the [server prerequisites, exact replay commands, and metric denominators](inference-code/README.md#server-preparation-and-production-vqa-replay) before reporting results.


---

## 5. WebApp Dashboard & Interactive Session

The webapp is the operator's live console during a VBS task: single/batch queries, database stats, and the interactive session tools described below. Each operator runs their own backend instance (no multi-tenant session store) — this matches VBS's one-workstation-per-operator setup.

**Network exposure.** The backend (`AEGIS_HOST`, default `127.0.0.1`, port 8000) and the Vite dev server (`VITE_HOST`, default `127.0.0.1`) listen on localhost; CORS allows only the local dev origins unless `CORS_ALLOW_ORIGINS` is set. Expose them only behind your own access control. `run_webapp.py` refuses to start when ports 8000/5173 are busy; `--free-ports` explicitly permits terminating the processes holding them.

### Start the WebApp Dev Servers

Concurrently run both the React frontend and the FastAPI backend dev servers with:

```bash
# Python Runner (Resolves port conflicts & detects venv automatically)
uv run python run_webapp.py

# Bash Runner
./run_webapp.sh
```

- Open **Dashboard (Vite)**: [http://localhost:5173](http://localhost:5173)
- Open **API Docs (FastAPI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Interactive Session Endpoints

Beyond the base `/api/search`, `/api/status`, and media endpoints, the backend exposes the interactive tools an operator uses mid-task:

| Endpoint | Purpose |
| --- | --- |
| `POST /api/feedback` | Rocchio-style relevance feedback (positive/negative result marks) — re-searches with an adjusted query vector |
| `POST /api/query-by-example` | Re-search using an already-indexed result's own stored vector, no re-embedding |
| `POST /api/search-by-image` | **KIS-V** — search by an uploaded photo/screenshot instead of text (embeds the upload, searches by that vector directly, no RRF fusion) |
| `POST /api/temporal-search` | **N-query temporal chain** search (`queries: string[]`, N≥2) — finds the best chronologically-ordered frame chain per video, one step per query, each within a frame window of the previous step (Exquisitor-inspired sequence-chain matching) |
| `GET /api/browse-video/{video_name}` | Full keyframe listing for a single video, for manual scrubbing |
| `POST /api/in-video-search` | Manually-triggered deep search restricted to one candidate video |
| `POST /api/dres/login` / `GET /api/dres/current-task` | DRES session login and current-task lookup (backend-proxied, credentials never reach the frontend) |
| `POST /api/dres/submit` | Submit an answer to DRES. Accepts optional `video_name`/`force` — resubmitting a video already submitted for the same `task_id` returns a 409 warning (not a hard block, overridable with `force: true`), since VBS's AVS scoring gives no extra credit for a duplicate video and penalizes wrong resubmissions |

Every search/feedback/query-by-example/temporal-search call is logged locally to `webapp/backend/logs/interaction_log.jsonl` first, then best-effort pushed to DRES if `DRES_*` env vars are configured — a DRES outage never blocks the operator's response (`webapp/backend/interaction_log.py`).

### Result Quality & Precision Controls

- **Explainability**: `/api/search` results include a `matched_via` field (e.g. `["query", "hyde"]`) showing which fusion source(s) surfaced each hit, plus OCR/scene-narrative evidence shown unconditionally on VQA answer cards — so the operator can judge trust before acting instead of seeing one opaque score.
- **Temporal coherence re-scoring**: candidates from the same video within a small frame window boost each other's score, so a real event isn't left fragmented across several marginal individually-scored frames.
- **KIS-C clarification**: when the top results spread across many unrelated videos with no clear winner (ambiguous query), the response includes a `clarification` field with one system-generated narrowing question — shown as an amber banner in the UI. Gated behind `AMBIGUITY_THRESHOLD` (default `0.7`) so the common unambiguous case pays no extra cost.
- **Escalate precision on-demand**: `/api/search` accepts optional `exact` (force exact/brute-force Qdrant search), `verify` (force verification reranking), and `hnsw_ef` (graduated HNSW search-time recall/latency tradeoff, a middle ground between the default and `exact`) — all `None`/unset by default, so behavior only changes when an operator explicitly opts in per-search (the frontend exposes `exact`/`verify` as two checkboxes; `hnsw_ef` is API-only for now).
- **AVS diversification**: `/api/search` collapses candidates to the single highest-scoring hit per video+scene (`diversify_by_scene`) right after temporal coherence re-scoring, so results aren't flooded by near-duplicate keyframes of the same event — directly serving AVS's diversity-across-videos scoring.

### DRES Configuration

Copy `.env.template` to `.env` at the repo root (or into `webapp/backend/`) and fill in the competition's DRES details once known:

```bash
DRES_BASE_URL=
DRES_USERNAME=
DRES_PASSWORD=
DRES_EVALUATION_ID=
```

These are unverified against a live DRES instance — confirm against the actual VBS 2027 DRES deployment before competition day.

---

## 6. Batch Queries (Process Multiple Queries)

### CLI Execution

To execute multiple queries in batch from the terminal:

1. Place your queries in the `queries/queries.json` file.
2. Run:

```bash
cd inference-code
python batch_query.py --query_file ../queries/queries.json --output_dir ../queries/
```

3. Batch outputs will be saved to `queries/batch_results.json` and `queries/batch_results.csv`.

### WebApp Dashboard Execution

1. Navigate to the **Process Multiple Queries** tab in the main console.
2. Click the **Process Multiple Queries** button to run batch inference.
3. Review logs live in the terminal output widget, and interact with the results list (including click-to-play support for matched segments).

---

## 7. System Evaluation & Benchmarking

We provide a standalone, decoupled evaluation runner to measure **End-to-End Latency** and **Accuracy Metrics** (Recall@K, MRR, and optionally Ragas Faithfulness/Answer Correctness/Context Recall) without altering production codebase files.

### Running Benchmarks via CLI

Run the evaluation script from the repository root:

`evaluation/eval_queries.json` is a fictional schema demonstration, **not** an annotated benchmark.
Use your own independently annotated query file for real evaluation:

```bash
# Inference dependencies plus optional Ragas/datasets diagnostics
uv sync --frozen --group inference --group evaluation
uv run --group inference --group evaluation python evaluation/run_eval.py \
    --query_file /path/to/annotated-queries.json --dataset_dir datasets \
    --output_file evaluation/eval_results.json
```

- **Output Metrics**: Evaluates **Recall@1**, **Recall@5**, **MRR**, **Latency Breakdown** (HyDE, Search, Rerank), and **QPS Throughput** across Type 1 (KIS), Type 2 (VQA), and Type 3 (Temporal) queries. Ragas-based generation metrics report `N/A` if `ragas` isn't installed, rather than a fabricated score.
- Accuracy metrics require a `ground_truth`-annotated query file — do not point `--query_file` at `queries/queries.json`, which is the production query registry and has no ground truth.
- Detailed results are printed to stdout and saved to `evaluation/eval_results.json`. See `evaluation/README.md` for complete documentation.
- The end-to-end HTTP benchmark (corpus indexing, per-query replay through `/api/search`, offline scoring, recorded run and its limits) is documented in [`evaluation/README.md`](evaluation/README.md#api-boundary-retrieval-benchmark-v3c-corpus-on-a-gpu-server).

---

## License

This repository's code is licensed under the [MIT License](LICENSE). This covers only the code in this repo — it does **not** extend to third-party model weights or code fetched by `download_assets.py` (Whisper large-v3-turbo, YOLOE-26, SAM3, SmolVLM2, Real-ESRGAN, NTT's M2D runtime, etc.), each of which carries its own license/usage terms, nor to the V3C videos (NIST data agreement). SAM3 is a gated Hugging Face repo requiring separate license acceptance; the M2D runtime is licensed by NTT for evaluation use only and is therefore downloaded, not shipped. See [`NOTICE.md`](NOTICE.md) and [Getting Started](#1-download-model-checkpoints).

## Citation

Please cite the paper listed in [`CITATION.cff`](CITATION.cff) once its final reference is available.
