# Preprocessing & Indexing Pipeline (VBS 2027)

This module handles the extraction, description, embedding generation, and Qdrant indexing of video, image, and audio files.

## Project Structure

```
preprocessing/
├── config.py              # Configuration settings and model selection switches
├── main.py                # Main orchestrator script to run the preprocessing pipeline
├── download_assets.py     # Download optional TransNetV2 checkpoint
├── v3c_assets.py          # Optional V3C shot/keyframe/metadata/ASR asset adapter
├── requirements.txt       # Python dependencies
├── setup.sh               # Environment configuration script
├── CHANGELOG.md           # Log of project changes
├── host_qdrant.sh         # Starts Qdrant (via Docker or standalone binary download)
├── docker-compose.yml     # Docker Compose configuration for hosting Qdrant
├── video/
│   ├── scene_detector.py  # TransNetV2/PySceneDetect shot detection and keyframe sampling
│   ├── transnet_detector.py # Streaming TransNetV2 adapter with fallback
│   ├── ocr.py             # OCR text extraction and Unicode normalization
│   └── captioner.py       # Temporal/Scene narrative captions & structured attributes
├── audio/
│   └── audio_processor.py # Audio transcription and CLAP ambient features pipeline
└── indexing/
    ├── indexer.py         # Qdrant vector database client connection and indexer
    └── heagle.py          # H-EAGLE-lite shot aggregation helpers
```

*(Note: Shared models logic has been moved to the root `/models/` directory, and `host_vllm.sh` lives at the repo root since it's shared with `inference-code/` too).*

## Features

1. **Shot Boundary Detection & Adaptive Keyframe Sampling**: Uses official V3C boundaries/keyframes where mounted, otherwise streaming TransNetV2 with an explicit PySceneDetect fallback. CLIP variance sets a per-shot budget; WeMM vectors drive farthest-point selection with an optional sharpness bonus.
2. **Replaceable VLM and fixed primary embedding**:
   - VLM options (`VLM_OPTION`): local offline HuggingFace models (`generate_batch()` runs one true batched `model.generate()` call) or any OpenAI-compatible API (`OPENAI_BASE_URL`/`OPENAI_VLM_MODEL_NAME` - OpenAI itself, an alternative provider such as QwenCloud, or a self-hosted vLLM server for batch inference via the root `host_vllm.sh`). `generate_batch()` issues concurrent requests (`VLM_BATCH_CONCURRENCY`) so a batch-serving backend gets real throughput benefit.
   - Primary text/image embeddings use `WeMMEmbedding4BEmbedder` at the pinned Tencent revision, native 2560d by default. Only supported MRL dimensions are accepted. No Qwen/cloud primary-embedding fallback remains; VLM backend selection is independent.
   - Object Detection: YOLOE-26 (open-vocabulary, text-prompted, NMS-free) to locate objects zero-shot based on label lists, with a supplementary tiled detection pass for small objects (e.g. license plates) and optional example-crop visual prompting for categories that are awkward to phrase in text.
3. **OCR via PP-OCRv6**: Detection + recognition run directly through PP-OCRv6; only low-confidence crops below `OCR_REC_SCORE_THRESHOLD` get escalated to the lightweight fallback VLM for a re-read. OCR text is preserved after generic Unicode NFC normalization without language-specific accent mapping. Optional overlapping-tile pass (`OCR_USE_TILING`, off by default) handles small/corner text.
4. **Unified Per-Frame VLM Analysis**: One JSON call per keyframe (caption + objects/colors/count/scene_type/attributes) instead of two separate calls, batched across a scene's keyframes via `generate_batch()`.
5. **Speech & Audio Feature Extractors**: Speech transcription via faster-whisper (Whisper large-v3-turbo, with VAD + confidence filtering), environment audio indexing via M2D-CLAP.
6. **Qdrant Integration**: Writes the configured `VISUAL_COLLECTION_NAME` (default `visual_keyframes_v1`), `SHOT_COLLECTION_NAME` (default `vbs_shot_index`) and independent `audio_env_index`. Primary points carry a model/revision/dimension/pooling fingerprint. Unknown or mixed populated collections fail safely, without deletion. Uploads remain batched with explicit video/process flushes.

## Optional V3C official assets

V3C/VBS distributions can be mounted beside the raw videos with these
directories:

```text
assets/
├── msb/<video-stem>.txt       # shot rows: shot id, start, end
├── keyframes/<video-stem>/*   # one representative image per shot
├── metadata/<video-stem>.json  # title/description/category metadata
└── asr/<video-stem>.csv        # start,end,transcript
```

Set `V3C_ASSETS_DIR=/path/to/assets` in `preprocessing/.env`, or leave it
blank to probe the `--data_dir` itself. `V3C_ASSETS_ENABLED=true` enables the
adapter. Each asset family is independent: malformed/missing shot files use
TransNetV2 and then PySceneDetect, missing ASR uses local faster-whisper, and an ambiguous
keyframe mapping uses raw-video extraction. Official keyframe timestamps are
the shot midpoint unless the dataset provides a more precise mapping, so the
payload also stores `shot_id` and `asset_source` for auditability.

### TransNetV2 and H-EAGLE-lite

From the repository root, download the optional PyTorch checkpoint before processing raw videos:

```bash
uv run --group preprocessing python preprocessing/download_assets.py --transnetv2
```

The detector reads low-resolution frames in a bounded streaming window, so it
does not load an entire long video into RAM. Configure `SHOT_DETECTOR`,
`TRANSNETV2_MODEL_PATH`, `TRANSNETV2_DEVICE`, and `TRANSNETV2_THRESHOLD` in
`.env`. Official V3C `msb` files still take precedence, and setting
`SHOT_DETECTOR=pyscenedetect` is the immediate rollback.

Primary visual/shot identity or schema mismatches require fresh collection names and an explicit re-embedding run on the server. `QDRANT_ALLOW_RECREATE=true` does not override the populated-primary identity guard. Do not attach a new fingerprint to old vectors. For an intentional one-video rebuild in an already compatible index, `QDRANT_REBUILD_VIDEO_ON_START=true` remains a separately destructive opt-in; back up first.

H-EAGLE-lite preprocessing writes normalized shot aggregates to `SHOT_COLLECTION_NAME`.
Query-time coarse-to-fine routing is implemented but off by default; enable
`HEAGLE_LITE_ENABLED=true` in `inference-code/.env` only after measuring
recall and latency on the real corpus. The full H-EAGLE narrative-action/VLM
level is intentionally deferred.

Learned saliency is intentionally not part of the default pipeline. DAKE,
Laplacian sharpness and embedding-space diversity remain the low-cost,
text-preserving keyframe selector; a saliency model must pass a separate
text/location recall benchmark before it can be enabled.

## Installation

Run all commands below from the **repository root**. The setup script uses one root `.venv` through `uv`;
there is no separate `preprocessing/venv` to activate.

```bash
bash preprocessing/setup.sh
cp preprocessing/.env.template preprocessing/.env
```

Configure the ignored `.env` with your VLM settings and Hugging Face access. Install `ffmpeg` with your package
manager. Full preprocessing initializes **all** model stages, including SAM3 and the ambient-audio model,
even when processing visual files. Accept SAM3's gated terms and NTT's separate evaluation-only M2D terms
([`NOTICE.md`](../NOTICE.md)) before acquiring their assets:

```bash
uv run --group preprocessing python download_assets.py --accept-m2d-license
```

NTT's runtime is downloaded from its pinned upstream source, not redistributed in this repository.
SAM3 access/download errors mean full preprocessing setup is incomplete; they are not silently skipped.
For a lightweight visual-only index without these full-pipeline stages, see
[`evaluation/index_v3c_corpus.py`](../evaluation/index_v3c_corpus.py).

## Self-hosting Qdrant

You can host Qdrant locally/on your server using Docker or as a standalone binary:

Run the script to launch Qdrant:

```bash
(cd preprocessing && bash host_qdrant.sh)
```

* **Docker Mode**: Qdrant v1.10.1 runs with persistent storage in `preprocessing/qdrant_storage`; ports are bound to localhost.
* **Standalone Binary Mode**: If Docker is unavailable or startup fails, the script downloads the same pinned release for the supported OS/architecture, stores it in `preprocessing/qdrant_bin`, and waits for localhost readiness.

Once started, access the Web Dashboard at: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

## Self-hosting the VLM via vLLM (optional, GPU required)

`host_vllm.sh` lives at the repo root (shared with `inference-code/`) rather than here - see the root `README.md` for setup instructions.

## Usage

1. Configure the ignored `preprocessing/.env` with your APIs, models and thresholds.
2. Run from the repository root using the shared environment:

```bash
uv run --group preprocessing python preprocessing/main.py --data_dir /path/to/raw/dataset --temp_dir ./temp
```
