# Inference Engine — VBS 2027

This module provides hybrid retrieval, conversational rewriting/HyDE, evidence-aware VQA and a legacy temporal-alignment tool. The primary text/image embedding is fixed to **Tencent WeMM-Embedding-4B**; VLM generation and ambient-audio encoding remain independent.

## Primary embedding contract

- `models.embedding.WeMMEmbedding4BEmbedder` uses the official native `model.embedding()` API at pinned revision `41a4fafdd26252ddff05c025413af7e61e26f08e`.
- Default dimension is 2560. `EMBEDDING_MRL_DIM` accepts only a supported checkpoint dimension: 64, 128, 256, 512, 1024 or 2560. Historical 2048d software prefixes are not a supported current-MRL option.
- No `QwenVL8BEmbedder`, cloud-embedding fallback or `EMBEDDING_OPTION` selection remains. Old primary-embedding environment keys do not select another encoder.
- `VLM_OPTION` and its model/provider configuration select the replaceable generation backend, not the retrieval embedding. Qwen-named VLM code or WeMM's underlying backbone does not mean Qwen3-VL-Embedding is active.
- Set `VISUAL_COLLECTION_NAME` and `SHOT_COLLECTION_NAME` consistently with preprocessing. Defaults are `visual_keyframes_v1` and `vbs_shot_index`.
- Search checks a fingerprint of model, revision, dimension and pooling recipe. Unknown, mixed or incompatible primary vectors are rejected rather than silently searched. Choose fresh visual **and** shot collections and re-embed from media; never retag old vectors or delete production data to bypass the guard.
- Optional SigLIP requires its matching named vectors and separate revision/dimension validation; it is off by default.

## Modules

| Path | Responsibility |
| --- | --- |
| `config.py`, `.env.template` | Pinned WeMM and independent VLM/detector settings. |
| `main.py`, `batch_query.py` | CLI/batch entrypoints using the same primary WeMM loader. |
| `search/hybrid_search.py` | Guarded Qdrant retrieval, payload matching, RRF, feedback and browsing. |
| `search/query_processor.py` | Replaceable VLM prompts for rewriting, expansion and clarification. |
| `search/reranker.py` | Metadata/VQA/temporal scoring and resolved-frame evidence. |
| `../models/wemm_embedding.py` | Native WeMM preprocessing, normalized text/image encoding and bounded image batches. |

## Server preparation and production VQA replay

The project requires Python 3.12 (`>=3.12,<3.13`) and committed dependency locks. CPU unit/contract checks do not establish answer quality, grounding, GPU latency or speedup. A separate V100 offline run is documented in [`evaluation/README.md`](../evaluation/README.md#recorded-run-2026-10-28-one-v100-32-gb-1765-v3c-videos), with its subset, title-lookup and annotation limitations. It does not establish a VQA accuracy/hallucination rate or official DRES performance.

### Server prerequisites

- Provision a GPU and driver/runtime compatible with the locked PyTorch/Transformers stack, with sufficient memory for the selected inference path. The BF16 WeMM checkpoint is about 10.35 GB by itself; this is not a total VRAM requirement. `capture_environment.py` records NVIDIA `nvidia-smi` metadata when available; it does not prove compute compatibility or model readiness.
- Provide a running Qdrant service, populated with the actual visual/shot collections used for the evaluation. Configure `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY` if needed, and `VISUAL_COLLECTION_NAME` / `SHOT_COLLECTION_NAME` to match the index.
- Make the indexed source media available under this checkout's `datasets/` directory. The HTTP replay sends no `dataset_dir`, so the backend resolves VQA evidence from its default repository `datasets/` root; media paths must resolve to the real indexed frames/videos.
- Match the populated index's WeMM model, revision, dimension, and pooling recipe. The current default is the pinned revision `41a4fafdd26252ddff05c025413af7e61e26f08e` at native 2560 dimensions; supported smaller MRL dimensions are 64, 128, 256, 512, and 1024. Use `WEMM_MODEL_REVISION`, `EMBEDDING_MRL_DIM`, and both collection names consistently with indexing. Runtime fingerprint checks reject unknown or incompatible populated collections. Do not assume historical 2048-dimensional vectors match the current native recipe, relabel old vectors, or bypass the guard; resolve a mismatch before replay.
- Configure an actual vision-capable VLM in the protected server environment or ignored `inference-code/.env`: `VLM_OPTION=openai` uses `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, and `OPENAI_VLM_MODEL_NAME`; `VLM_OPTION=local` uses `QWEN_VLM_MODEL_ID` and its provisioned local model. The WeMM retrieval checkpoint is still required with either VLM mode. Never commit `.env`, place credentials in commands/reports, or send keys in chat.
- `DETECTOR_OPTION=none` is an explicit supported full-frame VQA mode. If `DETECTOR_OPTION=local` is selected, the detector must load successfully; a requested detector load failure is an error, not a successful detector-backed result.
- Create a real, independently annotated manifest before evaluation. It must be UTF-8 JSON with exactly `schema_version`, `dataset_provenance`, `query_provenance`, and `queries`; use integer `schema_version: 1`, nonempty provenance declarations, and a query list whose entries contain exactly `id`, `question`, `answerability`, and `accepted_references`. IDs must be unique nonempty strings; questions nonempty; `answerability` is `answerable` or `unanswerable`; answerable entries need one or more accepted reference strings, while unanswerable entries need an empty list. Annotation provenance is recorded as a declaration, not independently verified by the replay tool.

The existing `evaluation/eval_queries.json` is fictional legacy schema data and is not a real annotation set or a valid production replay manifest.

### Locked server environment

From the repository root on the server:

```bash
uv lock --check
uv sync --python 3.12 --locked --group inference
```

The replay workflow assumes the compatible model checkpoint, real media, Qdrant index, and chosen VLM configuration are already provisioned. These commands install the existing locked software environment; they do not download model weights or rebuild an index. Configure runtime values from `inference-code/.env.template` without exposing secrets.

## CLI usage on the server

```bash
uv run --locked --group inference python inference-code/main.py --type 1 --query "a cyclist outside a building"
uv run --locked --group inference python inference-code/main.py --type 2 --query "what is the person holding?" --dataset_dir /path/to/real-media
uv run --locked --group inference python inference-code/main.py --type 3 --query "a person opens a door, then walks outside"
```

Type 3 is a legacy temporal-alignment tool, not another official VBS task label. VQA requires actual resolvable frames and a configured generation backend; schema validity/abstention is not factual correctness or DRES validation. No new model or performance result is implied by these commands.

### Production HTTP VQA replay workflow

Run from the repository root on the same server as the backend. Replace the three example path values below with distinct paths to the real annotated manifest, environment capture, and output report. Keep the manifest and report access-controlled if they contain sensitive questions or data.

In terminal B, capture the installed server environment and validate the complete replay inputs before contacting the backend:

```bash
MANIFEST=/absolute/path/to/real-independent-vqa-manifest.json
ENV_CAPTURE=/absolute/path/to/server-environment.json
REPORT=/absolute/path/to/vqa-replay.json

uv run --locked --group inference python evaluation/capture_environment.py \
  --output "$ENV_CAPTURE"

uv run --locked --group inference python evaluation/replay_vqa.py \
  --manifest "$MANIFEST" \
  --base-url http://127.0.0.1:8000 \
  --output "$REPORT" \
  --environment "$ENV_CAPTURE" \
  --validate-only
```

`--validate-only` validates the manifest, URL, timeout defaults, and optional capture without contacting the backend or writing a report. The `--output` path is still required and must not overwrite either input.

In terminal A, start the backend bound only to loopback:

```bash
uv run --locked --group inference uvicorn \
  --app-dir webapp/backend main:app \
  --host 127.0.0.1 \
  --port 8000
```

After the server is running, return to terminal B and perform the actual replay:

```bash
uv run --locked --group inference python evaluation/replay_vqa.py \
  --manifest "$MANIFEST" \
  --base-url http://127.0.0.1:8000 \
  --output "$REPORT" \
  --environment "$ENV_CAPTURE"
```

The replay sends one Type 2 `POST /api/search` request per manifest query with `reset_session: true`; it does not retry or drop failed requests. The backend may initialize the WeMM embedder, detector, or VLM lazily on the first request, so a listening server alone is not evidence of a successful model run. This local-loopback workflow does not expose the backend publicly. For a remote replay endpoint, use HTTPS; plain HTTP is accepted only for loopback.

### Reading replay evidence

- `exact_accuracy` divides exact matches by **all annotated answerable queries**. An abstention, API/runtime error, missing answer, or incorrect answer remains a miss in that denominator. `answered_exact_accuracy` has the narrower denominator of answered answerable queries and is reported separately.
- Unanswerable outcomes are counted separately; refusals are not automatically credited as correct. Hallucination/visual-grounding rates remain `null` without independent per-answer support labels; exact match is not a hallucination or grounding measure.
- The report retains per-query answer/status/raw response/evidence/latency/error, the manifest SHA-256 and declared provenance, and successful response `runtime_metadata`. That runtime metadata includes the WeMM/index identity check, VLM/model identity, configured/used VQA worker limits, detector-active flag, and nested RRF weights. The optional `--environment` capture records observed Python/platform/package/source hashes and NVIDIA metadata; it does not contain environment secrets or establish GPU compatibility.
- CPU-only unit/contract fixtures are not scientific performance or answer-quality results. Report future GPU results only from the actual server replay against the real independent manifest and matching media/index, preserving its report and environment capture. No accuracy, hallucination, speedup, or exact GPU-performance result is implied in advance.

