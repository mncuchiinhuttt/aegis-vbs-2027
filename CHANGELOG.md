# Changelog

## 0.1.0 - first public release (2026-10)

Research code accompanying the AEGIS paper (Video Browser Showdown 2027 at MMM 2027). Development history before
this release lives in a private repository and was not carried over.

### System
- Primary encoder: native Tencent WeMM-Embedding-4B (pinned revision, 2,560-d, MRL prefixes 64-1024). Every point
  carries an embedding fingerprint; unknown or mixed collections are refused rather than rewritten.
- Retrieval: dense and Qdrant text-match lanes, HyDE branch, optional SigLIP lane, nested weighted reciprocal rank
  fusion (uniform default weights), temporal coherence boost and scene deduplication.
- KIS-C: history-aware query rewriting, ambiguity-triggered clarification questions, clarification-answer boost,
  negative-feedback penalty and Rocchio relevance feedback.
- VQA: evidence is resolved from real frames, optionally cropped with a detector, answered under a strict JSON
  contract, and refused (`UNKNOWN`) when evidence is missing or invalid; bounded parallel generation for API VLM clients.
- KIS-V clip/image search, in-video search, keyframe browsing and N-query temporal chain search.
- FastAPI + React operator console with a DRES client and interaction logging (not exercised against a live DRES).

### Evaluation tooling
- `evaluation/index_v3c_corpus.py`, `run_api_benchmark.py`, `score_api_benchmark.py`, `run_server_benchmark.sh` and
  `scripts/fetch_v3c_subset.py`: index a V3C subset, replay queries through the real HTTP API with one raw record per
  call, and score offline (Wilson intervals, no pooling across tasks). `/api/search` can return the pre-rerank pool
  (`include_retrieval_stage`) and the Type-1 VLM rerank can be switched off (`TYPE1_VLM_RERANK_ENABLED=false`) for ablations.
- Model-free VQA replay client with strict manifest validation, plus server environment capture.
- Recorded run (V100, 1,765 V3C videos, title-only text index): see `evaluation/README.md` for numbers and limits.

### Fixes found while running on a GPU server
- WeMM chose emulated bfloat16 on pre-Ampere GPUs (several times slower); it now requires native support, else fp16.
- Local Qwen batched generation used right padding and decoded garbage for shorter prompts.
- Importing ultralytics made `Image.open` raise `ModuleNotFoundError` for video files, which crashed VQA evidence resolution.
- Type-1 rerank scoring runs concurrently for API VLM clients (identical ranking to the serial path).
- Gateways that append text to replies, or rate-limit concurrent calls, are handled by `OPENAI_VLM_STRIP_REGEX` and
  `OPENAI_VLM_MAX_RETRIES`.

### Release hardening
- Backend, Vite dev server and CORS default to localhost (`AEGIS_HOST`, `VITE_HOST`, `VITE_ALLOWED_HOSTS`,
  `CORS_ALLOW_ORIGINS`); `run_webapp.py` no longer terminates processes on busy ports unless `--free-ports` is given.
- `download_assets.py` no longer downloads ffmpeg binaries or a legacy dataset. Full preprocessing requires
  explicit acceptance of NTT's evaluation-only terms; the runtime is fetched from a pinned licensor source,
  SHA-256 checked, and excluded from this release (see `NOTICE.md`). Required SAM3 access failures remain errors.
- Qdrant Docker images are pinned to the tested v1.10.1; all launch modes bind to localhost, and failed
  Docker startup falls back to standalone startup with a bounded readiness check instead of claiming success.

### Known limitations
- The benchmark text index holds only V3C titles, descriptions and tags; the title-based VLM rerank lowers recall on
  independent visual queries, and the legacy suite's VQA "negative" labels are not valid ground truth.
- No official VBS/DRES score exists for this code. One backend process serves one operator session.
- `webapp/backend/main.py` and `webapp/frontend/src/App.tsx` are large single files.
