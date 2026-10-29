# Evaluation & Benchmarking Module

---

## Overview

Evaluation has two distinct paths:

- **Legacy offline runner (`run_eval.py`)** loads the inherited retrieval stack directly. Type 1 and Type 3 are legacy diagnostics, not official VBS 2027 scores. Type 2 now reports the answer/status/evidence produced by `rerank_type2_vqa` itself; it does not make a second full-frame answer call. Optional Ragas judge metrics remain separate diagnostics, are never exact-match scores, and are `N/A` when the judge dependency or call is unavailable.
- **Current production replay (`replay_vqa.py`)** sends each annotated question to the running backend's `/api/search` Type 2 endpoint with `reset_session=true`. It measures the actual first answer-bearing result and preserves per-query status, raw response, evidence, and successful response `runtime_metadata`.

The repository's `evaluation/eval_queries.json` is a fictional legacy schema demonstration, not a labeled dataset and not a valid production replay manifest. It must not be used to claim quality or latency.

## VBS 2027 evaluation contract

The official VBS 2027 call defines KIS-V, KIS-T, KIS-C, AVS, and VQA tasks over the V3C collection plus marine and laparoscopic data. The collection is available ahead of the event, but the live queries and task outcomes are not an offline test set. See the [official VBS call](https://videobrowsershowdown.org/call-for-papers/), [existing data and tools](https://videobrowsershowdown.org/about-vbs/existing-data-and-tools/), and [DRES communication guide](https://videobrowsershowdown.org/about-vbs/communication-with-dres/) before creating the final manifest.

Use two clearly labelled tracks:

1. **Live DRES track:** preserve evaluation/task/session IDs, task clock, ordered query and clarification turns, candidate IDs shown, every submission, correct/false or judge outcome, and the official score/rank. Report time-to-first, time-to-first-correct, false submissions, and task-specific diversity/evidence fields.
2. **Offline replay track:** freeze the accessible dataset manifest, index snapshot, model/configuration revisions, query labels, raw ranked outputs, and latency traces. Report retrieval and grounding metrics only as replay diagnostics; never call them the VBS 2027 leaderboard score.

Task-specific replay/live diagnostics should be:

- **KIS-V/T/C:** target video and frame/interval hit, Recall@K/MRR for replay, time-to-correct and false submissions for live; keep KIS-C clarification turns in the same session.
- **AVS:** accepted shots/source videos/ranges, unique-video coverage, duplicate rate, and live judge outcomes. Do not substitute ordinary Recall@K when accepted labels are not complete.
- **VQA:** candidate localization, answer correctness or judge outcome, grounded-answer rate, evidence identity parity, and answer latency.

VBS 2027 does not advertise a novice session. Any internal usability test should be named as an internal replay or operator study, not as an official novice score. The [2025 VBS results](https://arxiv.org/html/2509.12000v1) are a historical evaluation precedent, not a guarantee of the 2027 timing or scoring configuration.

---

## Production HTTP VQA replay

The replay client is model-free: it imports only the evaluation helpers and Python standard library. It never initializes the backend, VLM, embedder, detector, or Ragas. Supply a real, explicitly annotated manifest and a user-selected output path:

```bash
python evaluation/replay_vqa.py \
  --manifest /path/to/your-annotated-manifest.json \
  --base-url http://127.0.0.1:8000 \
  --output /path/to/your-vqa-replay.json
```

HTTP is permitted only for localhost/loopback; use HTTPS for a remote backend. The CLI performs exactly one request per query, records each request/API failure in the outcome denominator, and does not retry or drop failures. `--validate-only` validates the complete manifest and optional environment capture without contacting the backend or writing a report:

```bash
python evaluation/replay_vqa.py \
  --manifest /path/to/your-annotated-manifest.json \
  --base-url http://127.0.0.1:8000 \
  --output /path/to/your-vqa-replay.json \
  --validate-only
```

### Replay manifest contract

The required JSON object has `schema_version: 1`, nonempty free-form `dataset_provenance` and `query_provenance` declarations, and a `queries` list. Provenance text is preserved as a user declaration and marked **not independently verified**; it is not validated as a dataset/year/source enum.

Each query must contain exactly:

| Field | Requirement |
| --- | --- |
| `id` | Unique, nonempty string |
| `question` | Nonempty string |
| `answerability` | `answerable` or `unanswerable` |
| `accepted_references` | For answerable queries, one or more explicitly accepted nonempty answer strings; for unanswerable queries, an empty list |

Schema illustration only (placeholders, not an example dataset):

```json
{
  "schema_version": 1,
  "dataset_provenance": "<free-form dataset declaration>",
  "query_provenance": "<free-form query/annotation declaration>",
  "queries": [
    {
      "id": "<unique annotation id>",
      "question": "<annotated question>",
      "answerability": "answerable",
      "accepted_references": ["<explicitly accepted reference string>"]
    }
  ]
}
```

### Metrics and report

Exact match uses Unicode NFKC normalization, casefolding, and whitespace normalization only. It does not use substrings, word overlap, LLM-as-exact-match, or inferred semantic credit. `exact_accuracy` divides exact correct answers by **all labeled answerable questions**, counting abstentions, request errors, and incorrect/missing answers as misses. `answered_exact_accuracy` is restricted to answered, labeled answerable questions and reports its own denominator. Unanswerable cases are counted separately; refusals are never automatically credited. Empty denominators are JSON `null`. Hallucination and visual-grounding rates are `null` without independent per-answer support labels. Ragas diagnostics from the legacy runner are separate.

The report contains the annotation manifest SHA-256 and declared provenance, the backend URL, per-query answer/status/raw response/evidence/latency/error, the actual successful response's `runtime_metadata`, optional server-environment capture, and explicit metric numerators/denominators. No report is created until a real replay is run.

### Server environment capture

On the server host and in the backend's Python environment, capture observed Python/platform/package/source-hash and optional NVIDIA GPU/driver/VRAM information:

```bash
python evaluation/capture_environment.py --output /path/to/server-environment.json
```

The capture command uses only the standard library, does not import model packages, and emits explicit unavailable values when `nvidia-smi` or device data is absent. Pair a capture with the replay using `--environment /path/to/server-environment.json`; the replay records its SHA-256 and captured contents. This is host information observed at capture time, not an inference drawn from configuration.

## API-boundary retrieval benchmark (V3C corpus on a GPU server)

Three steps run the real system end to end and keep every raw response:

1. **Index** - `evaluation/index_v3c_corpus.py` samples every video identically (up to 40 frames evenly spread over the whole video, never closer than 3 s), embeds frames with the pinned WeMM encoder and writes them through the maintained `QdrantIndexer` (embedding fingerprint included). Payload text is the V3C info JSON (title, description, tags) and the scene id comes from the official master shot boundaries. There is **no OCR/ASR/object/VLM-caption enrichment**, so the Type-1 rerank scores the video title only. Use a fresh `VISUAL_COLLECTION_NAME`; several `--shard i/n` processes can share one GPU and skip already indexed videos. Download videos with `scripts/fetch_v3c_subset.py` (credentials only via a curl config file).
2. **Replay** - `evaluation/run_server_benchmark.sh LABEL COLLECTION "ENV=VAL" -- <runner args>` starts backends, warms them up and runs `evaluation/run_api_benchmark.py`, which sends each query through `/api/search` (KIS-T/AVS: type 1; KIS-C: two type-1 turns with the scripted reply as `clarification_answer`; VQA: type 2; KIS-V: `/api/search-by-video` with a 6 s clip cut from the target video; VBS-Archive textual tasks: type 1 per progressive hint). Type-1 requests also ask for the pre-rerank pool (`include_retrieval_stage`), so end-to-end ranking and retrieval-stage ranking come from the same HyDE text and candidates. Transport errors, upstream-gateway HTTP 500s and gateway rate limits are retried (visible as `retried_errors`); everything else is logged as an error, and `--retry-errors` reruns only errored jobs while keeping the failed records. Keep `VQA_MAX_WORKERS` low enough for the gateway: a hosted router that allowed about 5 concurrent calls turned 8 workers on two backends into HTTP 429s for about 95% of Type-1 requests.
3. **Score** - `evaluation/score_api_benchmark.py LABEL=results.jsonl --out summary.json` applies the hit rules offline: video hit, and for VBS-Archive tasks temporal overlap of the frame's covered interval (timestamp +/- half its sampling stride) with the human-labelled range. Recall values carry Wilson 95% intervals; tasks are never pooled.

Limits to quote with any number: the API returns at most 10 reranked Type-1 results (R@20 is only available from the retrieval stage); title-derived suite queries and scripted KIS-C replies share the indexed title text, so they measure lexical lookup, not visual retrieval; the legacy suite has no answer labels for VQA (only abstention, evidence and negative-question behaviour are scored); KIS-V clips are cut from the indexed file, not screen-recorded.

Inputs for the recorded run below: `evaluation/data/v3c_benchmark_video_ids.txt` (the 1,765 V3C ids: every video that the VBS-Archive tasks and the legacy suite point to, plus the remaining V3C1 ids up to 01705) and `evaluation/data/vbs_rag_benchmark_1000.json` (the legacy 1,000-query suite). The suite's KIS-T/KIS-C/AVS/KIS-V/VQA queries were generated from the indexed title text of the corpus, so they measure title lookup, and its VQA "negative" labels are not valid abstention ground truth (see findings). The independent queries are the 63 human-written textual tasks of the public [VBS-Archive](https://github.com/lucaro/VBS-Archive) (`aggregated/VBS_TextualQueries_19-24.json`, not copied here). V3C videos are distributed by NIST under a data agreement; see [VBS data access](https://videobrowsershowdown.org/about-vbs/existing-data-and-tools/). Example:

```bash
uv run --group inference python evaluation/index_v3c_corpus.py --videos <dir> --info-dirs <v3c info json dirs> \
    --ids evaluation/data/v3c_benchmark_video_ids.txt --manifest-out index_manifest.json   # fresh VISUAL_COLLECTION_NAME
VLM_ENV_FILE=<env file with OPENAI_*> evaluation/run_server_benchmark.sh main <collection> "" -- \
    --suite evaluation/data/vbs_rag_benchmark_1000.json --tasks KIS-V,AVS,KIS-C:60,KIS-T:150,VQA:40 \
    --vbs-archive <VBS_TextualQueries_19-24.json>
```

### Recorded run (2026-10-28, one V100 32 GB, 1,765 V3C videos)

Index `aegis_v3c_wemm_k40`: 70,600 points (1,765 videos x 40 frames), WeMM-Embedding-4B revision `41a4fafd...`, fp16, 2,560-d, one embedding fingerprint, title/description/tags as the only text. Hosted VLM `gemini-3.8-flash` through an OpenAI-compatible gateway; two backends. Video-level hit, Wilson 95% intervals in brackets (raw per-query logs are not included in this repository):

| Queries | Condition | R@1 | R@10 |
|---|---|---|---|
| VBS-Archive hints (n=189, 63 human-written tasks x 3 hints) | dense only (lexical off, no rerank) | 0.757 [0.69-0.81] | 0.852 [0.79-0.90] |
| | lexical on, no rerank | 0.762 [0.70-0.82] | 0.862 [0.81-0.90] |
| | full pipeline (title rerank on) | 0.354 [0.29-0.42] | 0.503 [0.43-0.57] |
| same, frame covers the labelled range | dense only / lexical on / full | 0.566 / 0.577 / 0.095 | 0.778 / 0.788 / 0.418 |
| KIS-T, title-derived suite queries (n=150) | dense only | 0.540 [0.46-0.62] | 0.733 [0.66-0.80] |
| | lexical on, no rerank | 0.627 [0.55-0.70] | 0.993 [0.96-1.00] |
| | full pipeline | 0.967 [0.92-0.99] | 0.987 [0.95-1.00] |
| KIS-C, scripted two turns (n=60) | full: turn 1 -> turn 2 | 0.483 -> 0.650 | 0.483 -> 0.650 |
| AVS (30 distinct queries) | full | 0.367 | 0.633 |
| KIS-V (n=75, clips cut from the indexed file) | `/api/search-by-video` | 0.933 | 0.947 |

Findings: (1) on independent human-written queries the title-only VLM rerank removes correct results (33 of 189 calls returned nothing; R@10 0.86 -> 0.50), because the index has no frame-level text for it to read; (2) the title-derived suite queries are answered by the lexical lane and the title rerank (R@10 0.73 -> 0.99, R@1 0.63 -> 0.97), so they measure title lookup; (3) VQA (n=40): 25 answered, 15 abstained. Of the 26 legacy "negative" questions 12 were answered, and a visual check of the evidence frames (by an AI annotator, not human-verified) shows most contain the asked object (submarine, volcano, space shuttle, dragon, flying saucer), so those labels are not valid abstention ground truth; (4) median client latency per call was 22 s for the full KIS-T path (3 workers per backend under a rate-limited gateway), 6-7 s without rerank, 67 s for VQA. The corpus is 1,765 videos, much smaller than the 28k-video VBS collection, so absolute recall is optimistic.

## Legacy offline runner and archived examples

`run_eval.py` is the inherited direct/offline harness. Its `--query_file`, `--dataset_dir`, and `--output_file` options remain for compatibility; Type 1/Type 3 retrieval/temporal metrics are diagnostics, and Type 2's optional Ragas metrics are judge-based diagnostics. This path is not the production `/api/search` replay and is not the official DRES scorer.

The existing `evaluation/eval_queries.json` contains fictional Type 1/2/3 records solely to illustrate the older `ground_truth` schema. Its video names, questions, answers, timestamps, and event frames are not verified annotations or measured outputs. Do not use it to report benchmark accuracy or latency. The production replay requires the separate manifest schema above and a real annotation set.

---

