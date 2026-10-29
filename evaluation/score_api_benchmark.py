#!/usr/bin/env python3
"""Score `run_api_benchmark.py` logs into task-separated metrics.

Usage: score_api_benchmark.py LABEL=results.jsonl [LABEL=results2.jsonl ...] --out summary.json

Hit rules (applied here, never at run time):
  * video hit   - a returned frame comes from the ground-truth video.
  * temporal hit (VBS-Archive only) - same video and the frame's covered interval
    [t - stride/2, t + stride/2] overlaps the ground-truth range; `strict` means t
    itself lies inside the range.
Ranks are positions in the list the API returned (type 1 returns at most 10).
MRR counts 0 for a miss. Recall values carry Wilson 95% intervals. Nothing is
averaged across tasks.
"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

KS = (1, 5, 10, 20)


def canon(value) -> str:
    return str(value or "").replace(".mp4", "").strip()


def wilson(hits: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def first_rank(results: list, predicate):
    return next((i for i, r in enumerate(results, 1) if predicate(r)), None)


def temporal_match(result: dict, gt: dict, strict: bool) -> bool:
    if canon(result.get("source_file")) != canon(gt["video_name"]) or result.get("timestamp") is None:
        return False
    t, half = float(result["timestamp"]), (0.0 if strict else float(result.get("sample_stride_sec") or 0.0) / 2)
    return t + half >= gt["start_sec"] and t - half <= gt["end_sec"]


def rank_block(ranks: list) -> dict:
    n = len(ranks)
    block = {"n": n, "mrr": round(sum(1 / r for r in ranks if r) / n, 4) if n else None}
    for k in KS:
        hits = sum(1 for r in ranks if r and r <= k)
        block[f"R@{k}"] = [round(hits / n, 4) if n else None, *wilson(hits, n)]
    return block


def latency_block(values: list) -> dict:
    values = sorted(v for v in values if v is not None)
    if not values:
        return {}
    return {"median": round(statistics.median(values), 2), "p95": round(values[min(len(values) - 1, int(0.95 * len(values)))], 2),
            "mean": round(statistics.fmean(values), 2)}


def summarize(records: list, source: str = "results") -> dict:
    by_task = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    out = {}
    for task, rows in sorted(by_task.items()):
        calls = [c for r in rows for c in r["calls"]]
        if source != "results" and not any((c.get("response") or {}).get(source) is not None for c in calls):
            continue  # this task has no such pool (VQA, KIS-V)
        summary = {"jobs": len(rows), "calls": len(calls), "http_errors": sum(1 for c in calls if "error" in c),
                   "calls_retried_then_ok": sum(1 for c in calls if c.get("retried_errors")),
                   "empty_result_calls": sum(1 for c in calls if "response" in c and not (c["response"].get(source)))}
        summary["latency_per_call_sec"] = latency_block([c.get("latency_sec") for c in calls])
        if task == "VQA":
            out[task] = {**summary, **vqa_block(rows)}
            continue
        turns = sorted({c["turn"] for r in rows for c in r["calls"]})
        for turn in turns:
            ranks_video, ranks_t, ranks_strict, distinct, precision = [], [], [], [], []
            for r in rows:
                call = next((c for c in r["calls"] if c["turn"] == turn), None)
                if call is None or "response" not in call:
                    ranks_video.append(None); ranks_t.append(None); ranks_strict.append(None)
                    continue
                results, gt = call["response"].get(source) or [], r["ground_truth"]
                relevant = {canon(gt["video_name"]), *map(canon, gt.get("distinct_target_videos", []))}
                ranks_video.append(first_rank(results, lambda x: canon(x.get("source_file")) in relevant))
                if task == "AVS" and results[:10]:
                    precision.append(sum(canon(x.get("source_file")) in relevant for x in results[:10]) / len(results[:10]))
                if task == "VBSA":
                    ranks_t.append(first_rank(results, lambda x: temporal_match(x, gt, False)))
                    ranks_strict.append(first_rank(results, lambda x: temporal_match(x, gt, True)))
                top = results[:10]
                if top:
                    distinct.append(len({canon(x.get("source_file")) for x in top}) / len(top))
            block = {"video": rank_block(ranks_video), "mean_distinct_video_ratio_top10": round(statistics.fmean(distinct), 3) if distinct else None}
            if task == "AVS":
                block["mean_precision_top10_listed_relevant"] = round(statistics.fmean(precision), 3) if precision else None
            if task == "VBSA":
                block["temporal_overlap"], block["temporal_strict"] = rank_block(ranks_t), rank_block(ranks_strict)
                block["by_hint_level"] = hint_levels(rows, turn, source)
            if task == "KIS-C":
                answered = [next((c for c in r["calls"] if c["turn"] == turn), {}).get("response", {}) for r in rows]
                block["clarification_question_rate"] = round(sum(1 for a in answered if a.get("clarification")) / len(rows), 3)
                block["clarification_boost_applied_rate"] = round(sum(1 for a in answered if a.get("clarification_boost_applied")) / len(rows), 3)
            summary[f"turn{turn}"] = block
        summary["latency_per_job_sec"] = latency_block([sum(c.get("latency_sec", 0) for c in r["calls"]) for r in rows])
        out[task] = summary
    return out


def hint_levels(rows: list, turn: int, source: str = "results") -> dict:
    levels = defaultdict(lambda: ([], []))
    for r in rows:
        call = next((c for c in r["calls"] if c["turn"] == turn), None)
        results = (call or {}).get("response", {}).get(source) or []
        level = r["item"]["hint_level"]
        levels[level][0].append(first_rank(results, lambda x: canon(x.get("source_file")) == canon(r["ground_truth"]["video_name"])))
        levels[level][1].append(first_rank(results, lambda x: temporal_match(x, r["ground_truth"], False)))
    return {f"hint{level}": {"video": rank_block(v), "temporal_overlap": rank_block(t)} for level, (v, t) in sorted(levels.items())}


def vqa_block(rows: list) -> dict:
    status, groups = Counter(), {"answerable_by_label": [], "negative": []}
    for r in rows:
        call = r["calls"][0]
        first = ((call.get("response") or {}).get("results") or [{}])[0]
        negative = bool(r["ground_truth"].get("fail_closed_required"))
        valid = bool(first.get("vqa_answer_valid"))
        status[first.get("vqa_status") or ("http_error" if "error" in call else "none")] += 1
        groups["negative" if negative else "answerable_by_label"].append(
            {"valid": valid, "evidence": bool(first.get("vqa_evidence_available")),
             "video_hit": canon(first.get("evidence_media_name")) == canon(r["ground_truth"].get("video_name"))})
    def block(items):
        n = len(items)
        return {"n": n, "answered": sum(i["valid"] for i in items), "abstained_or_error": n - sum(i["valid"] for i in items),
                "evidence_available": sum(i["evidence"] for i in items), "answer_evidence_video_matches_target": sum(i["valid"] and i["video_hit"] for i in items)}
    return {"status_counts": dict(status), "negative_questions": block(groups["negative"]), "other_questions": block(groups["answerable_by_label"]),
            "note": "No answer-correctness label exists for these questions; only abstention, evidence and negative-case behaviour are measured."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="LABEL=results.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        report[label] = {"source": path, "jobs": len(records), "tasks": summarize(records),
                         "retrieval_stage_tasks": summarize(records, "retrieval_stage")}
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({label: {task: {k: v for k, v in s.items() if k.startswith("turn") or k in ("jobs", "http_errors")} for task, s in run["tasks"].items()}
                      for label, run in report.items()}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
