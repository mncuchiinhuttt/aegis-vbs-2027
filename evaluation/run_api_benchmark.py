#!/usr/bin/env python3
"""Replay queries against running AEGIS backends and log every raw response.

Each query goes through the real HTTP API (HyDE, retrieval, RRF, reranking, VQA),
one conversation per backend at a time because a backend keeps one operator
session. Pass several `--base-urls` to run that many backends in parallel.
Results are appended to a JSONL file (one record per HTTP call); scoring is a
separate step (`score_api_benchmark.py`) so hit rules can change without rerun.

Task mapping to the API: KIS-T/AVS -> /api/search type 1; KIS-C -> two type-1
turns (vague opening, then the scripted reply as `clarification_answer`);
VQA -> type 2; KIS-V -> /api/search-by-video with a clip cut from the target
video; VBSA -> type 1 with each progressive hint of a VBS-Archive textual task.
"""
import argparse
import json
import queue
import random
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

KEEP_PAYLOAD = ("source_file", "timestamp", "frame_idx", "scene_id", "sample_stride_sec")


def trim_result(result: dict) -> dict:
    payload = result.get("payload") or {}
    kept = {key: value for key, value in result.items()
            if key != "payload" and isinstance(value, (int, float, bool, type(None), list))}
    kept.update({key: (value[:300] if isinstance(value, str) else value) for key, value in result.items()
                 if key != "payload" and isinstance(value, str)})
    kept.update({key: payload[key] for key in KEEP_PAYLOAD if key in payload})
    return kept


def trim_response(body: dict) -> dict:
    out = {key: value for key, value in body.items() if key not in ("results", "retrieval_stage")}
    out["results"] = [trim_result(r) for r in (body.get("results") or [])[:20]]
    if body.get("retrieval_stage") is not None:
        out["retrieval_stage"] = [trim_result(r) for r in body["retrieval_stage"][:50]]
    return out


class Backend:
    def __init__(self, base_url: str, timeout: float, attempts: int = 3):
        self.base_url, self.timeout, self.attempts, self.http = base_url.rstrip("/"), timeout, attempts, requests.Session()

    def call(self, path: str, **kwargs) -> dict:
        """POST once, retrying only transport errors and HTTP 500s caused by the upstream VLM gateway (errors, rate limits).

        Every failed attempt is kept in `retried_errors`, so retries stay visible in the log.
        """
        started = time.perf_counter()
        record, failures = {"path": path}, []
        for attempt in range(1, self.attempts + 1):
            record.pop("error", None)
            try:
                response = self.http.post(f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
                record["http_status"] = response.status_code
                body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                if response.ok:
                    record["response"] = trim_response(body)
                    break
                record["error"] = str(body.get("detail") or response.text[:300])
                if response.status_code != 500 or not any(t in record["error"].lower() for t in ("upstream", "rate limit", "429")):
                    break
            except requests.RequestException as exc:
                record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            failures.append(record["error"])
            if attempt < self.attempts:
                time.sleep(2.0 * attempt)
        record["attempts"] = min(attempt, self.attempts)
        if failures and "response" in record:
            record["retried_errors"] = failures
        record["latency_sec"] = round(time.perf_counter() - started, 3)
        return record


def clip_bytes(video: Path, center_sec: float, length_sec: float = 6.0) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(max(0.0, center_sec - length_sec / 2)), "-t", str(length_sec),
                        "-i", str(video), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-an", str(out)], check=True)
        return out.read_bytes()


def search_body(query: str, qtype: int, **extra) -> dict:
    return {"json": {"type": qtype, "query": query, "reset_session": True, "include_retrieval_stage": qtype == 1, **extra}}


def build_jobs(args) -> list:
    jobs = []
    if args.suite:
        items = json.loads(Path(args.suite).read_text())
        for spec in args.tasks:
            task, _, count = spec.partition(":")  # "KIS-T:150" samples 150 of that task
            limit = int(count) if count else args.sample_per_task
            rng = random.Random(f"{args.seed}-{task}")  # per-task seed: a task's sample never depends on the other tasks
            pool = sorted((i for i in items if i["type_name"] == task), key=lambda i: i["id"])
            if task == "AVS":  # the 125 AVS items cover only 30 distinct queries; replay each query once
                pool = list({i["query"]: i for i in pool}.values())
            if limit and limit < len(pool):
                # VQA negatives are the only items with a known-correct abstention, so they are always kept.
                kept = [i for i in pool if task == "VQA" and i["ground_truth"].get("fail_closed_required")]
                rest = [i for i in pool if i not in kept]
                pool = sorted(kept + rng.sample(rest, max(0, limit - len(kept))), key=lambda i: i["id"])
            jobs += [{"job_id": i["id"], "task": task, "item": i, "gt": i["ground_truth"]} for i in pool]
    if args.vbs_archive:
        for entry in json.loads(Path(args.vbs_archive).read_text()):
            gt = {"video_name": f"{entry['answer']}.mp4", "start_sec": entry["videorange"]["start"] / 1000,
                  "end_sec": entry["videorange"]["end"] / 1000}
            for level, hint in enumerate(entry["hints"], start=1):
                jobs.append({"job_id": f"vbsa-{entry['query_name']}-h{level}", "task": "VBSA", "gt": gt,
                             "item": {"query": hint, "hint_level": level, "hint_count": len(entry["hints"])}})
    return jobs


def run_job(backend: Backend, job: dict, videos_dir: Path) -> list:
    task, item, calls = job["task"], job["item"], []
    if task in ("KIS-T", "AVS", "VBSA"):
        calls.append({"turn": 1, **backend.call("/api/search", **search_body(item["query"], 1))})
    elif task == "VQA":
        calls.append({"turn": 1, **backend.call("/api/search", **search_body(item["query"], 2))})
    elif task == "KIS-C":
        history = item.get("history") or [{"text": item["query"]}]
        calls.append({"turn": 1, **backend.call("/api/search", **search_body(history[0]["text"], 1))})
        reply = {"json": {"type": 1, "query": history[-1]["text"], "reset_session": False, "include_retrieval_stage": True,
                          "clarification_answer": item.get("system_answer", "")}}
        calls.append({"turn": 2, **backend.call("/api/search", **reply)})
    elif task == "KIS-V":
        clip = clip_bytes(videos_dir / job["gt"]["video_name"], float(job["gt"].get("timestamp", 0.0)))
        calls.append({"turn": 1, **backend.call("/api/search-by-video", files={"file": ("clip.mp4", clip, "video/mp4")}, data={"top_k": 20})})
    return calls


def worker(backend, jobs_queue, sink, lock, videos_dir):
    while True:
        try:
            job = jobs_queue.get_nowait()
        except queue.Empty:
            return
        try:
            calls = run_job(backend, job, videos_dir)
        except Exception as exc:  # a failed job is logged, not fatal to the run
            calls = [{"turn": 1, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}]
        record = {"job_id": job["job_id"], "task": job["task"], "backend": backend.base_url,
                  "item": {k: v for k, v in job["item"].items() if k != "ground_truth"}, "ground_truth": job["gt"], "calls": calls}
        with lock:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-urls", required=True, help="comma-separated backend URLs")
    parser.add_argument("--suite", help="legacy 1,000-query suite JSON")
    parser.add_argument("--tasks", default="KIS-T,KIS-C,AVS,KIS-V,VQA", type=lambda s: s.split(","))
    parser.add_argument("--sample-per-task", type=int, default=0, help="seeded random subset per task (0 = all)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vbs-archive", help="VBS-Archive aggregated/VBS_TextualQueries_*.json")
    parser.add_argument("--videos-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--retry-errors", action="store_true",
                        help="rerun jobs whose logged calls errored; the errored records are kept in <out>.errored-<time>.jsonl")
    args = parser.parse_args()

    done = set()
    if args.out.exists():
        records = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
        if args.retry_errors:
            failed = [r for r in records if any("error" in c for c in r["calls"])]
            if failed:
                args.out.with_name(f"{args.out.name}.errored-{int(time.time())}.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in failed))
                records = [r for r in records if r not in failed]
                args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        done = {r["job_id"] for r in records}
    jobs = [job for job in build_jobs(args) if job["job_id"] not in done]
    jobs_queue = queue.Queue()
    for job in jobs:
        jobs_queue.put(job)
    urls = args.base_urls.split(",")
    print(f"{len(jobs)} jobs ({len(done)} already done) across {len(urls)} backend(s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock, started = threading.Lock(), time.monotonic()
    with args.out.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = [pool.submit(worker, Backend(url, args.timeout), jobs_queue, sink, lock, args.videos_dir) for url in urls]
        while any(not f.done() for f in futures):
            time.sleep(30)
            print(f"[{time.monotonic() - started:.0f}s] remaining={jobs_queue.qsize()}", flush=True)
        for future in futures:
            future.result()
    print(f"done in {time.monotonic() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
