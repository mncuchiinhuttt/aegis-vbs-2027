#!/usr/bin/env python3
"""Download selected V3C 'small' videos over SFTP with parallel curl.

Credentials are read by curl from a config file (`--curlrc`, e.g. a chmod-600
file containing `user = "name:password"`), never from argv or the environment,
so they cannot show up in `ps`, shell history or logs.

Usage:
    fetch_v3c_subset.py --ids ids.txt --out datasets/v3c/videos \
        --curlrc ~/aegis-run/inputs/v3c.curlrc --sizes remote_sizes.json --workers 64

`ids.txt` holds one V3C id per line (e.g. 00001); order is download priority.
`--sizes` is a JSON {id: bytes} built from the remote `ls -l` listing
(`curl -K curlrc sftp://ftp.itec.aau.at/V3C/small/`). With it a file counts as
done only when its size matches exactly; truncated finished files and
over-long partial files are discarded and fetched again. Without it a curl exit
status of 0 is trusted, which is how truncated videos slipped through once.
Each result is appended to `<out>/../fetch_manifest.jsonl`; re-running skips
verified videos and resumes valid partial ones.
"""
import argparse
import json
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SFTP_BASE = "sftp://ftp.itec.aau.at/V3C"
MIN_VIDEO_BYTES = 1 << 20


def size_ok(path: Path, expected) -> bool:
    return path.is_file() and path.stat().st_size >= MIN_VIDEO_BYTES and (expected is None or path.stat().st_size == expected)


def fetch_one(video_id: str, out_dir: Path, curlrc: Path, flavor: str, retries: int, expected) -> dict:
    final = out_dir / f"{video_id}.mp4"
    if size_ok(final, expected):
        return {"id": video_id, "status": "cached", "bytes": final.stat().st_size, "sec": 0.0}
    final.unlink(missing_ok=True)  # finished but wrong size: truncated or corrupt
    part = final.with_name(final.name + ".part")
    started = time.monotonic()
    last_error = ""
    for attempt in range(1, retries + 1):
        if expected is not None and part.is_file() and part.stat().st_size > expected:
            part.unlink()  # over-long partial file cannot be resumed
        command = [
            "curl", "-K", str(curlrc), "-C", "-", "--max-time", "1800",
            "-o", str(part), f"{SFTP_BASE}/{flavor}/{video_id}.mp4",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and size_ok(part, expected):
            part.replace(final)
            return {
                "id": video_id, "status": "ok", "bytes": final.stat().st_size,
                "sec": round(time.monotonic() - started, 1), "attempt": attempt,
            }
        last_error = (result.stderr or f"exit {result.returncode}").strip()[-200:]
        if result.returncode == 0:
            last_error = f"size {part.stat().st_size if part.is_file() else 0} != expected {expected}"
        # The AAU SFTP host throttles bursts of new sessions; back off with jitter.
        time.sleep(min(30.0, 2.0 * attempt * attempt) + random.uniform(0, 2))
    return {"id": video_id, "status": "failed", "error": last_error, "sec": round(time.monotonic() - started, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--curlrc", type=Path, required=True)
    parser.add_argument("--sizes", type=Path, help="JSON {id: expected_bytes}; enables exact size verification")
    parser.add_argument("--flavor", default="small", choices=["small", "verysmall"])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()

    if not args.curlrc.is_file():
        parser.error(f"curl config not found: {args.curlrc}")
    sizes = json.loads(args.sizes.read_text()) if args.sizes else {}
    video_ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out.parent / "fetch_manifest.jsonl"
    lock = threading.Lock()
    counts = {"ok": 0, "cached": 0, "failed": 0}
    total_bytes = 0
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool, manifest_path.open("a", encoding="utf-8") as manifest:
        futures = [pool.submit(fetch_one, vid, args.out, args.curlrc, args.flavor, args.retries, sizes.get(vid))
                   for vid in video_ids]
        for future in as_completed(futures):
            record = future.result()
            with lock:
                counts[record["status"]] += 1
                total_bytes += record.get("bytes", 0) if record["status"] == "ok" else 0
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                done = sum(counts.values())
                elapsed = time.monotonic() - started
                if done % 25 == 0 or record["status"] == "failed":
                    rate = total_bytes / max(elapsed, 1e-6) / 1e6
                    print(f"[{done}/{len(video_ids)}] ok={counts['ok']} cached={counts['cached']} "
                          f"failed={counts['failed']} fetched={total_bytes / 1e9:.2f}GB rate={rate:.2f}MB/s", flush=True)
    print(f"done in {time.monotonic() - started:.0f}s: {counts}", flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
