#!/usr/bin/env python3
"""Index V3C 'small' videos into Qdrant with the pinned WeMM encoder.

Every video is sampled with the same policy: up to `--frames-per-video` frames
evenly spaced over the whole video (never closer than `--min-stride-sec`), so
no video gets more retrieval chances than another. Frames are embedded with the
maintained `WeMMEmbedding4BEmbedder` and written through the maintained
`QdrantIndexer`, which stamps the embedding fingerprint the searcher requires.
Payload text comes from the V3C info JSON (title, description, tags) and the
scene id from the official master shot boundaries when present.

Usage (set VISUAL_COLLECTION_NAME to a fresh collection first):
    index_v3c_corpus.py --videos DIR --info-dirs DIR... --msb-dirs DIR... \
        --ids ids.txt --manifest-out index_manifest.json
"""
import argparse
import bisect
import collections
import html
import json
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "inference-code")]

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

from models.wemm_embedding import WeMMEmbedding4BEmbedder  # noqa: E402
from preprocessing.config import VISUAL_COLLECTION_NAME  # noqa: E402
from preprocessing.indexing.indexer import QdrantIndexer  # noqa: E402
from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: E402

TAG_RE = re.compile(r"<[^>]+>")


def clean_text(value) -> str:
    text = html.unescape(TAG_RE.sub(" ", str(value or "")))
    text = "".join(ch for ch in text if ch.isprintable() or ch.isspace())  # no control/surrogate characters
    return re.sub(r"\s+", " ", text).strip()


def load_text(info_dirs, video_id: str):
    for directory in info_dirs:
        path = directory / f"{video_id}.json"
        if path.is_file():
            info = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            title = clean_text(info.get("title"))
            tags = " ".join(clean_text(tag) for tag in (info.get("tags") or []))
            blob = " ".join(part for part in (title, clean_text(info.get("description")), tags) if part)
            return title or f"V3C video {video_id}", blob
    return f"V3C video {video_id}", ""


def load_shot_starts(msb_dirs, video_id: str) -> list:
    for directory in msb_dirs:
        path = directory / f"{video_id}.tsv"
        if path.is_file():
            starts = []
            for line in path.read_text().splitlines()[1:]:
                fields = line.split("\t")
                if len(fields) >= 3 and fields[0].isdigit():
                    starts.append(int(fields[0]))
            return starts
    return []


def sample_frames(path: Path, frames_per_video: int, min_stride_sec: float):
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError("video has no readable frames")
        count = max(1, min(frames_per_video, int((total / fps) // min_stride_sec)))
        frames = []
        for step in range(count):
            frame_idx = min(total - 1, int((step + 0.5) * total / count))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = capture.read()
            if ok:
                frames.append((frame_idx, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
        if len(frames) * 10 < count * 9:
            raise RuntimeError(f"only {len(frames)}/{count} sampled frames readable (truncated download?)")
        return fps, total, frames
    finally:
        capture.release()


def index_video(indexer, embedder, video_id: str, sampled, text, shot_starts) -> int:
    fps, total, frames = sampled
    stride_sec = round(total / fps / max(len(frames), 1), 3)
    caption, blob = text
    vectors = embedder.embed_images_batch([image for _idx, image in frames])
    for (frame_idx, _image), vector in zip(frames, vectors):
        scene = bisect.bisect_right(shot_starts, frame_idx) - 1 if shot_starts else int(frame_idx / (fps * 10))
        indexer.index_visual_point(
            str(uuid.uuid5(uuid.NAMESPACE_DNS, f"v3c_{video_id}_{frame_idx}")),
            vector,
            {
                "modality": "visual", "video_id": video_id, "source_file": f"{video_id}.mp4",
                "frame_idx": frame_idx, "timestamp": round(frame_idx / fps, 3), "scene_id": max(scene, 0),
                "caption": caption, "text_blob": blob, "asset_source": "v3c_small_uniform",
                "sample_stride_sec": stride_sec,
            },
        )
    indexer.flush_visual()
    return len(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--info-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--msb-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="index only the first N ready ids (pilot)")
    parser.add_argument("--frames-per-video", type=int, default=40)
    parser.add_argument("--min-stride-sec", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--idle-exit-sec", type=float, default=300.0, help="stop after this long with nothing ready")
    parser.add_argument("--shard", default="0/1", help="i/n: process ids[i::n] so several indexers can share one GPU")
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()

    shard, shards = (int(part) for part in args.shard.split("/"))
    pending = collections.deque([v for v in args.ids.read_text().split() if v][shard::shards])
    embedder = WeMMEmbedding4BEmbedder(batch_size=args.batch_size)
    indexer = QdrantIndexer(visual_dim=embedder.dimension, embedding_identity=embedder.embedding_identity)
    per_video, failures = {}, {}
    started = last_ready = time.monotonic()
    decoder = ThreadPoolExecutor(max_workers=3)
    inflight = collections.deque()  # (video_id, future)

    skipped = []

    def already_indexed(video_id: str) -> bool:
        flt = Filter(must=[FieldCondition(key="source_file", match=MatchValue(value=f"{video_id}.mp4"))])
        return indexer.client.count(collection_name=VISUAL_COLLECTION_NAME, count_filter=flt, exact=True).count > 0

    def refill():
        scanned = len(pending)
        while scanned and len(inflight) < 4 and (not args.limit or len(per_video) + len(inflight) < args.limit):
            video_id = pending.popleft()
            scanned -= 1
            path = args.videos / f"{video_id}.mp4"
            if not path.is_file():
                pending.append(video_id)
            elif already_indexed(video_id):
                skipped.append(video_id)  # a previous run or sibling shard finished it
            else:
                inflight.append((video_id, decoder.submit(sample_frames, path, args.frames_per_video, args.min_stride_sec)))

    while True:
        refill()
        if not inflight:
            if not pending or (args.limit and len(per_video) >= args.limit) \
                    or time.monotonic() - last_ready > args.idle_exit_sec:
                break
            time.sleep(10)
            continue
        last_ready = time.monotonic()
        video_id, future = inflight.popleft()
        try:
            text = load_text(args.info_dirs, video_id)
            per_video[video_id] = index_video(indexer, embedder, video_id, future.result(), text,
                                              load_shot_starts(args.msb_dirs, video_id))
        except Exception as exc:  # keep going; failures are recorded in the manifest
            failures[video_id] = f"{type(exc).__name__}: {str(exc)[:200]}"
        done = len(per_video) + len(failures)
        if done % 20 == 0:
            frames = sum(per_video.values())
            print(f"[{done}] videos={len(per_video)} failed={len(failures)} frames={frames} "
                  f"rate={frames / (time.monotonic() - started):.1f} frames/s pending={len(pending)}", flush=True)

    indexer.flush_visual()
    count = indexer.client.count(collection_name=VISUAL_COLLECTION_NAME, exact=True).count
    manifest = {
        "collection": VISUAL_COLLECTION_NAME, "points": count, "videos_indexed": len(per_video),
        "frames_indexed": sum(per_video.values()), "failures": failures, "not_ready": list(pending),
        "skipped_already_indexed": len(skipped), "shard": args.shard,
        "embedding_identity": embedder.embedding_identity, "embedding_fingerprint": indexer.embedding_fingerprint,
        "frames_per_video": args.frames_per_video, "min_stride_sec": args.min_stride_sec,
        "encoder_dtype": str(next(embedder.model.parameters()).dtype), "batch_size": args.batch_size,
        "wall_sec": round(time.monotonic() - started, 1), "frames_per_video_actual": per_video,
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2))
    print(f"done: {len(per_video)} videos, {count} points, {len(failures)} failures, {len(pending)} not ready", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
