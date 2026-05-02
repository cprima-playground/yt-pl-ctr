#!/usr/bin/env python3
"""Backfill tags into cached metadata.json using YouTube Data API.

Episodes downloaded before tags were added to the fetch pipeline have no
tags field. This script batches them 50 at a time and updates metadata.json
in place.

Usage:
    uv run python scripts/enrich_metadata.py
    uv run python scripts/enrich_metadata.py --limit 500
    uv run python scripts/enrich_metadata.py --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cache as cache_mod
from yt_pl_ctr.youtube import YouTubeAPIError, YouTubeClient


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "cache"


def _needs_tags(cache_dir: Path, video_id: str) -> bool:
    meta = cache_mod.read_metadata(cache_dir, video_id)
    if meta is None:
        return False
    return not meta.get("tags")


def main():
    parser = argparse.ArgumentParser(description="Backfill YouTube tags into cached metadata")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cache_dir = args.cache_dir or _default_cache_dir()
    if not cache_dir.exists():
        print(f"Cache not found: {cache_dir}")
        return 1

    try:
        youtube = YouTubeClient.from_env()
        info = youtube.get_channel_info()
        print(f"Authenticated as: {info.get('title')}")
    except YouTubeAPIError as e:
        print(f"Auth error: {e}", file=sys.stderr)
        print("Set YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN in .env", file=sys.stderr)
        return 1

    index = cache_mod.read_index(cache_dir)
    targets = [e["video_id"] for e in index if _needs_tags(cache_dir, e["video_id"])]

    if args.limit:
        targets = targets[: args.limit]

    print(f"Cache: {cache_dir}")
    print(f"Episodes needing tags: {len(targets)}")
    if args.dry_run:
        print("(dry run — no writes)")
    print()

    updated = 0
    batch_size = 50

    for batch_start in range(0, len(targets), batch_size):
        batch_ids = targets[batch_start : batch_start + batch_size]
        batch_end = batch_start + len(batch_ids)
        print(f"  Batch {batch_start + 1}-{batch_end}/{len(targets)} ... ", end="", flush=True)

        try:
            videos = youtube.get_videos_metadata(batch_ids)
        except Exception as e:
            print(f"error: {e}")
            continue

        tags_by_id = {v.video_id: v.tags for v in videos}

        for vid in batch_ids:
            tags = tags_by_id.get(vid, [])
            meta = cache_mod.read_metadata(cache_dir, vid)
            if meta is None:
                continue
            meta["tags"] = tags
            if not args.dry_run:
                cache_mod.write_metadata(cache_dir, meta)
            updated += 1

        print(f"ok ({len(videos)} returned)")

    print(f"\nDone. Updated: {updated} episodes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
