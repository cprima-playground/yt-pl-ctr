#!/usr/bin/env python3
"""Fetch video metadata for any configured channel via YouTube Data API.

Writes to a structured cache directory (default: $YT_CACHE_DIR):
  index.json              — lightweight episode list
  episodes/{id}/
    metadata.json         — full VideoMetadata

Incremental and idempotent — safe to interrupt and resume.

Usage:
    # Fetch metadata for default (first) channel
    uv run python scripts/download_test_data.py --limit 2500

    # Fetch metadata for a specific channel (applies max_age_days from config)
    uv run python scripts/download_test_data.py --channel "Candace Owens"

    # Then fetch transcripts
    uv run python scripts/fetch_transcripts.py --channel "Candace Owens"
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cache as cache_mod
from yt_pl_ctr.fetcher import YouTubeAPIFetcher
from yt_pl_ctr.models import Config
from yt_pl_ctr.youtube import YouTubeAPIError, YouTubeClient


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "cache"


def _default_config() -> Path:
    return Path(__file__).parent.parent / "configs" / "channels.yaml"


def main():
    parser = argparse.ArgumentParser(
        description="Fetch channel video metadata into structured cache"
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--channel", default=None,
        help="Channel name from config (default: first channel)",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max videos to fetch (default: all within age window)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N videos in listing")
    args = parser.parse_args()

    config_path = args.config or _default_config()
    with open(config_path, encoding="utf-8") as f:
        config = Config.model_validate(yaml.safe_load(f))

    if args.channel:
        matches = [c for c in config.channels if c.name.lower() == args.channel.lower()]
        if not matches:
            names = [c.name for c in config.channels]
            print(f"Channel {args.channel!r} not found. Available: {names}", file=sys.stderr)
            return 1
        channel_config = matches[0]
    else:
        channel_config = config.channels[0]

    cutoff = channel_config.min_upload_date_str()
    limit = args.limit or channel_config.ingest_limit

    print(f"Channel : {channel_config.name}")
    print(f"URL     : {channel_config.url}")
    print(f"Cutoff  : {cutoff or 'none'}")
    print(f"Limit   : {limit}")

    cache_dir = args.cache_dir or _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache   : {cache_dir}")
    print()

    try:
        youtube = YouTubeClient.from_env()
        info = youtube.get_channel_info()
        print(f"Authenticated as: {info.get('title')}")
    except YouTubeAPIError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Set YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN in .env", file=sys.stderr)
        return 1

    fetcher = YouTubeAPIFetcher(youtube)

    existing_ids = cache_mod.known_ids(cache_dir)
    index = cache_mod.read_index(cache_dir)
    index_by_id = {e["video_id"]: e for e in index}
    print(f"Already cached: {len(existing_ids)} episodes total")
    print("Fetching... (writes incrementally, safe to interrupt)")
    print()

    new_count = 0
    stopped_at_cutoff = False
    try:
        for i, video in enumerate(
            fetcher.fetch_channel_videos(channel_config.url, limit=limit, offset=args.offset), 1
        ):
            # Videos arrive newest-first; stop when we exceed the age window
            if cutoff and video.upload_date and video.upload_date < cutoff:
                print(f"  [{i}] Reached age cutoff ({video.upload_date} < {cutoff}) — stopping")
                stopped_at_cutoff = True
                break

            if video.video_id in existing_ids:
                print(f"  [{i}] SKIP {video.video_id} (already cached)")
                continue

            entry = {
                "video_id": video.video_id,
                "title": video.title,
                "description": video.description,
                "duration": video.duration,
                "upload_date": video.upload_date,
                "view_count": video.view_count,
                "channel_name": video.channel_name,
                "channel_id": video.channel_id,
                "tags": video.tags,
            }

            cache_mod.write_metadata(cache_dir, entry)
            index_by_id[video.video_id] = cache_mod.index_entry(entry)
            existing_ids.add(video.video_id)
            new_count += 1
            print(f"  [{i}] {video.title[:70]}", flush=True)

            if new_count % 10 == 0:
                cache_mod.write_index(cache_dir, list(index_by_id.values()))

    except KeyboardInterrupt:
        print("\nInterrupted.")

    cache_mod.write_index(cache_dir, list(index_by_id.values()))
    print(f"\nDone. New: {new_count}, Total in cache: {len(index_by_id)}")
    if cutoff and not stopped_at_cutoff:
        print(f"Note: age cutoff ({cutoff}) was not reached — all fetched videos are within window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
