#!/usr/bin/env python3
"""Fetch and cache transcripts for episodes that don't have one yet.

Reads episode list from the structured cache, skips episodes that already
have transcript.txt, fetches via yt-dlp for the rest.

Best run on a residential IP — yt-dlp transcript fetching is blocked on CI.

Usage:
    uv run python scripts/fetch_transcripts.py
    uv run python scripts/fetch_transcripts.py --limit 50   # fetch N missing transcripts
    uv run python scripts/fetch_transcripts.py --video-id MtoPEub7XwA  # specific episode
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
from yt_pl_ctr.fetcher import _TRANSCRIPT_SKIP_SECONDS, fetch_transcript, is_ci
from yt_pl_ctr.models import Config
from yt_pl_ctr.youtube import YouTubeClient, YouTubeAPIError


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "cache"


def main():
    parser = argparse.ArgumentParser(description="Fetch transcripts for cached episodes")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: $YT_CACHE_DIR or ./cache)",
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent.parent / "configs" / "channels.yaml",
    )
    parser.add_argument(
        "--channel", default=None,
        help="Only fetch transcripts for this channel (by name). Applies duration and age filters.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of transcripts to fetch in this run"
    )
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="Fetch transcript for specific video ID(s) only",
    )
    args = parser.parse_args()

    if is_ci():
        print("Refusing to run on CI — yt-dlp transcript fetching is bot-blocked on CI IPs.")
        return 1

    cache_dir = args.cache_dir or _default_cache_dir()
    if not cache_dir.exists():
        print(f"Cache not found: {cache_dir}")
        print("Run: uv run python scripts/download_test_data.py")
        return 1

    # Resolve channel config for filtering
    channel_config = None
    if args.channel:
        with open(args.config) as f:
            cfg = Config.model_validate(yaml.safe_load(f))
        matches = [c for c in cfg.channels if c.name.lower() == args.channel.lower()]
        if not matches:
            names = [c.name for c in cfg.channels]
            print(f"Channel {args.channel!r} not found. Available: {names}", file=sys.stderr)
            return 1
        channel_config = matches[0]

    index = cache_mod.read_index(cache_dir)
    if not index:
        print("Index is empty — run download_test_data.py first")
        return 1

    # Repair stale index entries (file on disk but index not updated, e.g. after a killed run)
    for e in index:
        if not e.get("has_transcript") and cache_mod.has_transcript(cache_dir, e["video_id"]):
            e["has_transcript"] = True

    if args.video_ids:
        targets = [e for e in index if e["video_id"] in args.video_ids]
    else:
        targets = [e for e in index if not cache_mod.has_transcript(cache_dir, e["video_id"])]
        if channel_config:
            # Resolve channel_id from URL if not set in config
            channel_id = channel_config.channel_id
            if not channel_id:
                try:
                    youtube = YouTubeClient.from_env()
                    channel_id = youtube.resolve_channel_id(channel_config.url)
                    print(f"Resolved channel_id for {channel_config.name!r}: {channel_id}")
                except (YouTubeAPIError, Exception) as e:
                    print(f"Warning: could not resolve channel_id for {channel_config.name!r}: {e}", file=sys.stderr)
                    print("  Add channel_id to configs/channels.yaml to avoid this.", file=sys.stderr)

            # Filter by channel_id (stored in episode metadata)
            if channel_id:
                filtered = []
                for e in targets:
                    meta = cache_mod.read_metadata(cache_dir, e["video_id"])
                    if meta and meta.get("channel_id") == channel_id:
                        filtered.append(e)
                targets = filtered
                print(f"Channel filter ({channel_config.name}): {len(targets)} episodes")
            # Apply duration filter
            if channel_config.min_duration:
                before = len(targets)
                targets = [e for e in targets if (e.get("duration") or 0) >= channel_config.min_duration]
                print(f"Duration filter (>= {channel_config.min_duration}s): {len(targets)}/{before}")
            # Apply age filter
            cutoff = channel_config.min_upload_date_str()
            if cutoff:
                before = len(targets)
                targets = [e for e in targets if (e.get("upload_date") or "") >= cutoff]
                print(f"Age filter (>= {cutoff}): {len(targets)}/{before}")

    total_missing = len(targets)
    if args.limit:
        targets = targets[: args.limit]

    print(f"Cache: {cache_dir}")
    print(f"Missing transcripts: {total_missing}, fetching: {len(targets)}")
    print()

    fetched = 0
    failed = 0
    index_dirty = False

    for i, entry in enumerate(targets, 1):
        vid = entry["video_id"]
        title = entry.get("title", vid)[:60]
        print(f"  [{i}/{len(targets)}] {title} ... ", end="", flush=True)

        text = fetch_transcript(vid, max_chars=None, skip_seconds=_TRANSCRIPT_SKIP_SECONDS)
        if text:
            cache_mod.write_transcript(cache_dir, vid, text)
            entry["has_transcript"] = True
            index_dirty = True
            fetched += 1
            print(f"ok ({len(text)} chars)")
        else:
            failed += 1
            print("no transcript")

        if index_dirty and i % 10 == 0:
            cache_mod.write_index(cache_dir, index)
            index_dirty = False

    if index_dirty:
        cache_mod.write_index(cache_dir, index)

    print(f"\nDone. Fetched: {fetched}, Failed/missing: {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
