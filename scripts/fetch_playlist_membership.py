#!/usr/bin/env python3
"""Bulk-fetch all YouTube playlist memberships into a local cache.

Fetches every playlist owned by the authenticated account and builds a
reverse index (video_id → [playlist_ids]) stored in cache_dir/playlist_membership.json.
Cache is valid for 12 hours; use --force to refresh immediately.

Usage:
    uv run python scripts/fetch_playlist_membership.py
    uv run python scripts/fetch_playlist_membership.py --force
    uv run python scripts/fetch_playlist_membership.py --invalidate
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

CACHE_MAX_AGE_HOURS = 12


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "cache"


def load_membership(cache_dir: Path, force: bool = False) -> dict:
    """Return membership dict, fetching from API if cache is missing/stale/forced."""
    if not force:
        cached = cache_mod.read_playlist_membership(cache_dir, max_age_hours=CACHE_MAX_AGE_HOURS)
        if cached:
            return cached

    youtube = YouTubeClient.from_env()
    data = youtube.load_all_membership()
    cache_mod.write_playlist_membership(cache_dir, data["playlists"], data["membership"])
    return cache_mod.read_playlist_membership(cache_dir, max_age_hours=9999)  # just written


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch all playlist memberships")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Ignore cache age and re-fetch")
    parser.add_argument("--invalidate", action="store_true", help="Delete cache file and exit")
    args = parser.parse_args()

    cache_dir = args.cache_dir or _default_cache_dir()

    if args.invalidate:
        removed = cache_mod.invalidate_playlist_membership(cache_dir)
        print("Removed." if removed else "No cache file found.")
        return 0

    if not args.force:
        cached = cache_mod.read_playlist_membership(cache_dir)
        if cached:
            fetched_at = cached.get("fetched_at", "unknown")
            n_pl = len(cached.get("playlists", {}))
            n_vid = len(cached.get("membership", {}))
            print(f"Cache hit (fetched {fetched_at})")
            print(f"Playlists: {n_pl}  |  Videos tracked: {n_vid}")
            _print_summary(cached)
            return 0

    print("Fetching all playlist memberships from YouTube API...")
    try:
        youtube = YouTubeClient.from_env()
    except YouTubeAPIError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    data = youtube.load_all_membership()
    cache_mod.write_playlist_membership(cache_dir, data["playlists"], data["membership"])

    n_pl = len(data["playlists"])
    n_vid = len(data["membership"])
    print(f"\nDone. Playlists: {n_pl}  |  Videos tracked: {n_vid}")
    _print_summary({"playlists": data["playlists"], "membership": data["membership"]})
    print(f"Cache: {cache_dir / 'playlist_membership.json'}")
    return 0


def _print_summary(data: dict) -> None:
    playlists = data.get("playlists", {})
    membership = data.get("membership", {})
    counts: dict[str, int] = {}
    for vids in membership.values():
        for pid in vids:
            counts[pid] = counts.get(pid, 0) + 1
    print()
    for pid, title in playlists.items():
        print(f"  {counts.get(pid, 0):4d}  {title}  ({pid})")


if __name__ == "__main__":
    sys.exit(main())
