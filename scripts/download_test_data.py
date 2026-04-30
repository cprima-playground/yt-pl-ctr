#!/usr/bin/env python3
"""Download video metadata for testing the classifier (metadata only, no video files).

Two-step process:
1. Fast: Get video IDs with --flat-playlist
2. Slow: Fetch full metadata for each video (includes complete descriptions)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

CHANNEL_URL = "https://www.youtube.com/@joerogan/videos"
OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
OUTPUT_FILE = OUTPUT_DIR / "jre_videos.json"
LIMIT = 100
DELAY_SECONDS = 1  # Delay between individual video fetches


def get_video_ids(channel_url: str, limit: int) -> list[str]:
    """Fast extraction of video IDs using flat-playlist mode."""
    print(f"Step 1: Getting video IDs (fast)...")
    result = subprocess.run(
        [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            f"--playlist-end={limit}",
            channel_url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        return []

    video_ids = []
    for line in result.stdout.strip().split("\n"):
        if line:
            data = json.loads(line)
            video_ids.append(data.get("id"))
    return video_ids


def get_full_metadata(video_id: str) -> dict | None:
    """Fetch full metadata for a single video (includes complete description)."""
    result = subprocess.run(
        [
            "yt-dlp",
            "--skip-download",
            "--dump-json",
            f"https://www.youtube.com/watch?v={video_id}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    data = json.loads(result.stdout)
    return {
        "video_id": data.get("id"),
        "title": data.get("title"),
        "description": data.get("description", ""),
        "duration": data.get("duration"),
        "view_count": data.get("view_count"),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching metadata for {LIMIT} videos from {CHANNEL_URL}...")

    # Step 1: Get video IDs quickly
    video_ids = get_video_ids(CHANNEL_URL, LIMIT)
    if not video_ids:
        print("Failed to get video IDs")
        return 1

    print(f"Found {len(video_ids)} videos")

    # Step 2: Fetch full metadata for each video
    print(f"Step 2: Fetching full metadata ({DELAY_SECONDS}s delay between requests)...")
    videos = []
    for i, video_id in enumerate(video_ids, 1):
        print(f"  [{i}/{len(video_ids)}] {video_id}", end=" ", flush=True)
        metadata = get_full_metadata(video_id)
        if metadata:
            videos.append(metadata)
            print(f"- {metadata['title'][:50]}...")
        else:
            print("- FAILED")

        if i < len(video_ids):
            time.sleep(DELAY_SECONDS)

    # Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(videos, f, indent=2)

    print(f"\nSaved {len(videos)} videos to {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    exit(main())
