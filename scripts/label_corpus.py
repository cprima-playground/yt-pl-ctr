#!/usr/bin/env python3
"""Generate silver labels from high-confidence classifier signals.

Only keeps labels where the match reason is deterministic:
  - guest: exact guest name match from config
  - priority_keyword: explicit title pattern (e.g. "JRE MMA Show")
  - title_keyword: title-only keyword match (higher precision than description)

Discards: wikipedia, description_keyword, description_pattern, transcript_keyword, default.

Outputs:
  labeled.json    — high-confidence labeled episodes
  unlabeled.json  — episodes with no confident label (candidates for LLM labeling)

Usage:
    uv run python scripts/label_corpus.py
    uv run python scripts/label_corpus.py --min-duration 3600
    uv run python scripts/label_corpus.py --show-stats
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cache as cache_mod
from yt_pl_ctr.classifier import VideoClassifier
from yt_pl_ctr.models import Config, VideoMetadata

HIGH_CONFIDENCE_REASONS = {"guest", "priority_keyword", "title_keyword"}


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "cache"


def _load_config(config_path: Path) -> Config:
    with open(config_path) as f:
        return Config.model_validate(yaml.safe_load(f))


def main():
    parser = argparse.ArgumentParser(
        description="Generate silver labels from high-confidence signals"
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent.parent / "configs" / "channels.yaml"
    )
    parser.add_argument(
        "--min-duration", type=int, default=0, help="Skip episodes shorter than N seconds"
    )
    parser.add_argument("--show-stats", action="store_true", help="Print per-category counts")
    args = parser.parse_args()

    cache_dir = args.cache_dir or _default_cache_dir()
    if not cache_dir.exists():
        print(f"Cache not found: {cache_dir}")
        return 1

    config = _load_config(args.config)
    index = cache_mod.read_index(cache_dir)
    if not index:
        print("Index is empty.")
        return 1

    labeled = []
    unlabeled = []
    skipped_duration = 0

    # Build lookup: channel_id -> channel_config for episode routing
    channel_by_id: dict[str, object] = {
        c.channel_id: c for c in config.channels if c.channel_id
    }

    for channel_config in config.channels:
        classifier = VideoClassifier(channel_config, use_transcripts=False)

        # Only process episodes that belong to this channel
        channel_entries = [
            e for e in index
            if not channel_by_id  # single-channel config: process all
            or cache_mod.read_metadata(cache_dir, e["video_id"]) is not None
            and cache_mod.read_metadata(cache_dir, e["video_id"]).get("channel_id")
            == channel_config.channel_id
        ]

        total_entries = len(channel_entries)
        for n, entry in enumerate(channel_entries, 1):
            if n % 250 == 0 or n == total_entries:
                print(f"  [{n}/{total_entries}] classifying...", flush=True)
            meta = cache_mod.read_metadata(cache_dir, entry["video_id"])
            if meta is None:
                continue

            duration = meta.get("duration") or 0
            if args.min_duration and duration < args.min_duration:
                skipped_duration += 1
                continue

            video = VideoMetadata(
                video_id=meta["video_id"],
                title=meta.get("title", ""),
                description=meta.get("description", ""),
                channel_name=meta.get("channel_name", ""),
                channel_id=meta.get("channel_id", ""),
                upload_date=meta.get("upload_date"),
                duration=duration,
                view_count=meta.get("view_count"),
                tags=meta.get("tags", []),
            )

            result = classifier.classify(video)

            record = {
                "video_id": video.video_id,
                "title": video.title,
                "upload_date": video.upload_date,
                "duration": duration,
                "channel_url": channel_config.url,
                "channel_id": channel_config.channel_id,
                "category": result.category_key,
                "match_reason": result.match_reason,
                "matched_value": result.matched_value,
                "skipped": result.skipped,
            }

            if result.match_reason in HIGH_CONFIDENCE_REASONS:
                labeled.append(record)
            else:
                unlabeled.append(record)

    out_dir = cache_dir
    (out_dir / "labeled.json").write_text(json.dumps(labeled, indent=2, ensure_ascii=False))
    (out_dir / "unlabeled.json").write_text(json.dumps(unlabeled, indent=2, ensure_ascii=False))

    print(f"Cache:            {cache_dir}")
    print(f"Total episodes:   {len(index)}")
    print(f"Skipped (dur):    {skipped_duration}")
    print(
        f"Labeled:          {len(labeled)} ({len(labeled) * 100 // (len(labeled) + len(unlabeled))}%)"  # noqa: E501
    )
    print(f"Unlabeled:        {len(unlabeled)}")
    print()

    if True:  # always show stats
        from collections import Counter

        reasons = Counter(r["match_reason"] for r in labeled)
        cats = Counter(r["category"] for r in labeled)
        print("Match reasons (labeled):")
        for reason, count in reasons.most_common():
            print(f"  {reason:<25} {count}")
        print()
        print("Category distribution (labeled):")
        for cat, count in cats.most_common():
            print(f"  {cat:<30} {count}")
        print()
        print("Unlabeled match reasons:")
        u_reasons = Counter(r["match_reason"] for r in unlabeled)
        for reason, count in u_reasons.most_common():
            print(f"  {reason:<25} {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
