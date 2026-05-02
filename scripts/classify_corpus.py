#!/usr/bin/env python3
"""Run classifier on the structured episode cache and produce a reviewable report.

Reads from the cache directory (index.json + episodes/*/metadata.json).
Uses cached transcript.txt files when available — no live yt-dlp calls needed.

Outputs (written next to index.json):
  classified.json   — machine-readable results for all episodes
  to_label.json     — template for manual labeling (uncertain predictions only)

Usage:
    uv run python scripts/classify_corpus.py
    uv run python scripts/classify_corpus.py --no-transcripts
    uv run python scripts/classify_corpus.py --category ancient_history
    uv run python scripts/classify_corpus.py --reason default
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cache as cache_mod
from yt_pl_ctr.classifier import VideoClassifier
from yt_pl_ctr.config import load_config
from yt_pl_ctr.models import VideoMetadata

CAT_COLORS = {
    "science_tech": "\033[36m",
    "ufo_aliens": "\033[35m",
    "paranormal": "\033[95m",
    "politics": "\033[31m",
    "comedy": "\033[33m",
    "mma_martial_arts": "\033[32m",
    "ancient_history": "\033[34m",
    "other": "\033[90m",
}
RESET = "\033[0m"


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "cache"


def main():
    parser = argparse.ArgumentParser(description="Classify episode cache and produce review report")
    parser.add_argument("--config", default="configs/channels.yaml", help="Config file")
    default_cache = _default_cache_dir()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache,
        help=f"Cache directory (default: {default_cache})",
    )
    parser.add_argument(
        "--no-transcripts",
        action="store_true",
        help="Skip transcript files (cached transcripts still read from disk)",
    )
    parser.add_argument(
        "--fetch-missing-transcripts",
        action="store_true",
        help="Fetch missing transcripts live via yt-dlp (residential IPs only)",
    )
    parser.add_argument("--category", help="Filter output to this category")
    parser.add_argument("--reason", help="Filter output to this match reason")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if not (cache_dir / "index.json").exists():
        print(f"Cache not found: {cache_dir}")
        print("Run: uv run python scripts/download_test_data.py")
        return 1

    config = load_config(args.config)
    channel_config = config.channels[0]

    # use_transcripts here means "fetch live via yt-dlp if no cached file"
    # We handle transcript loading from disk; live fetching only if --fetch-missing-transcripts
    classifier = VideoClassifier(channel_config, use_transcripts=False)

    index = cache_mod.read_index(cache_dir)
    print(f"Classifying {len(index)} episodes from {cache_dir}")
    live = "on" if args.fetch_missing_transcripts else "off"
    print(f"  cached transcripts=on, live fetch={live}")
    print()

    results = []
    category_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}

    for idx_entry in index:
        vid_id = idx_entry["video_id"]
        raw = cache_mod.read_metadata(cache_dir, vid_id)
        if not raw:
            continue

        video = VideoMetadata(**{k: v for k, v in raw.items() if k in VideoMetadata.model_fields})

        # Inject cached transcript into video description supplement
        # The classifier checks description; we pass transcript as extra context
        # by temporarily classifying with transcript text if available
        transcript = cache_mod.read_transcript(cache_dir, vid_id)

        if transcript is None and args.fetch_missing_transcripts and not args.no_transcripts:
            from yt_pl_ctr.fetcher import fetch_transcript

            transcript = fetch_transcript(vid_id)
            if transcript:
                cache_mod.write_transcript(cache_dir, vid_id, transcript)
                idx_entry["has_transcript"] = True

        # Run classifier; then re-run with transcript if still unclassified
        result = classifier.classify(video)

        if result.match_reason == "default" and transcript and not args.no_transcripts:
            # Inject transcript into a temporary VideoMetadata for a second pass
            video_with_transcript = video.model_copy(
                update={"description": (video.description or "") + "\n\n" + transcript}
            )
            result2 = classifier.classify(video_with_transcript)
            if result2.match_reason != "default":
                result = result2
                result.match_reason = "transcript_" + result.match_reason

        guest = classifier.extract_guest(video.title) or ""
        entry = {
            "video_id": vid_id,
            "title": video.title,
            "guest": guest,
            "duration_min": round((video.duration or 0) / 60),
            "predicted_category": result.category_key,
            "match_reason": result.match_reason,
            "matched_value": result.matched_value or "",
            "skipped": result.skipped,
            "has_transcript": transcript is not None,
        }
        results.append(entry)
        category_counts[result.category_key] = category_counts.get(result.category_key, 0) + 1
        reason_counts[result.match_reason] = reason_counts.get(result.match_reason, 0) + 1

    # Filters
    filtered = results
    if args.category:
        filtered = [r for r in filtered if r["predicted_category"] == args.category]
    if args.reason:
        filtered = [r for r in filtered if r["match_reason"].startswith(args.reason)]

    # Table
    print(f"{'#':<4} {'Title':<55} {'Guest':<22} {'Category':<18} {'T'} {'Reason':<22} {'Match'}")
    print("-" * 145)
    for i, r in enumerate(filtered, 1):
        cat = r["predicted_category"]
        color = CAT_COLORS.get(cat, "")
        skip = " [SKIP]" if r["skipped"] else ""
        t_flag = "T" if r["has_transcript"] else " "
        print(
            f"{i:<4} "
            f"{r['title'][:54]:<55} "
            f"{r['guest'][:21]:<22} "
            f"{color}{cat:<18}{RESET} "
            f"{t_flag} "
            f"{r['match_reason']:<22} "
            f"{str(r['matched_value'])[:30]}"
            f"{skip}"
        )

    print()
    print("=== Category distribution ===")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"  {cat:<22} {count:>4}  {pct:>5.1f}%  {bar}")

    print()
    print("=== Match reason distribution ===")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        print(f"  {reason:<28} {count:>4}  {pct:>5.1f}%")

    # Save outputs
    out_classified = cache_dir / "classified.json"
    out_label = cache_dir / "to_label.json"

    with open(out_classified, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_classified}")

    needs_review = [
        r for r in results if r["match_reason"] in ("no_model", "low_confidence", "unknown_category") or r["skipped"]
    ]
    labeling_template = [
        {
            "video_id": r["video_id"],
            "title": r["title"],
            "guest": r["guest"],
            "predicted_category": r["predicted_category"],
            "match_reason": r["match_reason"],
            "matched_value": r["matched_value"],
            "correct_category": r["predicted_category"],
            "notes": "",
        }
        for r in needs_review
    ]
    with open(out_label, "w") as f:
        json.dump(labeling_template, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(labeling_template)} entries needing review → {out_label}")

    # Update index with has_transcript flags
    cache_mod.write_index(cache_dir, cache_mod.read_index(cache_dir))

    return 0


if __name__ == "__main__":
    sys.exit(main())
