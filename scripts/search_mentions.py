#!/usr/bin/env python3
"""Find episodes that mention specific entities and add them to keyword playlists.

Uses the channel's `keyword_playlists` config block — no ML model required.
A video matches a playlist if any of its keywords appear (case-insensitive) in
the title, description, or transcript.

Usage:
    # Preview matches for the Candace Owens channel (no API writes)
    uv run python scripts/search_mentions.py --channel "Candace Owens"

    # Save a plan file for review
    uv run python scripts/search_mentions.py --channel "Candace Owens" --save-plan

    # Execute the saved plan against YouTube API
    uv run python scripts/search_mentions.py --channel "Candace Owens" --execute-plan

    # Classify and execute in one step
    uv run python scripts/search_mentions.py --channel "Candace Owens" --execute
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cache as cache_mod
from yt_pl_ctr.models import Config, ChannelConfig, KeywordPlaylistConfig
from yt_pl_ctr.youtube import YouTubeAPIError, YouTubeClient

_PLAN_FILENAME = "keyword_plan.json"


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "cache"


def _load_config(config_path: Path) -> Config:
    with open(config_path) as f:
        return Config.model_validate(yaml.safe_load(f))


def _select_channel(config: Config, channel_name: str | None) -> ChannelConfig:
    if channel_name:
        matches = [c for c in config.channels if c.name.lower() == channel_name.lower()]
        if not matches:
            names = [c.name for c in config.channels]
            print(f"Channel {channel_name!r} not found. Available: {names}", file=sys.stderr)
            sys.exit(1)
        return matches[0]
    return config.channels[0]


# ── Matching ───────────────────────────────────────────────────────────────────

def _load_transcript_text(cache_dir: Path, video_id: str) -> str:
    """Return transcript text for a video, or empty string if not cached."""
    transcript_path = cache_dir / "episodes" / video_id / "transcript.txt"
    if transcript_path.exists():
        return transcript_path.read_text(encoding="utf-8", errors="replace")
    return ""


def _matches(text: str, keywords: list[str]) -> str | None:
    """Return the first matching keyword (case-insensitive), or None."""
    lower = text.lower()
    for kw in keywords:
        if kw.lower() in lower:
            return kw
    return None


@dataclass
class MentionMatch:
    video_id: str
    title: str
    slug: str
    playlist_title: str
    matched_keyword: str
    matched_in: str   # "title" | "description" | "transcript"


# ── Phase 1: scan ──────────────────────────────────────────────────────────────

def scan(
    cache_dir: Path,
    config_path: Path,
    channel_name: str | None = None,
    limit: int | None = None,
) -> list[MentionMatch]:
    """Scan cached episodes for keyword matches. No API calls."""
    config = _load_config(config_path)
    channel_config = _select_channel(config, channel_name)

    if not channel_config.keyword_playlists:
        print(f"No keyword_playlists configured for {channel_config.name!r}.", file=sys.stderr)
        return []

    index = cache_mod.read_index(cache_dir)
    if not index:
        print("Cache index is empty — run download_test_data.py first.")
        return []

    # Filter to this channel
    if channel_config.channel_id:
        entries = []
        for e in index:
            meta = cache_mod.read_metadata(cache_dir, e["video_id"])
            if meta and meta.get("channel_id") == channel_config.channel_id:
                entries.append(e)
    else:
        entries = list(index)

    if channel_config.min_duration:
        entries = [e for e in entries if (e.get("duration") or 0) >= channel_config.min_duration]

    cutoff = channel_config.min_upload_date_str()
    if cutoff:
        entries = [e for e in entries if (e.get("upload_date") or "") >= cutoff]

    if limit is not None:
        entries = entries[:limit]

    total = len(entries)
    print(f"Scanning {total} episodes from {channel_config.name!r}")
    print(f"Keyword playlists: {list(channel_config.keyword_playlists)}")
    print()

    matches: list[MentionMatch] = []

    for i, entry in enumerate(entries, 1):
        if i % 200 == 0:
            print(f"  [{i}/{total}] matches so far: {len(matches)}", flush=True)

        vid = entry["video_id"]
        meta = cache_mod.read_metadata(cache_dir, vid)
        if meta is None:
            continue

        title = meta.get("title", "")
        description = meta.get("description", "")
        transcript = _load_transcript_text(cache_dir, vid)

        for slug, kpl in channel_config.keyword_playlists.items():
            # Check each text field in priority order; report where it first matched
            for text, source in [
                (title, "title"),
                (description, "description"),
                (transcript, "transcript"),
            ]:
                kw = _matches(text, kpl.keywords)
                if kw:
                    matches.append(MentionMatch(
                        video_id=vid,
                        title=title,
                        slug=slug,
                        playlist_title=kpl.title,
                        matched_keyword=kw,
                        matched_in=source,
                    ))
                    break  # one match per playlist per video is enough

    print()
    by_slug: dict[str, int] = {}
    for m in matches:
        by_slug[m.slug] = by_slug.get(m.slug, 0) + 1

    for slug, count in sorted(by_slug.items()):
        kpl = channel_config.keyword_playlists[slug]
        print(f"  {slug}: {count} episodes → \"{kpl.title}\"")
    print(f"\nTotal matches: {len(matches)} (a video can match multiple playlists)")

    return matches


# ── Phase 2: execute ───────────────────────────────────────────────────────────

def execute(matches: list[MentionMatch], config_path: Path, cache_dir: Path) -> None:
    """Add matched videos to their playlists via the YouTube API."""
    config = _load_config(config_path)
    youtube = YouTubeClient.from_env()
    playlist_ids: dict[str, str] = {}

    print(f"\nExecute: {len(matches)} playlist additions")
    added = 0
    errors = 0

    for m in matches:
        pl_title = m.playlist_title
        if pl_title not in playlist_ids:
            playlist_ids[pl_title] = youtube.ensure_playlist(
                pl_title,
                description=config.playlist_settings.description_template,
                privacy=config.playlist_settings.privacy,
            )

        pl_id = playlist_ids[pl_title]
        try:
            youtube.add_video_if_missing(pl_id, m.video_id)
            added += 1
            print(
                f"  [ADD] {m.video_id} → \"{pl_title}\" "
                f"(matched {m.matched_keyword!r} in {m.matched_in})",
                flush=True,
            )
        except YouTubeAPIError as e:
            print(f"  [ERR] {m.video_id}: {e}", file=sys.stderr)
            errors += 1

    print(f"\nDone. Added: {added}, Errors: {errors}")


# ── Plan file I/O ──────────────────────────────────────────────────────────────

def save_plan(matches: list[MentionMatch], cache_dir: Path) -> Path:
    plan_file = cache_dir / _PLAN_FILENAME
    data = {
        "count": len(matches),
        "matches": [asdict(m) for m in matches],
    }
    plan_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return plan_file


def load_plan(cache_dir: Path) -> list[MentionMatch]:
    plan_file = cache_dir / _PLAN_FILENAME
    if not plan_file.exists():
        print(f"No plan file found: {plan_file}", file=sys.stderr)
        print("Run with --save-plan first.", file=sys.stderr)
        sys.exit(1)
    data = json.loads(plan_file.read_text())
    return [MentionMatch(**m) for m in data["matches"]]


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find and playlist episodes by keyword/entity mention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  (default)         scan → print matches (no writes)
  --save-plan       scan → save keyword_plan.json (no writes)
  --execute         scan → execute immediately
  --execute-plan    load saved plan → execute (skip re-scan)
""",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent.parent / "configs" / "channels.yaml",
    )
    parser.add_argument("--channel", default=None,
                        help="Channel name (default: first in config)")
    parser.add_argument("--save-plan", action="store_true",
                        help="Scan and save keyword_plan.json (no YouTube writes)")
    parser.add_argument("--execute", action="store_true",
                        help="Scan and execute in one step")
    parser.add_argument("--execute-plan", action="store_true",
                        help="Execute saved keyword_plan.json without re-scanning")
    parser.add_argument("--limit", type=int, default=None,
                        help="Scan at most N episodes")
    args = parser.parse_args()

    cache_dir = args.cache_dir or _default_cache_dir()

    if args.execute_plan:
        matches = load_plan(cache_dir)
        print(f"Loaded plan: {len(matches)} matches from {cache_dir / _PLAN_FILENAME}")
        execute(matches, args.config, cache_dir)

    elif args.execute:
        matches = scan(cache_dir, args.config, args.channel, args.limit)
        if not matches:
            print("No matches — nothing to do.")
            return 0
        execute(matches, args.config, cache_dir)

    elif args.save_plan:
        matches = scan(cache_dir, args.config, args.channel, args.limit)
        plan_file = save_plan(matches, cache_dir)
        print(f"\nPlan saved: {plan_file}  ({len(matches)} matches)")
        print("Review the plan, then run: search_mentions.py --execute-plan")

    else:
        scan(cache_dir, args.config, args.channel, args.limit)

    return 0


if __name__ == "__main__":
    sys.exit(main())
