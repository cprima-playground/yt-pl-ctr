#!/usr/bin/env python3
"""Classify all cached episodes and sync them to playlists.

Two-phase design:
  Phase 1 — classify:  reads only from local cache, produces backfill_plan.json
  Phase 2 — execute:   reads backfill_plan.json, writes to YouTube API

This separation lets you inspect (and edit) the plan before touching any playlist.

Usage:
    uv run python scripts/backfill.py                      # classify → print plan
    uv run python scripts/backfill.py --save-plan          # classify → save plan file
    uv run python scripts/backfill.py --execute            # classify + execute in one go
    uv run python scripts/backfill.py --execute-plan       # execute saved plan only
    uv run python scripts/backfill.py --save-plan --limit 500
    uv run python scripts/backfill.py --save-plan --video-id MtoPEub7XwA
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cache as cache_mod
from yt_pl_ctr.classifier import VideoClassifier
from yt_pl_ctr.models import Config, VideoMetadata
from yt_pl_ctr.youtube import YouTubeAPIError, YouTubeClient

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

_PLAN_FILENAME = "backfill_plan.json"


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "cache"


def _load_config(config_path: Path) -> Config:
    with open(config_path) as f:
        return Config.model_validate(yaml.safe_load(f))


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class PlannedAction:
    """One playlist change decided by the classifier."""
    action: str           # "add" | "move" | "remove"
    video_id: str
    title: str
    target_pl_name: str | None    # None for remove
    target_category_key: str | None
    source_pl_name: str | None    # None for add
    source_pl_id: str | None      # None for add (needed by remove/move)
    source_category_key: str | None
    match_reason: str
    matched_value: str | None


@dataclass
class BackfillStats:
    added: int = 0
    moved: int = 0
    already_correct: int = 0
    skipped_category: int = 0
    skipped_no_meta: int = 0
    errors: int = 0
    actions: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.actions.append(msg)
        print(f"  {msg}", flush=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_placement_index(
    membership: dict, playlists: dict, classifier: VideoClassifier, channel_config
) -> dict[str, dict]:
    """Map video_id → {category_key, playlist_id} using the membership cache."""
    title_to_slug: dict[str, str] = {}
    for slug in channel_config.playlists:
        pl_name = classifier.get_playlist_name(slug)
        title_to_slug[pl_name] = slug

    placement: dict[str, dict] = {}
    for vid, pids in membership.items():
        for pid in pids:
            title = playlists.get(pid, "")
            slug = title_to_slug.get(title)
            if slug:
                placement[vid] = {"category_key": slug, "playlist_id": pid}
                break
    return placement


def _ensure_and_add(
    youtube: YouTubeClient,
    playlist_ids: dict[str, str],
    pl_name: str,
    video_id: str,
    config: Config,
) -> None:
    if pl_name not in playlist_ids:
        playlist_ids[pl_name] = youtube.ensure_playlist(
            pl_name,
            description=config.playlist_settings.description_template,
            privacy=config.playlist_settings.privacy,
        )
    youtube.add_video_if_missing(playlist_ids[pl_name], video_id)


# ── Phase 1: classify ──────────────────────────────────────────────────────────

def classify(
    cache_dir: Path,
    config_path: Path,
    limit: int | None = None,
    video_ids: list[str] | None = None,
    channel_name: str | None = None,
) -> tuple[list[PlannedAction], dict]:
    """Classify all episodes and return (planned_actions, context).

    No YouTube API calls. Reads only from local cache + Wikipedia lookups.
    Returns the action plan and the membership/playlists context needed by execute().
    """
    config = _load_config(config_path)
    if channel_name:
        matches = [c for c in config.channels if c.name.lower() == channel_name.lower()]
        channel_config = matches[0] if matches else config.channels[0]
    else:
        channel_config = config.channels[0]

    index = cache_mod.read_index(cache_dir)
    if not index:
        print("Cache index is empty — run fetch_transcripts.py first.")
        return [], {}

    membership_data = cache_mod.read_playlist_membership(cache_dir)
    if membership_data is None:
        print("Fetching playlist membership from YouTube API...")
        youtube = YouTubeClient.from_env()
        data = youtube.load_all_membership()
        cache_mod.write_playlist_membership(cache_dir, data["playlists"], data["membership"])
        membership_data = cache_mod.read_playlist_membership(cache_dir, max_age_hours=9999)

    membership: dict[str, list[str]] = membership_data["membership"]
    playlists: dict[str, str] = membership_data["playlists"]

    classifier = VideoClassifier(channel_config, use_transcripts=True)
    placement = _build_placement_index(membership, playlists, classifier, channel_config)

    if video_ids:
        entries = [e for e in index if e["video_id"] in video_ids]
    elif channel_config.channel_id:
        entries = []
        for e in index:
            meta = cache_mod.read_metadata(cache_dir, e["video_id"])
            if meta and meta.get("channel_id") == channel_config.channel_id:
                entries.append(e)
    else:
        entries = list(index)

    if channel_config.min_duration:
        before = len(entries)
        entries = [e for e in entries if (e.get("duration") or 0) >= channel_config.min_duration]
        print(f"Duration filter: kept {len(entries)}/{before} (>= {channel_config.min_duration}s)")

    cutoff = channel_config.min_upload_date_str()
    if cutoff:
        before = len(entries)
        entries = [e for e in entries if (e.get("upload_date") or "") >= cutoff]
        print(f"Age filter: kept {len(entries)}/{before} (>= {cutoff})")

    if limit is not None:
        entries = entries[:limit]

    total = len(entries)
    print(f"Classify: {total} episodes")
    print(f"Current placement: {len(placement)} videos across {len(playlists)} playlists")
    print()

    planned: list[PlannedAction] = []
    skipped_no_meta = 0
    already_correct = 0
    skipped_category = 0
    errors = 0

    for i, entry in enumerate(entries, 1):
        if i % 100 == 0:
            print(
                f"  [{i}/{total}] planned={len(planned)} correct={already_correct} "
                f"skip={skipped_category}",
                flush=True,
            )

        vid = entry["video_id"]
        meta_dict = cache_mod.read_metadata(cache_dir, vid)
        if meta_dict is None:
            skipped_no_meta += 1
            continue

        try:
            video = VideoMetadata(**meta_dict)
        except Exception as e:
            logger.warning("Could not load VideoMetadata for %s: %s", vid, e)
            errors += 1
            continue

        try:
            result = classifier.classify(video)
        except Exception as e:
            logger.warning("Classification failed for %s: %s", vid, e)
            errors += 1
            continue

        prev = placement.get(vid)

        if result.skipped:
            if prev:
                # If it was in a playlist but now skipped, it needs removal
                prev_in_playlist = prev["category_key"] in channel_config.playlists
                if not prev_in_playlist:
                    already_correct += 1
                else:
                    planned.append(PlannedAction(
                        action="remove",
                        video_id=vid,
                        title=video.title,
                        target_pl_name=None,
                        target_category_key=None,
                        source_pl_name=classifier.get_playlist_name(prev["category_key"]),
                        source_pl_id=prev["playlist_id"],
                        source_category_key=prev["category_key"],
                        match_reason="default/skip",
                        matched_value=None,
                    ))
            else:
                skipped_category += 1
            continue

        target_pl_name = classifier.get_playlist_name(result.category_key)

        if prev and prev["category_key"] == result.category_key:
            already_correct += 1
            continue

        if prev and prev["category_key"] != result.category_key:
            planned.append(PlannedAction(
                action="move",
                video_id=vid,
                title=video.title,
                target_pl_name=target_pl_name,
                target_category_key=result.category_key,
                source_pl_name=classifier.get_playlist_name(prev["category_key"]),
                source_pl_id=prev["playlist_id"],
                source_category_key=prev["category_key"],
                match_reason=result.match_reason,
                matched_value=result.matched_value,
            ))
        else:
            planned.append(PlannedAction(
                action="add",
                video_id=vid,
                title=video.title,
                target_pl_name=target_pl_name,
                target_category_key=result.category_key,
                source_pl_name=None,
                source_pl_id=None,
                source_category_key=None,
                match_reason=result.match_reason,
                matched_value=result.matched_value,
            ))

    print()
    adds = sum(1 for a in planned if a.action == "add")
    moves = sum(1 for a in planned if a.action == "move")
    removes = sum(1 for a in planned if a.action == "remove")
    print(f"Plan: {adds} add, {moves} move, {removes} remove | "
          f"correct={already_correct} skip={skipped_category} "
          f"no_meta={skipped_no_meta} errors={errors}")

    ctx = {"membership": membership, "playlists": playlists}
    return planned, ctx


# ── Phase 2: execute ───────────────────────────────────────────────────────────

def execute(
    planned: list[PlannedAction],
    cache_dir: Path,
    config_path: Path,
    ctx: dict,
) -> BackfillStats:
    """Execute a pre-classified plan against the YouTube API."""
    config = _load_config(config_path)
    stats = BackfillStats()

    membership: dict[str, list[str]] = ctx["membership"]
    playlists: dict[str, str] = ctx["playlists"]

    youtube = YouTubeClient.from_env()
    playlist_ids: dict[str, str] = {title: pid for pid, title in playlists.items()}

    # Pre-populate membership cache to skip redundant playlist_contains_video calls
    for vid, pids in membership.items():
        for pid in pids:
            youtube._added_videos.add((pid, vid))

    total = len(planned)
    print(f"Execute: {total} planned actions")
    print()

    for i, action in enumerate(planned, 1):
        vid = action.video_id
        title_short = action.title[:60]

        if action.action == "remove":
            stats.log(
                f"[REMOVE] #{i}/{total} '{title_short}' removed from "
                f"{action.source_pl_name} (now classifies as default/skip)"
            )
            try:
                old_contents = youtube.get_playlist_contents(action.source_pl_id)
                item_id = old_contents.get(vid)
                if item_id:
                    youtube.remove_playlist_item(item_id)
                stats.moved += 1
            except Exception as e:
                logger.error("Remove failed for %s: %s", vid, e)
                stats.errors += 1

        elif action.action == "move":
            stats.log(
                f"[MOVE] #{i}/{total} '{title_short}' "
                f"{action.source_pl_name} → {action.target_pl_name} "
                f"({action.match_reason}: {action.matched_value or ''})"
            )
            try:
                old_contents = youtube.get_playlist_contents(action.source_pl_id)
                item_id = old_contents.get(vid)
                if item_id:
                    youtube.remove_playlist_item(item_id)
                _ensure_and_add(youtube, playlist_ids, action.target_pl_name, vid, config)
                stats.moved += 1
            except Exception as e:
                logger.error("Move failed for %s: %s", vid, e)
                stats.errors += 1

        elif action.action == "add":
            stats.log(
                f"[ADD] #{i}/{total} '{title_short}' → {action.target_pl_name} "
                f"({action.match_reason}: {action.matched_value or ''})"
            )
            try:
                _ensure_and_add(youtube, playlist_ids, action.target_pl_name, vid, config)
                stats.added += 1
            except Exception as e:
                logger.error("Add failed for %s: %s", vid, e)
                stats.errors += 1

    return stats


# ── Plan file I/O ──────────────────────────────────────────────────────────────

def save_plan(planned: list[PlannedAction], cache_dir: Path) -> Path:
    plan_file = cache_dir / _PLAN_FILENAME
    data = {
        "count": len(planned),
        "actions": [asdict(a) for a in planned],
    }
    plan_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return plan_file


def load_plan(cache_dir: Path) -> list[PlannedAction]:
    plan_file = cache_dir / _PLAN_FILENAME
    if not plan_file.exists():
        print(f"No plan file found: {plan_file}", file=sys.stderr)
        print("Run with --save-plan first.", file=sys.stderr)
        sys.exit(1)
    data = json.loads(plan_file.read_text())
    return [PlannedAction(**a) for a in data["actions"]]


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify episodes and manage playlists (two-phase)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  (default)         classify → print plan (no writes)
  --save-plan       classify → save backfill_plan.json (no writes)
  --execute         classify → execute immediately
  --execute-plan    load saved plan → execute (skip re-classification)
""",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent.parent / "configs" / "channels.yaml",
    )
    parser.add_argument("--channel", default=None,
                        help="Channel name to process (default: first channel in config)")
    parser.add_argument("--save-plan", action="store_true",
                        help="Classify and save plan to backfill_plan.json (no YouTube writes)")
    parser.add_argument("--execute", action="store_true",
                        help="Classify and execute in one step")
    parser.add_argument("--execute-plan", action="store_true",
                        help="Execute the saved backfill_plan.json without re-classifying")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N episodes (classify phase only)")
    parser.add_argument("--video-id", action="append", dest="video_ids",
                        help="Process specific episode(s) only")
    args = parser.parse_args()

    cache_dir = args.cache_dir or _default_cache_dir()

    try:
        if args.execute_plan:
            planned = load_plan(cache_dir)
            # Need membership context for execute; re-read from cache
            membership_data = cache_mod.read_playlist_membership(cache_dir, max_age_hours=9999)
            if membership_data is None:
                print("No membership cache — run: just refresh-membership", file=sys.stderr)
                return 1
            ctx = {
                "membership": membership_data["membership"],
                "playlists": membership_data["playlists"],
            }
            print(f"Loaded plan: {len(planned)} actions from {cache_dir / _PLAN_FILENAME}")
            print()
            stats = execute(planned, cache_dir, args.config, ctx)

        elif args.execute:
            planned, ctx = classify(cache_dir, args.config, args.limit, args.video_ids,
                                    channel_name=args.channel)
            if not planned:
                print("Nothing to do.")
                return 0
            stats = execute(planned, cache_dir, args.config, ctx)

        elif args.save_plan:
            planned, _ = classify(cache_dir, args.config, args.limit, args.video_ids,
                                  channel_name=args.channel)
            plan_file = save_plan(planned, cache_dir)
            print(f"Plan saved: {plan_file}  ({len(planned)} actions)")
            print("Review the plan, then run: backfill.py --execute-plan")
            return 0

        else:
            # Default: classify and print plan, no writes
            planned, _ = classify(cache_dir, args.config, args.limit, args.video_ids,
                                  channel_name=args.channel)
            return 0

    except YouTubeAPIError as e:
        print(f"YouTube API error: {e}", file=sys.stderr)
        return 1

    print()
    print(f"{'─' * 50}")
    print(f"Added   : {stats.added}")
    print(f"Moved   : {stats.moved}")
    print(f"Errors  : {stats.errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
