#!/usr/bin/env python3
"""Idempotent end-to-end pipeline: ingest → features → membership → backfill → sync.

Each phase checks its own completion state before running. Re-running is safe.

Phases (in order):
  ingest      fetch_transcripts + enrich_metadata + fetch_wikipedia
  features    build_features (NLP token extraction)
  membership  fetch_playlist_membership (12 h cache; --force-membership to refresh)
  backfill    classify cached episodes and optionally write to YouTube
  sync        fetch new episodes and sync to playlists

Note: LLM labeling (llm_label.py) and model training (train_classifier.py) are
run manually as one-off preparation steps, not wired into this pipeline.

Cache invalidation:
  --invalidate-membership    Delete playlist_membership.json before membership phase
  --force-membership         Force re-fetch even if cache is fresh

Usage:
    uv run python scripts/pipeline.py
    uv run python scripts/pipeline.py --only labels
    uv run python scripts/pipeline.py --skip llm
    uv run python scripts/pipeline.py --llm-limit 50
    uv run python scripts/pipeline.py --invalidate-membership
    uv run python scripts/pipeline.py --force-membership
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SCRIPTS = Path(__file__).parent
PHASES = ["ingest", "features", "membership", "backfill", "sync"]


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else SCRIPTS.parent / "cache"


def _default_config() -> Path:
    return SCRIPTS.parent / "configs" / "channels.yaml"


def _run(cmd: list, label: str) -> int:
    import shlex
    print(f"  $ {' '.join(shlex.quote(str(c)) for c in cmd)}", flush=True)
    t0 = time.monotonic()
    result = subprocess.run(cmd)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        print(f"  [FAIL] {label} exited {result.returncode} ({elapsed:.1f}s)", file=sys.stderr)
    else:
        print(f"  [OK]   {label} ({elapsed:.1f}s)", flush=True)
    return result.returncode


def _uv(*args) -> list:
    return ["uv", "run", "python", *args]


# ── Phase status checks ───────────────────────────────────────────────────────


def _status_ingest(cache_dir: Path) -> str:
    index_f = cache_dir / "index.json"
    if not index_f.exists():
        return "index.json missing — not started"
    index = json.loads(index_f.read_text(encoding="utf-8"))
    total = len(index)
    with_transcript = sum(1 for e in index if e.get("has_transcript"))
    return f"{total} episodes | transcripts: {with_transcript}/{total}"


def _status_features(cache_dir: Path) -> str:
    index_f = cache_dir / "index.json"
    if not index_f.exists():
        return "no index.json"
    index = json.loads(index_f.read_text(encoding="utf-8"))
    with_transcript = sum(1 for e in index if e.get("has_transcript"))
    # sample first entry to check if features exist rather than scanning all
    sample = next(
        (e for e in index if e.get("has_transcript")), None
    )
    if sample and (cache_dir / "episodes" / sample["video_id"] / "features.json").exists():
        return f"built (transcripts: {with_transcript})"
    return f"not built | {with_transcript} transcripts ready"



def _status_membership(cache_dir: Path) -> str:
    import datetime
    f = cache_dir / "playlist_membership.json"
    if not f.exists():
        return "not cached"
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        fetched_at = datetime.datetime.fromisoformat(data["fetched_at"])
        age = datetime.datetime.now(datetime.UTC) - fetched_at
        hours = age.total_seconds() / 3600
        n_pl = len(data.get("playlists", {}))
        n_vid = len(data.get("membership", {}))
        fresh = "fresh" if hours < 12 else "STALE"
        return f"{n_pl} playlists, {n_vid} videos tracked — {fresh} ({hours:.1f}h old)"
    except Exception:
        return "corrupt cache"


def _status_sync(dry_run: bool) -> str:
    return "dry-run (pass --execute to write to YouTube)" if dry_run else "LIVE — writes to YouTube"



# ── Phase runners ─────────────────────────────────────────────────────────────


def run_ingest(cache_dir: Path, config: Path, channel: str | None = None) -> int:
    transcript_cmd = _uv(SCRIPTS / "fetch_transcripts.py", "--cache-dir", cache_dir,
                         "--config", config)
    if channel:
        transcript_cmd += ["--channel", channel]
    rc = _run(transcript_cmd, "fetch_transcripts")
    if rc != 0:
        return rc
    rc = _run(
        _uv(SCRIPTS / "enrich_metadata.py", "--cache-dir", cache_dir),
        "enrich_metadata",
    )
    if rc != 0:
        return rc
    return _run(
        _uv(SCRIPTS / "fetch_wikipedia.py", "--cache-dir", cache_dir),
        "fetch_wikipedia",
    )


def run_features(cache_dir: Path) -> int:
    return _run(
        _uv(SCRIPTS / "build_features.py", "--cache-dir", cache_dir),
        "build_features",
    )



def run_membership(cache_dir: Path, force: bool = False, invalidate: bool = False) -> int:
    cmd = _uv(SCRIPTS / "fetch_playlist_membership.py", "--cache-dir", cache_dir)
    if invalidate:
        inv_cmd = _uv(SCRIPTS / "fetch_playlist_membership.py", "--cache-dir", cache_dir,
                      "--invalidate")
        _run(inv_cmd, "invalidate membership")
    if force or invalidate:
        cmd.append("--force")
    return _run(cmd, "fetch_playlist_membership")



def run_backfill(cache_dir: Path, config: Path, dry_run: bool) -> int:
    cmd = _uv(SCRIPTS / "backfill.py", "--cache-dir", cache_dir, "--config", config)
    if not dry_run:
        cmd.append("--execute")
    return _run(cmd, "backfill" + (" [dry-run]" if dry_run else " [LIVE]"))


def run_sync(config: Path, limit: int | None, dry_run: bool) -> int:
    cmd = ["uv", "run", "yt-pl-ctr", "sync", "--config", str(config)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if dry_run:
        cmd.append("--dry-run")
    return _run(cmd, "sync" + (" [dry-run]" if dry_run else " [LIVE]"))


def _print_status(cache_dir: Path, dry_run: bool = True) -> None:
    print("Pipeline status")
    print("=" * 60)
    print(f"  cache : {cache_dir}")
    print(f"  config: {_default_config()}")
    print()
    print(f"  ingest     : {_status_ingest(cache_dir)}")
    print(f"  features   : {_status_features(cache_dir)}")
    print(f"  membership : {_status_membership(cache_dir)}")
    print(f"  sync       : {_status_sync(dry_run)}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Idempotent processing pipeline")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--channel", default=None,
                        help="Limit ingest/backfill to this channel name")
    parser.add_argument("--status", action="store_true",
                        help="Print status overview and exit without running any phase")
    parser.add_argument(
        "--only", choices=PHASES, metavar="PHASE",
        help=f"Run only this phase: {', '.join(PHASES)}",
    )
    parser.add_argument(
        "--skip", choices=PHASES, action="append", default=[], metavar="PHASE",
        help="Skip this phase (repeatable). E.g. --skip llm --skip sync",
    )
    parser.add_argument("--force-membership", action="store_true",
                        help="Re-fetch playlist membership even if cache is fresh")
    parser.add_argument("--invalidate-membership", action="store_true",
                        help="Delete membership cache before re-fetching")
    parser.add_argument("--execute", action="store_true",
                        help="Sync phase writes to YouTube (default: dry-run)")
    parser.add_argument("--sync-limit", type=int, default=None,
                        help="Cap sync to N videos per channel (default: no limit)")
    args = parser.parse_args()

    cache_dir = args.cache_dir or _default_cache_dir()
    config = args.config or _default_config()

    if args.status:
        _print_status(cache_dir, dry_run=True)
        return 0

    phases_to_run = [args.only] if args.only else PHASES
    phases_to_run = [p for p in phases_to_run if p not in args.skip]

    # ── Status overview ───────────────────────────────────────────────────────
    dry_run = not args.execute
    _print_status(cache_dir, dry_run)
    print()
    print(f"Phases: {' → '.join(phases_to_run)}")
    print("=" * 60)

    # ── Run phases ────────────────────────────────────────────────────────────
    runners = {
        "ingest":     lambda: run_ingest(cache_dir, config, channel=args.channel),
        "features":   lambda: run_features(cache_dir),
        "membership": lambda: run_membership(
            cache_dir, force=args.force_membership, invalidate=args.invalidate_membership
        ),
        "backfill":   lambda: run_backfill(cache_dir, config, dry_run),
        "sync":       lambda: run_sync(config, args.sync_limit, dry_run),
    }

    pipeline_start = time.monotonic()
    for phase in phases_to_run:
        print(f"\n{'─' * 60}", flush=True)
        print(f"Phase: {phase}", flush=True)
        print(f"{'─' * 60}", flush=True)
        rc = runners[phase]()
        if rc != 0:
            print(f"\nPipeline stopped at phase '{phase}' (exit {rc})")
            return rc

    total_elapsed = time.monotonic() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete. ({total_elapsed:.0f}s)")
    print()
    _print_status(cache_dir, dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
