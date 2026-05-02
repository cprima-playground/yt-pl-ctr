#!/usr/bin/env python3
"""LLM-assisted episode labeling using Claude API.

Classifies unlabeled episodes using title + description + Wikipedia summary +
top transcript tokens. Outputs to llm_labeled.json for review before merging
into the training set.

Usage:
    # Target specific episodes (recommended first — verify quality)
    uv run python scripts/llm_label.py --video-id MtoPEub7XwA --video-id znQB0FumtV8

    # Label N episodes from unlabeled.json
    uv run python scripts/llm_label.py --limit 20

    # Label all unlabeled episodes (~$0.35 with Haiku)
    uv run python scripts/llm_label.py --all

    # Use a smarter model for hard cases
    uv run python scripts/llm_label.py --limit 20 --model claude-sonnet-4-6
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

import cache as cache_mod
from yt_pl_ctr.models import Config

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "cache"


def _load_config(config_path: Path) -> Config:
    with open(config_path) as f:
        return Config.model_validate(yaml.safe_load(f))


def _build_system_prompt(leaf_nodes: list, channel_name: str = "podcast") -> str:
    """Build system prompt from taxonomy leaf nodes (TaxonomyNode objects)."""
    cat_lines = "\n".join(
        f"  - {node.slug}: {node.label}"
        for node in leaf_nodes
    )
    return f"""You are classifying {channel_name} episodes by their primary topic.

Topics (use the exact slug):
{cat_lines}
  - other: doesn't fit any topic above

Rules:
- Reply with JSON only:
  {{"category": "<slug>", "confidence": "high|medium|low", "reason": "<one short sentence>"}}
- Use the exact slug from the list above
- "other" is a valid answer if nothing fits clearly
- Base your decision on the actual conversation content, not guest identity"""


def _build_user_prompt(meta: dict, wikipedia: dict | None, features: dict | None) -> str:
    title = meta.get("title", "")
    desc = (meta.get("description", "") or "").split("\n")[0][:300]
    tags = ", ".join((meta.get("tags") or [])[:10])

    parts = [f"Title: {title}"]
    if desc:
        parts.append(f"Description: {desc}")
    if tags:
        parts.append(f"Tags: {tags}")

    if wikipedia and wikipedia.get("found"):
        summary = wikipedia.get("summary", "")[:400]
        parts.append(f"Wikipedia (guest): {summary}")

    if features and features.get("top_tokens"):
        top = [t for t, _ in features["top_tokens"][:20]]
        parts.append(f"Transcript top words: {', '.join(top)}")

    return "\n".join(parts)


def _label_episode(
    client: anthropic.Anthropic,
    system: str,
    meta: dict,
    wikipedia: dict | None,
    features: dict | None,
    model: str,
) -> tuple[dict, str]:
    """Returns (result_dict, user_prompt_text)."""
    user_msg = _build_user_prompt(meta, wikipedia, features)
    response = client.messages.create(
        model=model,
        max_tokens=100,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(m.group()) if m else {
            "category": "other", "confidence": "low", "reason": "parse error"
        }
    return result, user_msg


def _select_channel(config, name: str | None):
    if name is None:
        return config.channels[0]
    matches = [c for c in config.channels if c.name.lower() == name.lower()]
    if not matches:
        names = [c.name for c in config.channels]
        raise SystemExit(f"Channel {name!r} not found. Available: {names}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="LLM-assisted episode labeling")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent.parent / "configs" / "channels.yaml")
    parser.add_argument("--channel", default=None,
                        help="Channel name to label (default: first channel in config)")
    parser.add_argument("--video-id", action="append", dest="video_ids",
                        help="Label specific episode(s) by video ID")
    parser.add_argument("--limit", type=int, default=None,
                        help="Label first N episodes from the cache index")
    parser.add_argument("--all", action="store_true",
                        help="Label all episodes in the cache index (idempotent — skips cached)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Claude model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file (default: cache_dir/llm_labeled_{channel_slug}.json)")
    args = parser.parse_args()

    if not any([args.video_ids, args.limit, args.all]):
        parser.error("Specify --video-id, --limit N, or --all")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
        return 1

    cache_dir = args.cache_dir or _default_cache_dir()
    config = _load_config(args.config)
    channel_config = _select_channel(config, args.channel)
    leaf_nodes = [
        node
        for domain in channel_config.taxonomy
        for node in domain.leaf_nodes()
    ]
    leaf_slugs = [n.slug for n in leaf_nodes]

    taxonomy_snapshot = [{"slug": n.slug, "label": n.label} for n in leaf_nodes]

    client = anthropic.Anthropic(api_key=api_key)
    system = _build_system_prompt(leaf_nodes, channel_config.name)
    prompt_key = hashlib.md5("|".join(sorted(leaf_slugs)).encode()).hexdigest()[:8]

    # Determine targets
    if args.video_ids:
        targets = [{"video_id": vid} for vid in args.video_ids]
    else:
        index = cache_mod.read_index(cache_dir)
        if not index:
            print("Cache index is empty — run fetch scripts first")
            return 1
        # Filter to this channel's videos (by channel_id if set)
        if channel_config.channel_id:
            before = len(index)
            index = [e for e in index if e.get("channel_id") == channel_config.channel_id]
            print(f"Channel filter: {len(index)}/{before} episodes for {channel_config.name}")
        # Apply min_duration filter
        min_dur = channel_config.min_duration
        if min_dur:
            before = len(index)
            index = [e for e in index if (e.get("duration") or 0) >= min_dur]
            print(f"Duration filter ({min_dur}s): {len(index)}/{before} episodes qualify")
        # Apply max_age_days filter
        cutoff = channel_config.min_upload_date_str()
        if cutoff:
            before = len(index)
            index = [e for e in index if (e.get("upload_date") or "") >= cutoff]
            print(f"Age filter (>= {cutoff}): {len(index)}/{before} episodes qualify")
        targets = index if args.all else index[:args.limit]

    output_path = args.output or cache_dir / f"llm_labeled_{channel_config.slug}.json"
    # Load existing results to allow resuming
    existing = {}
    if output_path.exists():
        for r in json.loads(output_path.read_text()):
            existing[r["video_id"]] = r

    print(f"Model:   {args.model}")
    print(f"Target:  {len(targets)} episodes")
    print(f"Cached:  {len(existing)} already labeled (will skip)")
    print(f"Output:  {output_path}")
    print(flush=True)

    results = dict(existing)
    new_count = 0

    for i, entry in enumerate(targets, 1):
        vid = entry["video_id"]

        if vid in results and not args.video_ids:
            continue  # skip already labeled (unless explicitly targeted)

        meta = cache_mod.read_metadata(cache_dir, vid)
        if meta is None:
            print(f"  [{i}/{len(targets)}] {vid} — no metadata, skipping")
            continue

        wikipedia = cache_mod.read_wikipedia(cache_dir, vid)
        features_f = cache_mod.episode_dir(cache_dir, vid) / "features.json"
        features = json.loads(features_f.read_text()) if features_f.exists() else None

        title = meta.get("title", vid)[:60]

        # Per-episode response cache — avoids re-billing for same model
        cached = cache_mod.read_llm_response(cache_dir, vid, args.model, prompt_key)
        if cached and not args.video_ids:
            label = cached
            results[vid] = {
                "video_id": vid,
                "title": meta.get("title", ""),
                "category": label.get("category", "other"),
                "confidence": label.get("confidence", "low"),
                "reason": label.get("reason", ""),
                "model": args.model,
            }
            continue  # restored from cache, no API call

        print(f"  [{i}/{len(targets)}] {title} ... ", end="", flush=True)

        try:
            label, user_prompt = _label_episode(client, system, meta, wikipedia, features, args.model)
            cache_mod.write_llm_response(
                cache_dir, vid, args.model, label, prompt_key,
                user_prompt=user_prompt,
                system_prompt=system,
                taxonomy_snapshot=taxonomy_snapshot,
            )
        except Exception as e:
            print(f"error: {e}")
            time.sleep(2)
            continue

        record = {
            "video_id": vid,
            "title": meta.get("title", ""),
            "category": label.get("category", "other"),
            "confidence": label.get("confidence", "low"),
            "reason": label.get("reason", ""),
            "model": args.model,
        }
        results[vid] = record
        new_count += 1

        conf_color = {"high": "✓", "medium": "~", "low": "?"}.get(label.get("confidence"), "?")
        print(f"{conf_color} {label.get('category')} — {label.get('reason', '')[:60]}")

        # Flush every 10
        if new_count % 10 == 0:
            output_path.write_text(json.dumps(list(results.values()), indent=2, ensure_ascii=False))

    output_path.write_text(json.dumps(list(results.values()), indent=2, ensure_ascii=False))

    high = sum(1 for r in results.values() if r["confidence"] == "high")
    med  = sum(1 for r in results.values() if r["confidence"] == "medium")
    low  = sum(1 for r in results.values() if r["confidence"] == "low")
    print(f"\nDone. New: {new_count}, Total: {len(results)}")
    print(f"Confidence — high: {high}, medium: {med}, low: {low}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
