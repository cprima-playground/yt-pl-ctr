#!/usr/bin/env python3
"""Discover granular topic distribution across all cached JRE episodes.

Assigns each episode a single specific topic from a flat list of ~40 labels.
Outputs a ranked frequency table so you can decide which topics warrant playlists.
No hierarchy is enforced — the granularity is just for analytical clarity.

This is a standalone exploration tool — it writes nothing to the pipeline.

Outputs (in $YT_CACHE_DIR/):
  topic_discovery.json   — per-episode topic assignments
  topic_summary.txt      — ranked frequency table printed to stdout and saved

Usage:
    uv run python scripts/discover_topics.py
    uv run python scripts/discover_topics.py --limit 200
    uv run python scripts/discover_topics.py --only-other     # episodes not yet in a playlist
    uv run python scripts/discover_topics.py --model claude-haiku-4-5-20251001
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from datetime import timedelta

import anthropic
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cache as cache_mod

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Granular flat topic list — pick the most specific label that fits
CANDIDATE_TOPICS = [
    # History & Unexplained
    "ancient civilizations",
    "ufo / extraterrestrial",
    "paranormal / cryptozoology",
    "conspiracy theories",
    # Science
    "space & astronomy",
    "physics & mathematics",
    "biology & evolution",
    "natural history & wildlife",
    "environment & climate",
    "anthropology & archaeology",
    # Health & Mind
    "health & medicine",
    "nutrition & diet",
    "neuroscience",
    "longevity & biohacking",
    "psychology & mental health",
    "drugs & psychedelics",
    # Technology & Economy
    "technology",
    "artificial intelligence",
    "cryptocurrency & finance",
    "economics & business",
    "entrepreneurship",
    # Society & Culture
    "politics & government",
    "law & justice",
    "true crime",
    "military & veterans",
    "religion & spirituality",
    "philosophy",
    "social issues & culture",
    # Arts & Entertainment
    "comedy",
    "music",
    "film & television",
    "literature & storytelling",
    # Sports & Outdoors
    "combat sports & mma",
    "hunting & fishing",
    "fitness & training",
    "extreme sports & adventure",
    # Lifestyle
    "motivation & self-improvement",
    "relationships & dating",
    "food & cooking",
]

_SYSTEM = f"""You classify Joe Rogan Experience podcast episodes by their single primary topic.

Candidate topics (pick exactly one, use the exact string):
{chr(10).join(f"  - {t}" for t in CANDIDATE_TOPICS)}
  - other: <short phrase>   (use this only if nothing above fits; replace <short phrase> with the actual topic)

Rules:
- Reply with JSON only, no prose:
  {{
    "topic": "<topic string>",
    "secondary_topic": "<topic string or null>",
    "confidence": <0.0–1.0>,
    "reason": "<one sentence>"
  }}
- Choose the topic that best describes the *guest's expertise* and the *main discussion theme*
- secondary_topic: the next-best fit if the episode clearly spans two topics, otherwise null
- confidence: float 0.0–1.0 (1.0 = unambiguous, <0.7 = genuinely ambiguous)
- If the episode is a Fight Companion or solo rant, use the dominant theme discussed
- "other: <phrase>" must be specific, e.g. "other: entrepreneurship" not just "other"
"""


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "cache"


def _build_user_prompt(meta: dict, wikipedia: dict | None, features: dict | None) -> str:
    title = meta.get("title", "")
    desc = (meta.get("description", "") or "").split("\n")[0][:300]

    parts = [f"Title: {title}"]
    if desc:
        parts.append(f"Description: {desc}")

    if wikipedia and wikipedia.get("found"):
        summary = wikipedia.get("summary", "")[:400]
        parts.append(f"Wikipedia (guest): {summary}")

    if features and features.get("top_tokens"):
        top = [t for t, _ in features["top_tokens"][:20]]
        parts.append(f"Transcript top words: {', '.join(top)}")

    return "\n".join(parts)


def _discover_episode(
    client: anthropic.Anthropic,
    meta: dict,
    wikipedia: dict | None,
    features: dict | None,
    model: str,
) -> dict:
    user_msg = _build_user_prompt(meta, wikipedia, features)
    response = client.messages.create(
        model=model,
        max_tokens=150,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else {"topic": "other: parse error", "confidence": "low"}


def _load_existing(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}
    data = json.loads(output_path.read_text(encoding="utf-8"))
    return {r["video_id"]: r for r in data}


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover primary topics across JRE episodes")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N episodes")
    parser.add_argument("--only-other", action="store_true",
                        help="Only process episodes labeled 'other' in llm_labeled.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
        return 1

    cache_dir = args.cache_dir or _default_cache_dir()
    output_path = args.output or cache_dir / "topic_discovery.json"

    index = cache_mod.read_index(cache_dir)
    if not index:
        print("Cache index empty — run fetch_transcripts.py first.")
        return 1

    # Optionally filter to only episodes labeled 'other' by llm_label.py
    if args.only_other:
        llm_path = cache_dir / "llm_labeled.json"
        if not llm_path.exists():
            print("llm_labeled.json not found — run llm_label.py first.")
            return 1
        llm_data = json.loads(llm_path.read_text(encoding="utf-8"))
        other_ids = {
            r["video_id"] for r in llm_data
            if r.get("category") == "other"
        }
        index = [e for e in index if e["video_id"] in other_ids]
        print(f"Filtered to {len(index)} episodes labeled 'other'")

    if args.limit:
        index = index[:args.limit]

    existing = _load_existing(output_path)
    client = anthropic.Anthropic(api_key=api_key)

    total = len(index)
    new_count = 0
    results = dict(existing)
    t_start = time.monotonic()

    print(f"Model:   {args.model}")
    print(f"Target:  {total} episodes  ({len(existing)} already cached)")
    print(f"Output:  {output_path}")
    print()

    for i, entry in enumerate(index, 1):
        vid = entry["video_id"]

        if vid in results:
            continue

        meta = cache_mod.read_metadata(cache_dir, vid)
        if meta is None:
            continue

        wikipedia = cache_mod.read_wikipedia(cache_dir, vid)
        features_f = cache_mod.episode_dir(cache_dir, vid) / "features.json"
        features = json.loads(features_f.read_text(encoding="utf-8")) if features_f.exists() else None

        title = meta.get("title", vid)[:55]
        print(f"  [{i}/{total}] {title:<55} ", end="", flush=True)

        try:
            result = _discover_episode(client, meta, wikipedia, features, args.model)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str:
                # Exponential backoff: 30s, 60s, 120s
                for wait in (30, 60, 120):
                    print(f"rate limit — waiting {wait}s ...", flush=True)
                    time.sleep(wait)
                    try:
                        result = _discover_episode(client, meta, wikipedia, features, args.model)
                        break
                    except Exception:
                        continue
                else:
                    print(f"gave up after retries, skipping")
                    continue
            else:
                print(f"error: {e}")
                time.sleep(2)
                continue

        topic = result.get("topic", "other: unknown")
        secondary = result.get("secondary_topic") or None
        conf = result.get("confidence", 0.0)
        if isinstance(conf, str):
            conf = {"high": 0.9, "medium": 0.65, "low": 0.4}.get(conf, 0.5)
        reason = result.get("reason", "")
        conf_icon = "✓" if conf >= 0.8 else ("~" if conf >= 0.6 else "?")
        sec_str = f"  [{secondary}]" if secondary else ""
        print(f"{conf_icon} {conf:.2f}  {topic}{sec_str}")

        results[vid] = {
            "video_id": vid,
            "title": meta.get("title", ""),
            "topic": topic,
            "secondary_topic": secondary,
            "confidence": conf,
            "reason": reason,
        }
        new_count += 1
        time.sleep(1.5)  # throttle to ~40 req/min, leaving headroom for concurrent jobs

        if new_count % 20 == 0:
            output_path.write_text(json.dumps(list(results.values()), indent=2, ensure_ascii=False), encoding="utf-8")

        # Milestone: every 100 new episodes print elapsed, rate, ETA, and top-5 topics
        if new_count % 100 == 0:
            elapsed = time.monotonic() - t_start
            rate = new_count / (elapsed / 60) if elapsed > 0 else 0  # eps/min
            remaining = total - i
            eta_secs = (remaining / rate * 60) if rate > 0 else 0
            elapsed_fmt = str(timedelta(seconds=int(elapsed)))
            eta_fmt = str(timedelta(seconds=int(eta_secs)))
            counts_so_far = Counter(r["topic"] for r in results.values())
            top5 = "  |  ".join(f"{t} ({n})" for t, n in counts_so_far.most_common(5))
            print(f"\n── [{new_count} new / {i} seen] elapsed {elapsed_fmt}  rate {rate:.1f}/min  ETA {eta_fmt}")
            print(f"   Top 5: {top5}\n")

    output_path.write_text(json.dumps(list(results.values()), indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Summary ────────────────────────────────────────────────────────────────
    from collections import defaultdict
    counts = Counter(r["topic"] for r in results.values())
    conf_by_topic: dict[str, list[float]] = defaultdict(list)
    for r in results.values():
        c = r.get("confidence", 0.0)
        if isinstance(c, str):
            c = {"high": 0.9, "medium": 0.65, "low": 0.4}.get(c, 0.5)
        conf_by_topic[r["topic"]].append(c)

    summary_lines = [
        f"Topic discovery — {len(results)} episodes",
        f"{'─' * 65}",
        f"{'Topic':<35} {'count':>6}  {'avg_conf':>9}  {'low(<0.7)':>9}",
        f"{'─' * 65}",
    ]
    for topic, n in counts.most_common():
        confs = conf_by_topic[topic]
        avg = sum(confs) / len(confs)
        low = sum(1 for c in confs if c < 0.7)
        summary_lines.append(f"  {topic:<33} {n:>6}  {avg:>9.2f}  {low:>9}")

    summary_lines.append(f"{'─' * 65}")
    summary = "\n".join(summary_lines)
    print("\n" + summary)

    summary_path = cache_dir / "topic_summary.txt"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")
    print(f"Raw data:      {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
