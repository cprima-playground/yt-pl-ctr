#!/usr/bin/env python3
"""Unsupervised topic discovery using BERTopic.

Combines title + description + transcript for each episode, embeds with
sentence-transformers, reduces with UMAP, clusters with HDBSCAN, and extracts
topic keywords via c-TF-IDF. No seed taxonomy required.

Use the output to define the taxonomy in configs/channels.yaml, then run
llm_label.py to produce fine-grained per-episode labels for classifier training.

Install extra dependencies first:
    uv sync --extra topic-discovery

Usage:
    uv run python scripts/discover_topics_bertopic.py --channel "Neutrality Studies"
    uv run python scripts/discover_topics_bertopic.py --channel "Neutrality Studies" --min-topic-size 5
    uv run python scripts/discover_topics_bertopic.py --channel "Neutrality Studies" --nr-topics 20
    uv run python scripts/discover_topics_bertopic.py --channel "Neutrality Studies" --no-transcripts
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
from yt_pl_ctr.models import Config, ChannelConfig

# Transcript is truncated to keep embeddings within sentence-transformer token limits.
# The opening ~3000 chars typically cover the intro and main topic framing.
_TRANSCRIPT_CHARS = 3000
_DESCRIPTION_CHARS = 800


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "cache"


def _load_config(path: Path) -> Config:
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f))


def _select_channel(config: Config, channel_name: str | None) -> ChannelConfig:
    if channel_name:
        matches = [c for c in config.channels if c.name.lower() == channel_name.lower()]
        if not matches:
            print(f"Channel {channel_name!r} not found. Available: {[c.name for c in config.channels]}",
                  file=sys.stderr)
            sys.exit(1)
        return matches[0]
    return config.channels[0]


def _load_transcript(cache_dir: Path, video_id: str) -> str:
    path = cache_dir / "episodes" / video_id / "transcript.txt"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")[:_TRANSCRIPT_CHARS]
    return ""


def _build_document(title: str, description: str, transcript: str) -> str:
    """Combine fields into one document. Title repeated for embedding weight."""
    parts = [title, title]  # repeat title so it anchors the topic signal
    if description:
        parts.append(description[:_DESCRIPTION_CHARS])
    if transcript:
        parts.append(transcript)
    return " ".join(p.strip() for p in parts if p.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unsupervised topic discovery with BERTopic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent.parent / "configs" / "channels.yaml",
    )
    parser.add_argument("--channel", default=None,
                        help="Channel name from config (default: first)")
    parser.add_argument("--min-topic-size", type=int, default=3,
                        help="Minimum episodes per topic cluster (default: 3)")
    parser.add_argument("--nr-topics", type=int, default=None,
                        help="Target number of topics; None = auto (default: auto)")
    parser.add_argument("--no-transcripts", action="store_true",
                        help="Use only title + description (faster, less signal)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N episodes")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)")
    args = parser.parse_args()

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Missing dependencies. Install with:", file=sys.stderr)
        print("  uv sync --extra topic-discovery", file=sys.stderr)
        return 1

    cache_dir = args.cache_dir or _default_cache_dir()
    config = _load_config(args.config)
    channel = _select_channel(config, args.channel)

    index = cache_mod.read_index(cache_dir)
    if not index:
        print("Cache index is empty — run download_test_data.py first.")
        return 1

    # Filter to channel
    if channel.channel_id:
        entries = []
        for e in index:
            meta = cache_mod.read_metadata(cache_dir, e["video_id"])
            if meta and meta.get("channel_id") == channel.channel_id:
                entries.append(e)
    else:
        entries = list(index)

    if channel.min_duration:
        entries = [e for e in entries if (e.get("duration") or 0) >= channel.min_duration]

    cutoff = channel.min_upload_date_str()
    if cutoff:
        entries = [e for e in entries if (e.get("upload_date") or "") >= cutoff]

    if args.limit:
        entries = entries[:args.limit]

    total = len(entries)
    print(f"Channel  : {channel.name}")
    print(f"Episodes : {total}")
    print(f"Model    : {args.embedding_model}")
    print(f"Transcripts: {'no' if args.no_transcripts else 'yes (first %d chars)' % _TRANSCRIPT_CHARS}")
    print()

    # Build documents
    documents: list[str] = []
    video_ids: list[str] = []
    titles: list[str] = []

    print("Loading episode text...", flush=True)
    no_transcript = 0
    for e in entries:
        vid = e["video_id"]
        meta = cache_mod.read_metadata(cache_dir, vid)
        if meta is None:
            continue

        title = meta.get("title", "")
        description = meta.get("description", "")
        transcript = "" if args.no_transcripts else _load_transcript(cache_dir, vid)
        if not transcript:
            no_transcript += 1

        doc = _build_document(title, description, transcript)
        if doc.strip():
            documents.append(doc)
            video_ids.append(vid)
            titles.append(title)

    print(f"Documents built: {len(documents)} (missing transcript: {no_transcript})")
    print()

    if len(documents) < args.min_topic_size * 2:
        print(f"Too few documents ({len(documents)}) for meaningful clustering.", file=sys.stderr)
        return 1

    # Embed
    print(f"Embedding with {args.embedding_model} (first run downloads the model)...", flush=True)
    embedding_model = SentenceTransformer(args.embedding_model)

    # Fit BERTopic
    print("Fitting BERTopic...", flush=True)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=args.min_topic_size,
        nr_topics=args.nr_topics,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(documents)

    # ── Results ────────────────────────────────────────────────────────────────

    topic_info = topic_model.get_topic_info()
    # Topic -1 is the outlier cluster (unclustered docs)
    named_topics = topic_info[topic_info["Topic"] != -1]
    outliers = topic_info[topic_info["Topic"] == -1]["Count"].values[0] if -1 in topic_info["Topic"].values else 0

    print(f"\nDiscovered {len(named_topics)} topics  |  {outliers} unclustered episodes\n")
    print(f"{'#':<4} {'Count':>6}  {'Keywords'}")
    print("─" * 70)

    topic_assignments: list[dict] = []
    for _, row in named_topics.sort_values("Count", ascending=False).iterrows():
        tid = row["Topic"]
        count = row["Count"]
        keywords = [w for w, _ in topic_model.get_topic(tid)][:8]
        print(f"{tid:<4} {count:>6}  {', '.join(keywords)}")

    print()

    # Per-topic: show 3 representative episode titles
    print("── Representative episodes per topic ──")
    for _, row in named_topics.sort_values("Count", ascending=False).iterrows():
        tid = row["Topic"]
        keywords = [w for w, _ in topic_model.get_topic(tid)][:5]
        print(f"\nTopic {tid}  [{', '.join(keywords)}]")
        indices = [i for i, t in enumerate(topics) if t == tid][:3]
        for idx in indices:
            print(f"  • {titles[idx][:90]}")

    # ── Save results ───────────────────────────────────────────────────────────

    out_path = cache_dir / f"bertopic_{channel.slug}.json"
    results = {
        "channel": channel.name,
        "total_documents": len(documents),
        "n_topics": len(named_topics),
        "n_outliers": int(outliers),
        "topics": [
            {
                "topic_id": int(row["Topic"]),
                "count": int(row["Count"]),
                "keywords": [w for w, _ in topic_model.get_topic(row["Topic"])][:10],
                "representative_titles": [
                    titles[i] for i in
                    [j for j, t in enumerate(topics) if t == row["Topic"]][:5]
                ],
            }
            for _, row in named_topics.sort_values("Count", ascending=False).iterrows()
        ],
        "episode_assignments": [
            {"video_id": vid, "title": title, "topic_id": int(topic)}
            for vid, title, topic in zip(video_ids, titles, topics)
        ],
    }
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved: {out_path}")
    print("\nNext step: review topics above, define taxonomy in configs/channels.yaml,")
    print("then run: just llm-label-all")

    return 0


if __name__ == "__main__":
    sys.exit(main())
