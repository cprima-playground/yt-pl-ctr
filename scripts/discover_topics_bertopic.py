#!/usr/bin/env python3
"""Unsupervised topic discovery — TF-IDF + NMF (default) or BERTopic (--engine bertopic).

Default engine uses only scikit-learn (already installed, no extra deps):
  TF-IDF vectorisation → NMF decomposition → top keywords per topic

BERTopic engine (richer but requires heavy native extensions that may SIGBUS on WSL2):
  sentence-transformers embeddings → PCA/UMAP → HDBSCAN → c-TF-IDF
  Requires: uv sync --extra topic-discovery

No seed taxonomy required. Use the output to define the taxonomy in
configs/channels.yaml, then run llm_label.py.

Usage:
    # Default (sklearn TF-IDF + NMF, works everywhere):
    uv run python scripts/discover_topics_bertopic.py --channel "Neutrality Studies"

    # Control topic count:
    uv run python scripts/discover_topics_bertopic.py --channel "Neutrality Studies" --nr-topics 20

    # BERTopic engine (requires extra deps):
    uv run python scripts/discover_topics_bertopic.py --channel "Neutrality Studies" --engine bertopic
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
from yt_pl_ctr.youtube import YouTubeClient, YouTubeAPIError

# BERTopic engine: sentence-transformers truncate at 512 tokens (~2000 chars)
_TRANSCRIPT_CHARS_BERTOPIC = 3000
# NMF engine: TF-IDF handles arbitrary length — use full transcript
_TRANSCRIPT_CHARS_NMF = None  # no limit
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


def _load_transcript(cache_dir: Path, video_id: str, max_chars: int | None = None) -> str:
    path = cache_dir / "episodes" / video_id / "transcript.txt"
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars] if max_chars else text
    return ""


def _build_document(title: str, description: str, transcript: str, no_transcripts: bool) -> str:
    parts = [title, title]  # repeat title to anchor topic signal
    if description:
        parts.append(description[:_DESCRIPTION_CHARS])
    if not no_transcripts and transcript:
        parts.append(transcript)
    return " ".join(p.strip() for p in parts if p.strip())


# Spoken-language fillers not covered by sklearn's English stop list
_TRANSCRIPT_FILLERS = frozenset([
    "uh", "um", "yeah", "okay", "ok", "right", "gonna", "wanna", "gotta",
    "let", "just", "really", "actually", "basically", "literally",
    "kind", "sort", "thing", "things", "way", "lot",
    "think", "said", "say", "says", "saying", "mean", "means",
    "look", "looks", "come", "coming", "goes", "going", "getting",
    "want", "wanted", "see", "saw", "know", "like",
])


# ── Engine: sklearn TF-IDF + NMF ──────────────────────────────────────────────

def _run_nmf(
    documents: list[str],
    titles: list[str],
    nr_topics: int,
    min_topic_size: int,
) -> list[dict]:
    print(f"[2/3] Fitting TF-IDF + NMF ({nr_topics} topics) ...", flush=True)
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    from sklearn.decomposition import NMF

    stop_words = list(ENGLISH_STOP_WORDS | _TRANSCRIPT_FILLERS)
    vectorizer = TfidfVectorizer(
        max_df=0.70,
        min_df=max(2, min_topic_size // 2),
        max_features=5000,
        ngram_range=(1, 2),
        stop_words=stop_words,
    )
    tfidf = vectorizer.fit_transform(documents)
    print(f"  Vocabulary: {len(vectorizer.get_feature_names_out())} terms", flush=True)

    model = NMF(n_components=nr_topics, random_state=42, max_iter=400)
    W = model.fit_transform(tfidf)  # doc-topic matrix
    H = model.components_            # topic-term matrix
    print("  Done.", flush=True)

    terms = vectorizer.get_feature_names_out()
    topic_assignments = W.argmax(axis=1)

    topics = []
    for tid in range(nr_topics):
        top_idx = H[tid].argsort()[::-1][:10]
        keywords = [terms[i] for i in top_idx]
        doc_indices = [i for i, t in enumerate(topic_assignments) if t == tid]
        if len(doc_indices) < min_topic_size:
            continue
        # sort by topic weight descending for representative titles
        doc_indices_sorted = sorted(doc_indices, key=lambda i: W[i, tid], reverse=True)
        topics.append({
            "topic_id": tid,
            "count": len(doc_indices),
            "keywords": keywords,
            "representative_titles": [titles[i] for i in doc_indices_sorted[:5]],
            "doc_indices": doc_indices,
        })

    topics.sort(key=lambda t: t["count"], reverse=True)
    return topics


# ── Engine: BERTopic ───────────────────────────────────────────────────────────

def _run_bertopic(
    documents: list[str],
    titles: list[str],
    video_ids: list[str],
    cache_dir: Path,
    channel_slug: str,
    nr_topics: int | None,
    min_topic_size: int,
    embedding_model_name: str,
    dim_reduction: str,
) -> tuple[list[dict], list[int]]:
    # Must be set before any numba-backed package is imported (umap-learn triggers numba
    # at load time and SIGBUS on WSL2 without this).
    import os
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_bertopic_cache")

    print("[1/5] Importing BERTopic stack ...", flush=True)
    try:
        import numpy as np
        print("  numpy ... ok", flush=True)
        from bertopic import BERTopic
        print("  bertopic ... ok", flush=True)
        from fast_hdbscan import HDBSCAN
        print("  fast_hdbscan ... ok", flush=True)
        from sentence_transformers import SentenceTransformer
        print("  sentence_transformers ... ok", flush=True)
        if dim_reduction == "umap":
            from umap import UMAP
            print("  umap ... ok", flush=True)
        else:
            from sklearn.decomposition import PCA
            print("  sklearn PCA ... ok", flush=True)
    except ImportError as e:
        print(f"  Missing: {e}", file=sys.stderr)
        print("  Install with: uv sync --extra topic-discovery", file=sys.stderr)
        sys.exit(1)

    # Embed (cached)
    embeddings_path = cache_dir / f"bertopic_embeddings_{channel_slug}.npy"
    if embeddings_path.exists():
        print(f"[3/5] Loading cached embeddings: {embeddings_path}", flush=True)
        embeddings = np.load(str(embeddings_path))
        if embeddings.shape[0] != len(documents):
            print("  Size mismatch — discarding cache.", flush=True)
            embeddings = None
        else:
            print(f"  Loaded {embeddings.shape}", flush=True)
    else:
        embeddings = None

    if embeddings is None:
        print(f"[3/5] Embedding {len(documents)} docs with {embedding_model_name} ...", flush=True)
        emb_model = SentenceTransformer(embedding_model_name)
        embeddings = emb_model.encode(documents, show_progress_bar=True, batch_size=32)
        np.save(str(embeddings_path), embeddings)
        print(f"  Saved to {embeddings_path}", flush=True)

    n_components = min(50, len(documents) - 1)
    if dim_reduction == "umap":
        print(f"[4/5] UMAP → {n_components}d (low_memory=True) ...", flush=True)
        dim_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                         metric="cosine", low_memory=True)
    else:
        print(f"[4/5] PCA → {n_components}d ...", flush=True)
        dim_model = PCA(n_components=n_components)

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True, core_dist_n_jobs=1,
    )

    print("[5/5] Fitting BERTopic ...", flush=True)
    topic_model = BERTopic(
        umap_model=dim_model, hdbscan_model=hdbscan_model,
        min_topic_size=min_topic_size, nr_topics=nr_topics,
        calculate_probabilities=False, verbose=False,
    )
    raw_topics, _ = topic_model.fit_transform(documents, embeddings)
    print("  Done.", flush=True)

    topic_info = topic_model.get_topic_info()
    named = topic_info[topic_info["Topic"] != -1]
    topics = []
    for _, row in named.sort_values("Count", ascending=False).iterrows():
        tid = row["Topic"]
        doc_indices = [i for i, t in enumerate(raw_topics) if t == tid]
        topics.append({
            "topic_id": int(tid),
            "count": int(row["Count"]),
            "keywords": [w for w, _ in topic_model.get_topic(tid)][:10],
            "representative_titles": [titles[i] for i in doc_indices[:5]],
            "doc_indices": doc_indices,
        })
    return topics, raw_topics


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unsupervised topic discovery (default: TF-IDF + NMF, no extra deps)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent.parent / "configs" / "channels.yaml")
    parser.add_argument("--channel", default=None)
    parser.add_argument("--nr-topics", type=int, default=20,
                        help="Number of topics to discover (default: 20)")
    parser.add_argument("--min-topic-size", type=int, default=3,
                        help="Minimum episodes per topic (default: 3)")
    parser.add_argument("--no-transcripts", action="store_true",
                        help="Use only title + description")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--engine", choices=["nmf", "bertopic"], default="nmf",
                        help="nmf = TF-IDF+NMF, no extra deps (default); bertopic = BERTopic")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="[bertopic only] sentence-transformers model")
    parser.add_argument("--dim-reduction", choices=["pca", "umap"], default="pca",
                        help="[bertopic only] dim reduction (default: pca)")
    args = parser.parse_args()

    cache_dir = args.cache_dir or _default_cache_dir()
    config = _load_config(args.config)
    channel = _select_channel(config, args.channel)

    transcript_limit = None if args.engine == "nmf" else _TRANSCRIPT_CHARS_BERTOPIC
    transcript_desc = "no" if args.no_transcripts else ("full" if not transcript_limit else f"first {transcript_limit} chars")

    print(f"Channel    : {channel.name}")
    print(f"Engine     : {args.engine}")
    print(f"Topics     : {args.nr_topics}")
    print(f"Transcripts: {transcript_desc}")
    print()

    # Load index and filter
    index = cache_mod.read_index(cache_dir)
    if not index:
        print("Cache index is empty — run download_test_data.py first.")
        return 1

    # Resolve channel_id from URL if not set in config
    channel_id = channel.channel_id
    if not channel_id:
        try:
            youtube = YouTubeClient.from_env()
            channel_id = youtube.resolve_channel_id(channel.url)
            print(f"Resolved channel_id for {channel.name!r}: {channel_id}")
        except (YouTubeAPIError, Exception) as e:
            print(f"Warning: could not resolve channel_id for {channel.name!r}: {e}", file=sys.stderr)
            print("  Add channel_id to configs/channels.yaml to avoid this.", file=sys.stderr)

    if channel_id:
        entries = []
        for e in index:
            meta = cache_mod.read_metadata(cache_dir, e["video_id"])
            if meta and meta.get("channel_id") == channel_id:
                entries.append(e)
        print(f"Channel filter ({channel.name}): {len(entries)}/{len(index)} episodes")
    else:
        entries = list(index)
        print(f"Warning: no channel_id — using all {len(entries)} episodes from cache", file=sys.stderr)

    if channel.min_duration:
        entries = [e for e in entries if (e.get("duration") or 0) >= channel.min_duration]
    cutoff = channel.min_upload_date_str()
    if cutoff:
        entries = [e for e in entries if (e.get("upload_date") or "") >= cutoff]
    if args.limit:
        entries = entries[:args.limit]

    print(f"[1/3] Loading {len(entries)} episodes ...", flush=True)
    documents, video_ids, titles = [], [], []
    no_transcript = 0
    for e in entries:
        vid = e["video_id"]
        meta = cache_mod.read_metadata(cache_dir, vid)
        if not meta:
            continue
        transcript = _load_transcript(cache_dir, vid, max_chars=transcript_limit)
        if not transcript:
            no_transcript += 1
        doc = _build_document(meta.get("title", ""), meta.get("description", ""),
                              transcript, args.no_transcripts)
        if doc.strip():
            documents.append(doc)
            video_ids.append(vid)
            titles.append(meta.get("title", ""))

    print(f"  {len(documents)} documents (no transcript: {no_transcript})", flush=True)

    if len(documents) < args.min_topic_size * 2:
        print(f"Too few documents ({len(documents)}).", file=sys.stderr)
        return 1

    # Run chosen engine
    if args.engine == "nmf":
        topics = _run_nmf(documents, titles, args.nr_topics, args.min_topic_size)
    else:
        topics, _ = _run_bertopic(
            documents, titles, video_ids, cache_dir, channel.slug,
            args.nr_topics, args.min_topic_size,
            args.embedding_model, args.dim_reduction,
        )

    # ── Print results ──────────────────────────────────────────────────────────
    outliers = len(documents) - sum(t["count"] for t in topics)
    print(f"\nDiscovered {len(topics)} topics  |  {outliers} unclustered\n")
    print(f"{'ID':<4} {'Count':>6}  Keywords")
    print("─" * 72)
    for t in topics:
        print(f"{t['topic_id']:<4} {t['count']:>6}  {', '.join(t['keywords'][:8])}")

    print()
    print("── Representative episodes per topic ──")
    for t in topics:
        print(f"\nTopic {t['topic_id']}  [{', '.join(t['keywords'][:5])}]")
        for title in t["representative_titles"][:3]:
            print(f"  • {title[:90]}")

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = cache_dir / f"topics_{channel.slug}.json"
    out_path.write_text(json.dumps({
        "channel": channel.name,
        "engine": args.engine,
        "total_documents": len(documents),
        "n_topics": len(topics),
        "n_unclustered": outliers,
        "topics": [{k: v for k, v in t.items() if k != "doc_indices"} for t in topics],
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")
    print("Next: review topics, define taxonomy in configs/channels.yaml, then: just llm-label-all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
