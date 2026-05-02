#!/usr/bin/env python3
"""Unsupervised topic discovery — three engines, all sklearn-compatible.

Engines (--engine):
  nmf       TF-IDF → NMF  — fast, deterministic, sharp keywords  [default]
  lda       Count  → LDA  — probabilistic, softer topic boundaries
  bertopic  sentence-transformers → PCA/UMAP → HDBSCAN → c-TF-IDF
            Requires: uv sync --extra topic-discovery

nmf and lda use only scikit-learn (already installed, no extra deps).
BERTopic uses fast-hdbscan (Cython, no numba) to avoid SIGBUS on WSL2.

No seed taxonomy required. Use the output to define the taxonomy in
configs/channels.yaml, then run llm_label.py.

Usage:
    uv run python scripts/discover_topics_bertopic.py --channel "Neutrality Studies"
    uv run python scripts/discover_topics_bertopic.py --channel "Neutrality Studies" --engine lda
    uv run python scripts/discover_topics_bertopic.py --channel "Neutrality Studies" --engine bertopic
    uv run python scripts/discover_topics_bertopic.py --channel "Neutrality Studies" --nr-topics 20
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


# Spoken-language fillers not covered by sklearn's English stop list.
# Includes apostrophe-stripped contractions from auto-generated transcripts
# (e.g. "that's" → "thats", "don't" → "dont").
_TRANSCRIPT_FILLERS = frozenset([
    # hesitation / filler
    "uh", "um", "yeah", "okay", "ok", "right", "gonna", "wanna", "gotta",
    "let", "just", "really", "actually", "basically", "literally",
    "kind", "sort", "thing", "things", "way", "lot",
    "think", "said", "say", "says", "saying", "mean", "means",
    "look", "looks", "come", "coming", "goes", "going", "getting",
    "want", "wanted", "see", "saw", "know", "like",
    # apostrophe-stripped contractions (auto-transcript artefacts)
    "thats", "dont", "doesnt", "didnt", "wont", "cant", "wouldnt",
    "shouldnt", "couldnt", "isnt", "wasnt", "arent", "werent",
    "havent", "hasnt", "hadnt", "theyre", "youre", "youve", "youd", "youll",
    "im", "ive", "id", "ill", "theres", "whats", "whos", "hes", "shes",
    "theyd", "theyll", "theyve", "weve", "wed", "itll", "itd",
])


def _make_lemma_tokenizer():
    import re
    import nltk
    from nltk.stem import WordNetLemmatizer
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    _lem = WordNetLemmatizer()
    # Matches sklearn's default token pattern (words of 2+ chars)
    _token_re = re.compile(r"(?u)\b[a-z]{2,}\b")
    def tokenizer(text: str) -> list[str]:
        return [_lem.lemmatize(t) for t in _token_re.findall(text.lower())]
    return tokenizer


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
        tokenizer=_make_lemma_tokenizer(),
        token_pattern=None,
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


# ── Engine: sklearn LDA ───────────────────────────────────────────────────────

def _run_lda(
    documents: list[str],
    titles: list[str],
    nr_topics: int,
    min_topic_size: int,
) -> list[dict]:
    print(f"[2/3] Fitting Count + LDA ({nr_topics} topics) ...", flush=True)
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from sklearn.decomposition import LatentDirichletAllocation

    stop_words = list(ENGLISH_STOP_WORDS | _TRANSCRIPT_FILLERS)
    # LDA requires raw term counts (not TF-IDF weights)
    vectorizer = CountVectorizer(
        max_df=0.70,
        min_df=max(2, min_topic_size // 2),
        max_features=5000,
        ngram_range=(1, 2),
        stop_words=stop_words,
        tokenizer=_make_lemma_tokenizer(),
        token_pattern=None,
    )
    tf = vectorizer.fit_transform(documents)
    print(f"  Vocabulary: {len(vectorizer.get_feature_names_out())} terms", flush=True)

    model = LatentDirichletAllocation(
        n_components=nr_topics,
        random_state=42,
        max_iter=20,
        learning_method="online",
        n_jobs=-1,
    )
    W = model.fit_transform(tf)  # doc-topic matrix
    H = model.components_         # topic-term matrix
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
    embedding_docs: list[str],
    titles: list[str],
    video_ids: list[str],
    cache_dir: Path,
    channel_slug: str,
    nr_topics: int | None,
    min_topic_size: int,
    embedding_model_name: str,
    dim_reduction: str,
) -> tuple[list[dict], list[int]]:
    # NUMBA_DISABLE_JIT must be set before any numba-backed package is imported.
    # umap-learn and hdbscan both trigger numba at load time and SIGBUS on WSL2.
    import os
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_bertopic_cache")

    print("[1/5] Importing BERTopic stack ...", flush=True)
    try:
        import numpy as np
        print("  numpy ... ok", flush=True)
        from bertopic import BERTopic
        print("  bertopic ... ok", flush=True)
        from sklearn.cluster import KMeans
        from sentence_transformers import SentenceTransformer
        print("  sentence_transformers ... ok", flush=True)
        if dim_reduction == "umap":
            try:
                from umap import UMAP
                print("  umap ... ok", flush=True)
            except ImportError:
                print("  umap not installed — umap-learn is excluded from deps on WSL2", file=sys.stderr)
                print("  Use --dim-reduction pca (default) instead.", file=sys.stderr)
                sys.exit(1)
        else:
            from sklearn.decomposition import PCA
            print("  sklearn PCA ... ok", flush=True)
    except ImportError as e:
        print(f"  Missing: {e}", file=sys.stderr)
        print("  Install with: uv sync --extra topic-discovery", file=sys.stderr)
        sys.exit(1)

    # Embed using truncated docs (encoder max ~256-512 tokens); cache by count
    embeddings_path = cache_dir / f"bertopic_embeddings_{channel_slug}.npy"
    if embeddings_path.exists():
        print(f"[3/5] Loading cached embeddings: {embeddings_path}", flush=True)
        embeddings = np.load(str(embeddings_path))
        if embeddings.shape[0] != len(embedding_docs):
            print("  Size mismatch — discarding cache.", flush=True)
            embeddings = None
        else:
            print(f"  Loaded {embeddings.shape}", flush=True)
    else:
        embeddings = None

    if embeddings is None:
        print(f"[3/5] Embedding {len(embedding_docs)} docs with {embedding_model_name} ...", flush=True)
        emb_model = SentenceTransformer(embedding_model_name)
        embeddings = emb_model.encode(embedding_docs, show_progress_bar=True, batch_size=32)
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

    # KMeans: no hdbscan/numba dep; assigns all docs to a topic (no outlier cluster).
    # BERTopic accepts any sklearn-compatible clusterer via the hdbscan_model param.
    n_clusters = nr_topics if nr_topics else 20
    cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")

    # Pass our stop-word list and lemmatizer so BERTopic's c-TF-IDF produces clean keywords.
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    stop_words = list(ENGLISH_STOP_WORDS | _TRANSCRIPT_FILLERS)
    vectorizer_model = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        tokenizer=_make_lemma_tokenizer(),
        token_pattern=None,
    )

    print("[5/5] Fitting BERTopic ...", flush=True)
    topic_model = BERTopic(
        umap_model=dim_model, hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        nr_topics=nr_topics, calculate_probabilities=False, verbose=False,
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
    parser.add_argument("--engine", choices=["nmf", "lda", "bertopic"], default="nmf",
                        help="nmf = TF-IDF+NMF (default); lda = LDA; bertopic = BERTopic")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="[bertopic only] sentence-transformers model")
    parser.add_argument("--dim-reduction", choices=["pca", "umap"], default="pca",
                        help="[bertopic only] dim reduction (default: pca)")
    args = parser.parse_args()

    cache_dir = args.cache_dir or _default_cache_dir()
    config = _load_config(args.config)
    channel = _select_channel(config, args.channel)

    # bertopic: full transcript for c-TF-IDF; encoder sees only first _TRANSCRIPT_CHARS_BERTOPIC
    transcript_limit = None if args.engine in ("nmf", "lda") else _TRANSCRIPT_CHARS_BERTOPIC
    if args.no_transcripts:
        transcript_desc = "no"
    elif args.engine == "bertopic":
        transcript_desc = f"full (encoder truncated to {transcript_limit} chars)"
    else:
        transcript_desc = "full"

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
    documents, embedding_docs, video_ids, titles = [], [], [], []
    no_transcript = 0
    for e in entries:
        vid = e["video_id"]
        meta = cache_mod.read_metadata(cache_dir, vid)
        if not meta:
            continue
        transcript_full = _load_transcript(cache_dir, vid, max_chars=None)
        if not transcript_full:
            no_transcript += 1
        # Full transcript for c-TF-IDF keyword extraction
        doc_full = _build_document(meta.get("title", ""), meta.get("description", ""),
                                   transcript_full, args.no_transcripts)
        # Truncated for sentence-transformer encoder (model max ~256-512 tokens)
        transcript_trunc = transcript_full[:transcript_limit] if transcript_limit and transcript_full else transcript_full
        doc_trunc = _build_document(meta.get("title", ""), meta.get("description", ""),
                                    transcript_trunc, args.no_transcripts)
        if doc_full.strip():
            documents.append(doc_full)
            embedding_docs.append(doc_trunc)
            video_ids.append(vid)
            titles.append(meta.get("title", ""))

    print(f"  {len(documents)} documents (no transcript: {no_transcript})", flush=True)

    if len(documents) < args.min_topic_size * 2:
        print(f"Too few documents ({len(documents)}).", file=sys.stderr)
        return 1

    # Run chosen engine
    if args.engine == "nmf":
        topics = _run_nmf(documents, titles, args.nr_topics, args.min_topic_size)
    elif args.engine == "lda":
        topics = _run_lda(documents, titles, args.nr_topics, args.min_topic_size)
    else:
        topics, _ = _run_bertopic(
            documents, embedding_docs, titles, video_ids, cache_dir, channel.slug,
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
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print("Next: review topics, define taxonomy in configs/channels.yaml, then: just llm-label-all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
