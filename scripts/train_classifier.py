#!/usr/bin/env python3
"""Train a TF-IDF + logistic regression classifier on silver-labeled episodes.

Features: title (upweighted), description, transcript — vectorized separately,
then stacked into a single feature matrix.

Outputs (in $YT_CACHE_DIR/model/):
  pipeline.pkl          — fitted sklearn Pipeline (vectorizer + classifier)
  label_encoder.pkl     — LabelEncoder mapping category_key ↔ int
  training_report.json  — CV scores, class distribution, feature info

Usage:
    uv run python scripts/train_classifier.py
    uv run python scripts/train_classifier.py --min-examples 10
    uv run python scripts/train_classifier.py --no-transcript
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cache as cache_mod

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "cache"


def _load_text(cache_dir: Path, record: dict, use_transcript: bool) -> dict[str, str]:
    vid = record["video_id"]
    meta = cache_mod.read_metadata(cache_dir, vid) or {}
    title = meta.get("title", record.get("title", ""))
    desc = meta.get("description", "")
    # Use only first paragraph of description (avoids sponsor noise)
    desc = desc.split("\n")[0] if desc else ""

    transcript = ""
    if use_transcript:
        t_path = cache_dir / "episodes" / vid / "transcript.txt"
        if t_path.exists():
            transcript = t_path.read_text(encoding="utf-8")

    return {"title": title, "description": desc, "transcript": transcript}


def build_dataset(
    cache_dir: Path,
    labeled: list[dict],
    use_transcript: bool,
    min_examples: int,
    protected_slugs: set[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Load text fields for each labeled episode. Drop low-count classes.

    protected_slugs are never dropped regardless of count — use for playlist
    targets that must remain classifiable even with sparse training data.
    """
    counts = Counter(r["category"] for r in labeled)
    protected = protected_slugs or set()
    valid_cats = {cat for cat, n in counts.items() if n >= min_examples or cat in protected}
    dropped = {cat: n for cat, n in counts.items() if cat not in valid_cats}
    if dropped:
        logger.warning("Dropping classes with < %d examples: %s", min_examples, dropped)
    forced = {cat: n for cat, n in counts.items() if cat in protected and n < min_examples}
    if forced:
        logger.warning("Keeping protected playlist slugs despite low count: %s", forced)

    texts, labels = [], []
    missing = 0
    for record in labeled:
        cat = record["category"]
        if cat not in valid_cats:
            continue
        fields = _load_text(cache_dir, record, use_transcript)
        if not fields["title"] and not fields["transcript"]:
            missing += 1
            continue
        texts.append(fields)
        labels.append(cat)

    if missing:
        logger.warning("Skipped %d records with no title or transcript", missing)

    return texts, labels


def build_features(texts: list[dict], use_transcript: bool) -> list[str]:
    """Concatenate fields with title upweighted. Returns flat strings."""
    out = []
    for t in texts:
        # Title repeated 5× to upweight its signal relative to long transcripts
        combined = (t["title"] + " ") * 5
        if t["description"]:
            combined += t["description"] + " "
        if use_transcript and t["transcript"]:
            combined += t["transcript"]
        out.append(combined.strip())
    return out


def _default_model_dir(channel_slug: str = "") -> Path:
    base = Path(__file__).parent.parent / "models"
    return base / channel_slug if channel_slug else base


def train(
    cache_dir: Path,
    channel_slug: str = "",
    model_dir: Path | None = None,
    use_transcript: bool = True,
    min_examples: int = 5,
) -> None:
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder

    labeled_filename = f"llm_labeled_{channel_slug}.json" if channel_slug else "llm_labeled.json"
    llm_path = cache_dir / labeled_filename
    if not llm_path.exists():
        print(f"{labeled_filename} not found at {llm_path}")
        print("Run: just llm-label-all")
        sys.exit(1)

    llm_raw = json.loads(llm_path.read_text())
    # Exclude invented slugs (model hallucinations) and low-confidence labels
    cfg = None
    known_slugs = None
    channel_config = None
    try:
        import yaml
        config_path = Path(__file__).parent.parent / "configs" / "channels.yaml"
        with open(config_path) as f:
            from yt_pl_ctr.models import Config
            cfg = Config.model_validate(yaml.safe_load(f))
        if channel_slug:
            matches = [c for c in cfg.channels if c.slug == channel_slug]
            channel_config = matches[0] if matches else cfg.channels[0]
        else:
            channel_config = cfg.channels[0]
        known_slugs = set(channel_config.all_leaf_slugs()) | {"other"}
    except Exception as e:
        logger.warning("Could not load config for slug validation: %s", e)

    labeled = []
    skipped_slug = 0
    skipped_conf = 0
    for r in llm_raw:
        cat = r.get("category", "other")
        conf = r.get("confidence", "low")
        if known_slugs and cat not in known_slugs:
            skipped_slug += 1
            continue
        if conf == "low":
            skipped_conf += 1
            continue
        labeled.append({
            "video_id": r["video_id"],
            "title": r.get("title", ""),
            "category": cat,
            "match_reason": "llm",
            "skipped": False,
        })

    logger.info(
        "Loaded %d LLM labels (excluded %d invented slugs, %d low-confidence)",
        len(labeled), skipped_slug, skipped_conf,
    )

    playlist_slugs = set(channel_config.playlists.keys()) if channel_config else set()
    texts, labels = build_dataset(cache_dir, labeled, use_transcript, min_examples, playlist_slugs)
    logger.info("Dataset: %d examples, %d classes", len(texts), len(set(labels)))

    counts = Counter(labels)
    print("\nClass distribution:")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<30} {n:>4}")
    print()

    if len(set(labels)) < 2:
        print("Need at least 2 classes to train. Collect more labels first.")
        sys.exit(1)

    raw_texts = build_features(texts, use_transcript)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            min_df=2,
            max_features=30_000,
            ngram_range=(1, 2),
            analyzer="word",
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
        )),
    ])

    # Cross-validation — use min(3, min_class_count) folds
    min_class = min(counts.values())
    n_splits = min(5, min_class)
    if n_splits < 2:
        print(f"Smallest class has {min_class} example(s) — cannot cross-validate.")
        print("Training on full dataset without CV ...", flush=True)
        pipeline.fit(raw_texts, y)
        cv_results = None
    else:
        print(f"Cross-validation ({n_splits}-fold) on {len(raw_texts)} examples × {len(set(labels))} classes ...", flush=True)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = cross_validate(
            pipeline, raw_texts, y, cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            return_train_score=False,
        )
        for metric in ["test_accuracy", "test_f1_macro", "test_f1_weighted"]:
            scores = cv_results[metric]
            print(f"  {metric:<25} {scores.mean():.3f} ± {scores.std():.3f}")
        print()
        print("Fitting final model on full dataset ...", flush=True)
        pipeline.fit(raw_texts, y)

    # Per-class report on full training set (informational)
    from sklearn.metrics import classification_report
    y_pred = pipeline.predict(raw_texts)
    print("Training set classification report:")
    print(classification_report(y, y_pred, target_names=le.classes_))

    # Save artifacts
    if model_dir is None:
        model_dir = _default_model_dir(channel_slug)
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_dir / "pipeline.pkl")
    joblib.dump(le, model_dir / "label_encoder.pkl")

    report = {
        "n_examples": len(texts),
        "classes": le.classes_.tolist(),
        "class_counts": dict(counts),
        "use_transcript": use_transcript,
        "min_examples": min_examples,
        "tfidf_vocab_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "cv_splits": n_splits,
        "cv_scores": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in (cv_results or {}).items()
            if not k.startswith("fit_time") and not k.startswith("score_time")
        },
    }
    (model_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    print(f"\nModel saved to: {model_dir}")
    print(f"  pipeline.pkl       ({len(texts)} examples, "
          f"{len(pipeline.named_steps['tfidf'].vocabulary_)} TF-IDF features)")
    print(f"  label_encoder.pkl  ({len(le.classes_)} classes: {', '.join(le.classes_)})")
    print(f"  training_report.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TF-IDF + logistic regression classifier")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--channel", default=None,
                        help="Channel name to train for (default: first channel in config)")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Where to write model artifacts (default: models/{channel_slug}/)")
    parser.add_argument("--no-transcript", action="store_true",
                        help="Train on title + description only (faster, lower recall)")
    parser.add_argument("--min-examples", type=int, default=5,
                        help="Drop classes with fewer than N examples (default: 5)")
    args = parser.parse_args()

    cache_dir = args.cache_dir or _default_cache_dir()

    # Resolve channel slug from name
    channel_slug = ""
    if args.channel:
        import yaml
        config_path = Path(__file__).parent.parent / "configs" / "channels.yaml"
        with open(config_path) as f:
            from yt_pl_ctr.models import Config
            cfg = Config.model_validate(yaml.safe_load(f))
        matches = [c for c in cfg.channels if c.name.lower() == args.channel.lower()]
        if not matches:
            names = [c.name for c in cfg.channels]
            print(f"Channel {args.channel!r} not found. Available: {names}", file=sys.stderr)
            return 1
        channel_slug = matches[0].slug

    train(
        cache_dir=cache_dir,
        channel_slug=channel_slug,
        model_dir=args.model_dir,
        use_transcript=not args.no_transcript,
        min_examples=args.min_examples,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
