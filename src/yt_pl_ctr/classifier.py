"""Video classification using a trained ML model (TF-IDF + logistic regression).

Classification is content-first: title + description + transcript of the specific
episode drives the decision. Guest identity is not used — the same guest appearing
in multiple episodes may have different primary topics depending on the conversation.

The model is loaded from $YT_CACHE_DIR/model/ at instantiation time. If no model
is found, all episodes are skipped (no playlist assignment).
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .models import ChannelConfig, VideoMetadata

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of video classification."""

    category_key: str
    category_name: str
    match_reason: str          # "ml_model" | "no_model" | "low_confidence" | "unknown_category"
    matched_value: str | None = None   # confidence score as string
    skipped: bool = False


class VideoClassifier:
    """Content-first video classifier backed by a trained ML model.

    The model is trained by scripts/train_classifier.py and stored in
    $YT_CACHE_DIR/model/pipeline.pkl + label_encoder.pkl.
    """

    def __init__(
        self,
        channel_config: ChannelConfig,
        use_transcripts: bool = True,
        min_confidence: float | None = None,
    ):
        self.config = channel_config
        self.use_transcripts = use_transcripts
        self.min_confidence = min_confidence if min_confidence is not None else channel_config.ml_confidence_threshold
        self._pipeline = None
        self._label_encoder = None
        self._load_model()

    def _load_model(self) -> None:
        # Look in repo models/{channel_slug}/ then models/ then cache fallback
        repo_root = Path(__file__).parents[2]
        channel_slug = self.config.slug
        cache_env = os.environ.get("YT_CACHE_DIR")
        cache_model_dir = Path(cache_env) / "model" if cache_env else None

        candidates = [
            repo_root / "models" / channel_slug,
            repo_root / "models",
            cache_model_dir,
        ]
        model_dir = None
        for candidate in candidates:
            if candidate and (candidate / "pipeline.pkl").exists():
                model_dir = candidate
                break

        if model_dir is None:
            logger.warning("No trained model found — run train_classifier.py first")
            return

        pipeline_path = model_dir / "pipeline.pkl"
        le_path = model_dir / "label_encoder.pkl"
        if not pipeline_path.exists() or not le_path.exists():
            logger.warning("Incomplete model at %s — run train_classifier.py first", model_dir)
            return
        try:
            import joblib
            self._pipeline = joblib.load(pipeline_path)
            self._label_encoder = joblib.load(le_path)
            logger.info(
                "ML model loaded: %d classes (%s)",
                len(self._label_encoder.classes_),
                ", ".join(self._label_encoder.classes_),
            )
        except Exception as e:
            logger.error("Failed to load ML model: %s", e)

    def _build_features(self, video: VideoMetadata) -> str:
        """Replicate the feature string used during training (title ×5 + desc + transcript)."""
        title = video.title or ""
        desc = (video.description or "").split("\n")[0]
        transcript = ""
        if self.use_transcripts:
            transcript = self._load_transcript(video.video_id) or ""
        combined = (title + " ") * 5
        if desc:
            combined += desc + " "
        if transcript:
            combined += transcript
        return combined.strip()

    def _load_transcript(self, video_id: str) -> str | None:
        cache_dir = os.environ.get("YT_CACHE_DIR")
        if cache_dir:
            f = Path(cache_dir) / "episodes" / video_id / "transcript.txt"
            if f.exists():
                return f.read_text(encoding="utf-8")
        return None

    def classify(self, video: VideoMetadata) -> ClassificationResult:
        if self._pipeline is None:
            return ClassificationResult(
                category_key="other",
                category_name="Other",
                match_reason="no_model",
                skipped=True,
            )

        features = self._build_features(video)
        proba = self._pipeline.predict_proba([features])[0]
        best_idx = int(proba.argmax())
        confidence = float(proba[best_idx])
        category_key = self._label_encoder.classes_[best_idx]

        if confidence < self.min_confidence:
            logger.debug(
                "Video '%s' — low confidence %.2f for '%s', skipping",
                video.title, confidence, category_key,
            )
            return ClassificationResult(
                category_key="other",
                category_name="Other",
                match_reason="low_confidence",
                matched_value=f"{category_key}@{confidence:.2f}",
                skipped=True,
            )

        known_slugs = set(self.config.all_leaf_slugs())
        if category_key not in known_slugs:
            logger.debug(
                "Video '%s' — model predicted '%s' not in current taxonomy, skipping",
                video.title, category_key,
            )
            return ClassificationResult(
                category_key="other",
                category_name="Other",
                match_reason="unknown_category",
                matched_value=category_key,
                skipped=True,
            )

        # Skip if this topic is not configured as a playlist
        if category_key not in self.config.playlists:
            logger.debug(
                "Video '%s' -> %s (no playlist configured, skipping)",
                video.title, category_key,
            )
            return ClassificationResult(
                category_key=category_key,
                category_name=category_key.replace("_", " ").title(),
                match_reason="ml_model",
                matched_value=f"{confidence:.2f}",
                skipped=True,
            )

        logger.debug(
            "Video '%s' -> %s (ml_model, conf=%.2f)",
            video.title, category_key, confidence,
        )
        return ClassificationResult(
            category_key=category_key,
            category_name=category_key.replace("_", " ").title(),
            match_reason="ml_model",
            matched_value=f"{confidence:.2f}",
        )

    def get_playlist_name(self, slug: str) -> str:
        title = self.config.playlist_title(slug)
        if title:
            return title
        name = slug.replace("_", " ").title()
        if self.config.playlist_prefix:
            return f"{self.config.playlist_prefix} – {name}"
        return name
