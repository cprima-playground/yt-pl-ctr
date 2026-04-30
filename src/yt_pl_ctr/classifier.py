"""Video classification logic with per-channel category mappings and Wikipedia lookup."""

import logging
import re
from dataclasses import dataclass

from .models import Category, ChannelConfig, VideoMetadata
from .wikipedia import WikipediaInfo, lookup_person

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of video classification."""

    category_key: str
    category_name: str
    match_reason: str  # "priority_keyword", "guest", "wikipedia", "title_keyword", "description_keyword", "description_pattern", "default"
    matched_value: str | None = None
    wikipedia_info: WikipediaInfo | None = None
    skipped: bool = False  # True if this video should be skipped (default category + skip_default)


class VideoClassifier:
    """Classifier for videos based on channel-specific category mappings."""

    def __init__(self, channel_config: ChannelConfig, use_wikipedia: bool = True):
        """
        Initialize classifier with channel configuration.

        Args:
            channel_config: Channel-specific configuration with categories
            use_wikipedia: Whether to use Wikipedia for classification
        """
        self.config = channel_config
        self.use_wikipedia = use_wikipedia
        self._guest_pattern = re.compile(channel_config.guest_pattern)
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self._description_patterns: dict[str, list[re.Pattern]] = {}
        for key, category in self.config.categories.items():
            self._description_patterns[key] = [
                re.compile(p, re.IGNORECASE) for p in category.description_patterns
            ]

    def extract_guest(self, title: str) -> str | None:
        """Extract guest name from video title using channel's pattern."""
        match = self._guest_pattern.search(title or "")
        return match.group(1).strip() if match else None

    def _classify_by_wikipedia(self, guest: str) -> ClassificationResult | None:
        """
        Try to classify based on Wikipedia lookup.

        Args:
            guest: Guest name to look up

        Returns:
            ClassificationResult if a matching topic is found, None otherwise
        """
        if not self.use_wikipedia or not guest:
            return None

        # Handle multiple guests - try each one
        guest_names = [g.strip() for g in re.split(r"\s*[&,]\s*", guest)]

        for name in guest_names:
            if len(name) < 3:
                continue

            info = lookup_person(name)
            if not info.found or not info.topics:
                continue

            # Check if any Wikipedia topic matches our categories
            for topic in info.topics:
                if topic in self.config.categories:
                    category = self.config.categories[topic]
                    logger.info(
                        "Wikipedia: '%s' -> %s (from Wikipedia: %s)",
                        name,
                        topic,
                        info.url,
                    )
                    return ClassificationResult(
                        category_key=topic,
                        category_name=category.name,
                        match_reason="wikipedia",
                        matched_value=f"{name} ({', '.join(info.topics[:3])})",
                        wikipedia_info=info,
                    )

        return None

    def classify(self, video: VideoMetadata) -> ClassificationResult:
        """
        Classify a video into a category.

        Classification priority:
        1. Priority keywords (categories with priority set)
        2. Guest name match (config)
        3. Wikipedia lookup (by guest name)
        4. Title keyword match
        5. Description pattern match (regex)
        6. Description keyword match
        7. Default category

        Args:
            video: Video metadata to classify

        Returns:
            ClassificationResult with category and match reason
        """
        guest = self.extract_guest(video.title)

        # Use only the first paragraph of description (before any newline)
        # This avoids matching sponsor text in subsequent lines
        description = video.description.split("\n")[0] if video.description else ""
        title_lower = video.title.lower()
        desc_lower = description.lower()

        # Priority 1: Check priority categories first (e.g., ancient_history with priority: 1)
        priority_categories = [
            (key, cat) for key, cat in self.config.categories.items()
            if cat.priority is not None
        ]
        priority_categories.sort(key=lambda x: x[1].priority or 999)

        for key, category in priority_categories:
            for keyword in category.keywords:
                kw_lower = keyword.lower()
                if kw_lower in title_lower or kw_lower in desc_lower:
                    logger.debug(
                        "Video '%s' -> %s (priority keyword: %s)",
                        video.title,
                        key,
                        keyword,
                    )
                    return ClassificationResult(
                        category_key=key,
                        category_name=category.name,
                        match_reason="priority_keyword",
                        matched_value=keyword,
                    )

        # Priority 2: Guest name match from config (supports multiple guests)
        if guest:
            guest_lower = guest.lower()
            for key, category in self.config.categories.items():
                for known_guest in category.guests:
                    known_lower = known_guest.lower()
                    if known_lower == guest_lower or known_lower in guest_lower:
                        logger.debug(
                            "Video '%s' -> %s (guest: %s matched %s)",
                            video.title,
                            key,
                            known_guest,
                            guest,
                        )
                        return ClassificationResult(
                            category_key=key,
                            category_name=category.name,
                            match_reason="guest",
                            matched_value=known_guest,
                        )

        # Priority 3: Wikipedia lookup
        if guest and self.use_wikipedia:
            wiki_result = self._classify_by_wikipedia(guest)
            if wiki_result:
                return wiki_result

        # Priority 3: Title keyword match (skip priority categories already checked)
        priority_keys = {key for key, cat in self.config.categories.items() if cat.priority is not None}
        for key, category in self.config.categories.items():
            if key in priority_keys:
                continue
            for keyword in category.keywords:
                if keyword.lower() in title_lower:
                    logger.debug(
                        "Video '%s' -> %s (title keyword: %s)",
                        video.title,
                        key,
                        keyword,
                    )
                    return ClassificationResult(
                        category_key=key,
                        category_name=category.name,
                        match_reason="title_keyword",
                        matched_value=keyword,
                    )

        # Priority 4: Description pattern match (regex)
        for key, patterns in self._description_patterns.items():
            for pattern in patterns:
                if pattern.search(description):
                    category = self.config.categories[key]
                    logger.debug(
                        "Video '%s' -> %s (description pattern)",
                        video.title,
                        key,
                    )
                    return ClassificationResult(
                        category_key=key,
                        category_name=category.name,
                        match_reason="description_pattern",
                        matched_value=pattern.pattern,
                    )

        # Priority 5: Description keyword match (skip priority categories already checked)
        for key, category in self.config.categories.items():
            if key in priority_keys:
                continue
            for keyword in category.keywords:
                if keyword.lower() in desc_lower:
                    logger.debug(
                        "Video '%s' -> %s (description keyword: %s)",
                        video.title,
                        key,
                        keyword,
                    )
                    return ClassificationResult(
                        category_key=key,
                        category_name=category.name,
                        match_reason="description_keyword",
                        matched_value=keyword,
                    )

        # Priority 6: Default category
        logger.debug("Video '%s' -> %s (default)", video.title, self.config.default_category)
        result = ClassificationResult(
            category_key=self.config.default_category,
            category_name=self.config.default_category.replace("_", " ").title(),
            match_reason="default",
        )

        # Mark as skipped if skip_default is enabled
        if self.config.skip_default:
            result.skipped = True

        return result

    def get_playlist_name(self, category_key: str) -> str:
        """
        Get the full playlist name for a category.

        Args:
            category_key: Category key

        Returns:
            Full playlist name with prefix
        """
        category = self.config.categories.get(category_key)
        if category:
            name = category.name
        else:
            name = category_key.replace("_", " ").title()

        if self.config.playlist_prefix:
            return f"{self.config.playlist_prefix} – {name}"
        return name
