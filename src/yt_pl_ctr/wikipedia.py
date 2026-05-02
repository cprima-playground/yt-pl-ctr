"""Wikipedia lookup for guest information and topic derivation."""

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache

import wikipediaapi

logger = logging.getLogger(__name__)

# User agent required by Wikipedia API
USER_AGENT = "yt-pl-ctr/0.1.0 (https://github.com/cprima-playground/yt-pl-ctr)"


@dataclass
class WikipediaInfo:
    """Information extracted from Wikipedia about a person."""

    name: str
    found: bool = False
    summary: str = ""
    categories: list[str] = field(default_factory=list)
    url: str = ""

    # Derived topics based on categories and summary
    topics: list[str] = field(default_factory=list)


# Topic detection patterns - maps Wikipedia category/summary keywords to our topics.
# Only map to categories that actually exist in channels.yaml.
# Broad terms like "historian", "actor", "doctor" are intentionally excluded
# to avoid false positives from Wikipedia disambiguation to wrong people.
TOPIC_PATTERNS: dict[str, list[str]] = {
    "ufo_extraterrestrial": [
        r"ufolog",
        r"UFO researcher",
        r"UAP",
        r"extraterrestrial",
        r"paranormal investigator.*UFO",
        r"Area 51",
        r"disclosure.*alien",
    ],
    "paranormal_cryptozoology": [
        r"cryptozoolog",
        r"Bigfoot",
        r"Sasquatch",
        r"paranormal investigator",
        r"ghost hunter",
        r"cryptid",
    ],
    "ancient_history": [
        r"archaeolog",
        r"egyptolog",
        r"alternative history",
        r"pseudoarchaeolog",
        r"ancient civilization",
        r"geolog.*alternative",
        r"alternative archaeolog",
    ],
    "psychology": [
        r"psychologist",
        r"psychiatrist",
        r"neuroscientist",
        r"therapist",
        r"mental health",
        r"cognitive scientist",
        r"behavioral scientist",
    ],
}


def _create_wiki() -> wikipediaapi.Wikipedia:
    """Create Wikipedia API client."""
    return wikipediaapi.Wikipedia(
        user_agent=USER_AGENT,
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )


def _extract_topics(summary: str, categories: list[str]) -> list[str]:
    """
    Extract topics from Wikipedia summary and categories.

    Returns list of matching topic keys.
    """
    topics = []
    text = f"{summary} {' '.join(categories)}".lower()

    for topic, patterns in TOPIC_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if topic not in topics:
                    topics.append(topic)
                    logger.debug("Topic '%s' matched pattern '%s'", topic, pattern)
                break  # One match per topic is enough

    return topics


@lru_cache(maxsize=500)
def lookup_person(name: str) -> WikipediaInfo:
    """
    Look up a person on Wikipedia and extract relevant information.

    Uses LRU cache to avoid repeated API calls for the same person.

    Args:
        name: Person's name to search for

    Returns:
        WikipediaInfo with extracted data
    """
    if not name or len(name) < 2:
        return WikipediaInfo(name=name, found=False)

    logger.debug("Looking up '%s' on Wikipedia", name)

    try:
        wiki = _create_wiki()
        page = wiki.page(name)

        if not page.exists():
            # Try with disambiguation - sometimes names need context
            logger.debug("Page not found for '%s', trying search", name)
            return WikipediaInfo(name=name, found=False)

        # Get categories (filter out maintenance categories)
        categories = [
            cat.replace("Category:", "")
            for cat in page.categories
            if not any(
                x in cat.lower()
                for x in ["stub", "articles", "pages", "wikidata", "short description"]
            )
        ][:20]  # Limit to 20 most relevant

        # Get summary (first ~500 chars)
        summary = page.summary[:1000] if page.summary else ""

        # Extract topics from content
        topics = _extract_topics(summary, categories)

        info = WikipediaInfo(
            name=name,
            found=True,
            summary=summary[:500],
            categories=categories,
            url=page.fullurl,
            topics=topics,
        )

        logger.info(
            "Wikipedia: '%s' -> topics=%s",
            name,
            topics or ["none detected"],
        )

        return info

    except Exception as e:
        logger.warning("Wikipedia lookup failed for '%s': %s", name, e)
        return WikipediaInfo(name=name, found=False)


def get_primary_topic(name: str, default: str = "other") -> str:
    """
    Get the primary topic for a person based on Wikipedia.

    Args:
        name: Person's name
        default: Default topic if not found

    Returns:
        Primary topic key
    """
    info = lookup_person(name)
    if info.found and info.topics:
        return info.topics[0]
    return default


def clear_cache() -> None:
    """Clear the Wikipedia lookup cache."""
    lookup_person.cache_clear()
