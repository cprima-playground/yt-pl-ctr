"""Video metadata fetching using yt-dlp."""

import logging
import time
from collections.abc import Iterator
from functools import wraps

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .models import VideoMetadata

logger = logging.getLogger(__name__)

# Retry settings
MAX_RETRIES = 10
BASE_DELAY = 10.0  # seconds
MAX_DELAY = 3600.0  # 1 hour max


def retry_on_error(func):
    """Decorator to retry yt-dlp calls on transient errors with exponential backoff."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (DownloadError, Exception) as e:
                error_str = str(e).lower()
                # Retry on rate limits and transient errors
                if any(x in error_str for x in ["429", "rate", "too many", "temporarily"]):
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    logger.warning(
                        "yt-dlp error (attempt %d/%d), waiting %.1fs: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        str(e)[:100],
                    )
                    time.sleep(delay)
                    last_error = e
                else:
                    raise
        raise last_error

    return wrapper


def _create_ydl(flat: bool = False) -> YoutubeDL:
    """Create a YoutubeDL instance with standard options."""
    import os

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": flat,
        "ignoreerrors": True,
        "sleep_interval": 2,  # Wait between requests
        "max_sleep_interval": 5,
    }

    # Use cookies from browser if available
    cookies_from = os.environ.get("YT_COOKIES_FROM")  # e.g., "firefox", "chrome"
    cookies_file = os.environ.get("YT_COOKIES_FILE")  # e.g., "cookies.txt"

    if cookies_from:
        opts["cookiesfrombrowser"] = (cookies_from,)
    elif cookies_file:
        opts["cookiefile"] = cookies_file

    return YoutubeDL(opts)


def fetch_channel_videos(
    url: str, limit: int = 30, offset: int = 0
) -> Iterator[VideoMetadata]:
    """
    Fetch video metadata from a channel URL.

    Args:
        url: yt-dlp compatible URL (channel, playlist, etc.)
        limit: Maximum number of videos to fetch
        offset: Number of videos to skip (for pagination)

    Yields:
        VideoMetadata for each video
    """
    start = offset + 1  # yt-dlp uses 1-based indexing
    end = offset + limit
    logger.info("Fetching videos %d-%d from: %s", start, end, url)

    # First, get video IDs with flat extraction (faster)
    with _create_ydl(flat=True) as ydl:
        ydl.params["playliststart"] = start
        ydl.params["playlistend"] = end
        info = ydl.extract_info(url, download=False)

    if not info:
        logger.warning("No info returned for URL: %s", url)
        return

    entries = info.get("entries") or []
    logger.info("Found %d videos", len(entries))

    # Fetch full metadata for each video
    for entry in entries[:limit]:
        video_id = entry.get("id")
        if not video_id:
            continue

        try:
            metadata = fetch_video_metadata(video_id)
            if metadata:
                yield metadata
        except Exception as e:
            logger.error("Error fetching metadata for %s: %s", video_id, e)


@retry_on_error
def fetch_video_metadata(video_id: str) -> VideoMetadata | None:
    """
    Fetch full metadata for a single video.

    Args:
        video_id: YouTube video ID

    Returns:
        VideoMetadata or None if fetch failed
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    with _create_ydl(flat=False) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:
            error_str = str(e).lower()
            if any(x in error_str for x in ["429", "rate", "too many", "temporarily"]):
                raise  # Let retry decorator handle it
            logger.error("Failed to fetch video %s: %s", video_id, e)
            return None

    if not info:
        return None

    return VideoMetadata(
        video_id=info.get("id", video_id),
        title=info.get("title", ""),
        description=info.get("description", ""),
        channel_name=info.get("channel", ""),
        channel_id=info.get("channel_id", ""),
        upload_date=info.get("upload_date"),
        duration=info.get("duration"),
        view_count=info.get("view_count"),
    )
