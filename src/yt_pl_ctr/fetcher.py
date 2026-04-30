"""Video metadata fetching — adapter pattern over yt-dlp or YouTube Data API."""

import logging
import os
import time
from collections.abc import Iterator
from functools import wraps
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .models import VideoMetadata

if TYPE_CHECKING:
    from .youtube import YouTubeClient

logger = logging.getLogger(__name__)


# ── Protocol ─────────────────────────────────────────────────────────────────

@runtime_checkable
class VideoFetcherProtocol(Protocol):
    """Interface for fetching video metadata from a channel."""

    def fetch_channel_videos(
        self, url: str, limit: int = 30, offset: int = 0
    ) -> Iterator[VideoMetadata]: ...

    def fetch_video_metadata(self, video_id: str) -> VideoMetadata | None: ...


# ── yt-dlp adapter ───────────────────────────────────────────────────────────

_YTDLP_MAX_RETRIES = 10
_YTDLP_BASE_DELAY = 10.0
_YTDLP_MAX_DELAY = 3600.0


def _ytdlp_retry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(_YTDLP_MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if any(x in str(e).lower() for x in ["429", "rate", "too many", "temporarily"]):
                    delay = min(_YTDLP_BASE_DELAY * (2 ** attempt), _YTDLP_MAX_DELAY)
                    logger.warning(
                        "yt-dlp error (attempt %d/%d), waiting %.1fs: %s",
                        attempt + 1, _YTDLP_MAX_RETRIES, delay, str(e)[:100],
                    )
                    time.sleep(delay)
                    last_error = e
                else:
                    raise
        raise last_error
    return wrapper


def _create_ydl(flat: bool = False):
    from yt_dlp import YoutubeDL

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": flat,
        "ignoreerrors": True,
        "sleep_interval": 2,
        "max_sleep_interval": 5,
    }
    cookies_from = os.environ.get("YT_COOKIES_FROM")
    cookies_file = os.environ.get("YT_COOKIES_FILE")
    if cookies_from:
        opts["cookiesfrombrowser"] = (cookies_from,)
    elif cookies_file:
        opts["cookiefile"] = cookies_file
    return YoutubeDL(opts)


class YtDlpFetcher:
    """Fetches video metadata using yt-dlp. Works on residential IPs; blocked in CI."""

    def fetch_channel_videos(
        self, url: str, limit: int = 30, offset: int = 0
    ) -> Iterator[VideoMetadata]:
        start = offset + 1
        end = offset + limit
        logger.info("yt-dlp: fetching videos %d-%d from %s", start, end, url)

        with _create_ydl(flat=True) as ydl:
            ydl.params["playliststart"] = start
            ydl.params["playlistend"] = end
            info = ydl.extract_info(url, download=False)

        if not info:
            logger.warning("yt-dlp: no info for %s", url)
            return

        entries = info.get("entries") or []
        logger.info("yt-dlp: found %d videos", len(entries))

        for entry in entries[:limit]:
            video_id = entry.get("id")
            if not video_id:
                continue
            try:
                metadata = self.fetch_video_metadata(video_id)
                if metadata:
                    yield metadata
            except Exception as e:
                logger.error("yt-dlp: error fetching %s: %s", video_id, e)

    @_ytdlp_retry
    def fetch_video_metadata(self, video_id: str) -> VideoMetadata | None:
        url = f"https://www.youtube.com/watch?v={video_id}"
        with _create_ydl(flat=False) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
            except Exception as e:
                if any(x in str(e).lower() for x in ["429", "rate", "too many", "temporarily"]):
                    raise
                logger.error("yt-dlp: failed to fetch %s: %s", video_id, e)
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


# ── YouTube Data API adapter ──────────────────────────────────────────────────

class YouTubeAPIFetcher:
    """Fetches video metadata via YouTube Data API v3. No bot detection issues."""

    def __init__(self, client: "YouTubeClient") -> None:
        self._client = client

    def fetch_channel_videos(
        self, url: str, limit: int = 30, offset: int = 0
    ) -> Iterator[VideoMetadata]:
        logger.info("YouTubeAPI: fetching %d videos (offset=%d) from %s", limit, offset, url)

        channel_id = self._client.resolve_channel_id(url)
        uploads_playlist_id = self._client.get_uploads_playlist_id(channel_id)

        # Page through the uploads playlist, skipping `offset` videos
        skipped = 0
        fetched = 0
        page_token: str | None = None

        while fetched < limit:
            batch_size = min(50, limit - fetched + max(0, offset - skipped))
            video_ids, page_token = self._client.list_playlist_videos(
                uploads_playlist_id, max_results=batch_size, page_token=page_token
            )

            if not video_ids:
                break

            for vid_id in video_ids:
                if skipped < offset:
                    skipped += 1
                    continue
                if fetched >= limit:
                    break

                videos = self._client.get_videos_metadata([vid_id])
                if videos:
                    yield videos[0]
                    fetched += 1

            if not page_token:
                break

        logger.info("YouTubeAPI: fetched %d videos", fetched)

    def fetch_video_metadata(self, video_id: str) -> VideoMetadata | None:
        results = self._client.get_videos_metadata([video_id])
        return results[0] if results else None


# ── Module-level shims (backward compat) ─────────────────────────────────────

def fetch_channel_videos(
    url: str, limit: int = 30, offset: int = 0
) -> Iterator[VideoMetadata]:
    """Backward-compat shim using YtDlpFetcher."""
    yield from YtDlpFetcher().fetch_channel_videos(url, limit=limit, offset=offset)


def fetch_video_metadata(video_id: str) -> VideoMetadata | None:
    """Backward-compat shim using YtDlpFetcher."""
    return YtDlpFetcher().fetch_video_metadata(video_id)
