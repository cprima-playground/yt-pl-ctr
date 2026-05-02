"""Video metadata fetching — adapter pattern over yt-dlp or YouTube Data API."""

import json
import logging
import os
import re
import time
import urllib.request
from collections.abc import Iterator
from functools import wraps
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .models import VideoMetadata

if TYPE_CHECKING:
    from .youtube import YouTubeClient

logger = logging.getLogger(__name__)


def is_ci() -> bool:
    """True on GitHub Actions (GITHUB_ACTIONS=true). yt-dlp is bot-blocked on CI IPs."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


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
                    delay = min(_YTDLP_BASE_DELAY * (2**attempt), _YTDLP_MAX_DELAY)
                    logger.warning(
                        "yt-dlp error (attempt %d/%d), waiting %.1fs: %s",
                        attempt + 1,
                        _YTDLP_MAX_RETRIES,
                        delay,
                        str(e)[:100],
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

        # Page through the uploads playlist, skipping full pages for large offsets
        skipped = 0
        fetched = 0
        page_token: str | None = None

        while fetched < limit:
            video_ids, page_token = self._client.list_playlist_videos(
                uploads_playlist_id, max_results=50, page_token=page_token
            )

            if not video_ids:
                break

            # Apply offset by skipping whole pages when possible
            if skipped + len(video_ids) <= offset:
                skipped += len(video_ids)
                if not page_token:
                    break
                continue

            # Filter to IDs we actually want (after offset, up to limit)
            ids_to_fetch = []
            for vid_id in video_ids:
                if skipped < offset:
                    skipped += 1
                    continue
                if fetched + len(ids_to_fetch) >= limit:
                    break
                ids_to_fetch.append(vid_id)

            # Batch fetch metadata — 1 API call for up to 50 videos
            if ids_to_fetch:
                batch = self._client.get_videos_metadata(ids_to_fetch)
                for video in batch:
                    yield video
                    fetched += 1

            if not page_token or fetched >= limit:
                break

        logger.info("YouTubeAPI: fetched %d videos", fetched)

    def fetch_video_metadata(self, video_id: str) -> VideoMetadata | None:
        results = self._client.get_videos_metadata([video_id])
        return results[0] if results else None


# ── Module-level shims (backward compat) ─────────────────────────────────────


def fetch_channel_videos(url: str, limit: int = 30, offset: int = 0) -> Iterator[VideoMetadata]:
    """Backward-compat shim using YtDlpFetcher."""
    yield from YtDlpFetcher().fetch_channel_videos(url, limit=limit, offset=offset)


def fetch_video_metadata(video_id: str) -> VideoMetadata | None:
    """Backward-compat shim using YtDlpFetcher."""
    return YtDlpFetcher().fetch_video_metadata(video_id)


# ── Transcript fetching ───────────────────────────────────────────────────────

_TRANSCRIPT_MAX_CHARS = 10000  # runtime classifier: fast, first ~13 min
_TRANSCRIPT_CORPUS_MAX_CHARS = 30000  # corpus building: first ~40 min, past most intros
_TRANSCRIPT_SKIP_SECONDS = 120  # skip first 2 min (intro music, credits)
_TRANSCRIPT_LANGS = ("en", "en-US", "en-GB")


def fetch_transcript(
    video_id: str,
    max_chars: int = _TRANSCRIPT_MAX_CHARS,
    skip_seconds: int = _TRANSCRIPT_SKIP_SECONDS,
) -> str | None:
    """Fetch auto-generated transcript via yt-dlp. Returns None on any failure.

    Skips immediately on CI (yt-dlp is bot-blocked on GitHub Actions IPs).
    skip_seconds: drop captions from the first N seconds (intro music, sponsors).
    max_chars: truncate result for runtime use; use _TRANSCRIPT_CORPUS_MAX_CHARS for training.
    """
    if is_ci():
        logger.debug("Skipping transcript fetch on CI for %s", video_id)
        return None
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with _create_ydl(flat=False) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info:
            return None

        auto_caps: dict = info.get("automatic_captions") or {}
        manual_caps: dict = info.get("subtitles") or {}

        for lang in _TRANSCRIPT_LANGS:
            formats = auto_caps.get(lang) or manual_caps.get(lang)
            if not formats:
                continue
            for ext in ("json3", "vtt"):
                fmt = next((f for f in formats if f.get("ext") == ext), None)
                if fmt and fmt.get("url"):
                    text = _fetch_caption_url(
                        fmt["url"], ext, max_chars=max_chars, skip_seconds=skip_seconds
                    )
                    if text:
                        logger.debug(
                            "Transcript fetched for %s (%s %s, %d chars)",
                            video_id,
                            lang,
                            ext,
                            len(text),
                        )
                        return text

        logger.debug("No transcript found for %s", video_id)
        return None

    except Exception as e:
        logger.debug("Transcript fetch failed for %s: %s", video_id, str(e)[:120])
        return None


def _parse_vtt_timestamp(ts: str) -> float:
    """Parse HH:MM:SS.mmm or MM:SS.mmm VTT timestamp to seconds."""
    parts = ts.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return int(parts[0]) * 60 + float(parts[1])
    except (ValueError, IndexError):
        return 0.0


def _fetch_caption_url(
    url: str,
    ext: str,
    max_chars: int | None = _TRANSCRIPT_MAX_CHARS,
    skip_seconds: int = _TRANSCRIPT_SKIP_SECONDS,
) -> str | None:
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8")
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2**attempt)
    else:
        logger.debug("Caption URL fetch failed after 3 attempts: %s", last_err)
        return None

    skip_ms = skip_seconds * 1000

    if ext == "json3":
        data = json.loads(content)
        parts = []
        for event in data.get("events", []):
            if event.get("tStartMs", 0) < skip_ms:
                continue
            for seg in event.get("segs", []):
                text = seg.get("utf8", "").strip()
                if text and text != "\n":
                    parts.append(text)
        text = " ".join(parts)
    else:
        # VTT: parse timestamps to apply skip_seconds
        lines = []
        skip_block = False
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("WEBVTT") or re.match(r"^\d+$", line):
                continue
            if "-->" in line:
                start_ts = line.split("-->")[0].strip()
                skip_block = _parse_vtt_timestamp(start_ts) < skip_seconds
                continue
            if not skip_block:
                cleaned = re.sub(r"<[^>]+>", "", line)
                if cleaned:
                    lines.append(cleaned)
        text = " ".join(lines)

    text = " ".join(text.split())
    if not text:
        return None
    return text[:max_chars] if max_chars is not None else text
