"""YouTube Data API client for playlist management and video fetching."""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from functools import wraps

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .models import VideoMetadata

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube"]

# Retry settings for rate limits
MAX_RETRIES = 5
BASE_DELAY = 60.0  # seconds (YouTube rate limits need longer waits)
MAX_DELAY = 900.0  # 15 minutes max

# Minimum delay between API calls (seconds)
API_CALL_DELAY = 5.0
_last_api_call = 0.0


def _parse_iso8601_duration(duration: str) -> int:
    """Convert ISO 8601 duration to seconds. e.g. PT2H30M15S → 9015."""
    pattern = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
    m = pattern.match(duration or "")
    if not m:
        return 0
    hours, minutes, seconds = (int(x or 0) for x in m.groups())
    return hours * 3600 + minutes * 60 + seconds


def _parse_upload_date(published_at: str) -> str:
    """Convert ISO 8601 timestamp to YYYYMMDD. e.g. 2024-01-15T10:30:00Z → 20240115."""
    return published_at[:10].replace("-", "") if published_at else ""


def _rate_limit_delay():
    """Ensure minimum delay between API calls."""
    global _last_api_call
    now = time.time()
    elapsed = now - _last_api_call
    if elapsed < API_CALL_DELAY:
        sleep_time = API_CALL_DELAY - elapsed
        logger.debug("Rate limit delay: %.2fs", sleep_time)
        time.sleep(sleep_time)
    _last_api_call = time.time()


class YouTubeAPIError(Exception):
    """Exception for YouTube API errors."""

    pass


class RateLimitError(YouTubeAPIError):
    """Raised when rate limit retries are exhausted."""

    pass


def retry_on_rate_limit(func):
    """Decorator to retry on 429 errors with exponential backoff up to 24 hours."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except HttpError as e:
                if e.resp.status == 429:
                    delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)
                    hours = delay / 3600
                    if delay >= 3600:
                        logger.warning(
                            "Rate limited (attempt %d/%d), waiting %.1f hours...",
                            attempt + 1,
                            MAX_RETRIES,
                            hours,
                        )
                    else:
                        logger.warning(
                            "Rate limited (attempt %d/%d), waiting %.0f seconds...",
                            attempt + 1,
                            MAX_RETRIES,
                            delay,
                        )
                    time.sleep(delay)
                    last_error = e
                else:
                    raise
        raise RateLimitError(f"Rate limit exceeded after {MAX_RETRIES} retries") from last_error

    return wrapper


@dataclass
class YouTubeClient:
    """Client for YouTube Data API operations."""

    _service: object = field(repr=False)
    _playlist_cache: dict[str, str] = field(default_factory=dict, repr=False)
    _added_videos: set[tuple[str, str]] = field(
        default_factory=set, repr=False
    )  # (playlist_id, video_id)
    _channel_id_cache: dict[str, str] = field(default_factory=dict, repr=False)
    _uploads_playlist_cache: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(cls) -> "YouTubeClient":
        """
        Create client from environment variables.

        Required env vars:
            - YT_CLIENT_ID
            - YT_CLIENT_SECRET
            - YT_REFRESH_TOKEN
        """
        required = ["YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"]
        missing = [v for v in required if not os.environ.get(v)]
        if missing:
            raise YouTubeAPIError(f"Missing environment variables: {', '.join(missing)}")

        creds = Credentials(
            token=None,
            refresh_token=os.environ["YT_REFRESH_TOKEN"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.environ["YT_CLIENT_ID"],
            client_secret=os.environ["YT_CLIENT_SECRET"],
            scopes=SCOPES,
        )
        service = build("youtube", "v3", credentials=creds)
        return cls(_service=service)

    def get_channel_info(self) -> dict:
        """Get info about the authenticated channel."""
        _rate_limit_delay()
        logger.debug("API CALL: channels.list")
        response = self._service.channels().list(part="snippet", mine=True).execute()
        items = response.get("items", [])
        if items:
            snippet = items[0]["snippet"]
            return {
                "title": snippet.get("title"),
                "custom_url": snippet.get("customUrl"),
                "id": items[0].get("id"),
            }
        return {}

    @retry_on_rate_limit
    def find_playlist(self, title: str) -> str | None:
        """
        Find a playlist by exact title match.

        Args:
            title: Playlist title to search for

        Returns:
            Playlist ID if found, None otherwise
        """
        # Check cache first
        if title in self._playlist_cache:
            return self._playlist_cache[title]

        _rate_limit_delay()
        logger.info("API CALL: playlists.list")
        request = self._service.playlists().list(part="snippet", mine=True, maxResults=50)
        while request:
            response = request.execute()
            logger.debug("API CALL: playlists.list (next page)")
            for item in response.get("items", []):
                item_title = item["snippet"]["title"]
                self._playlist_cache[item_title] = item["id"]
                if item_title == title:
                    return item["id"]
                # Detect mojibake: UTF-8 bytes decoded as Latin-1 (Windows cp1252 misread)
                try:
                    if item_title.encode("latin-1").decode("utf-8") == title:
                        logger.warning(
                            "Matched mojibake playlist title %r → %r", item_title, title
                        )
                        self._playlist_cache[title] = item["id"]
                        return item["id"]
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            request = self._service.playlists().list_next(request, response)

        return None

    @retry_on_rate_limit
    def create_playlist(self, title: str, description: str = "", privacy: str = "public") -> str:
        """
        Create a new playlist.

        Note: Due to YouTube rate limits on public content, playlists are
        always created as PRIVATE. Set to public manually via YouTube Studio.

        Args:
            title: Playlist title
            description: Playlist description
            privacy: Ignored - always creates private (see note above)

        Returns:
            New playlist ID
        """
        _rate_limit_delay()
        # Always create as private to avoid rate limits on public content
        # YouTube has stricter rate limits for public playlist creation
        logger.info("API CALL: playlists.insert (%s) [as private]", title)
        try:
            response = (
                self._service.playlists()
                .insert(
                    part="snippet,status",
                    body={
                        "snippet": {"title": title, "description": description},
                        "status": {"privacyStatus": "private"},
                    },
                )
                .execute()
            )
            playlist_id = response["id"]
            self._playlist_cache[title] = playlist_id
            logger.info("Created private playlist: %s (%s)", title, playlist_id)
            return playlist_id
        except HttpError as e:
            if e.resp.status == 429:
                raise  # Let decorator handle quota wait
            raise YouTubeAPIError(f"Failed to create playlist '{title}': {e}") from e

    def ensure_playlist(self, title: str, description: str = "", privacy: str = "public") -> str:
        """
        Find existing playlist or create new one.

        Args:
            title: Playlist title
            description: Description for new playlist
            privacy: Privacy status for new playlist

        Returns:
            Playlist ID
        """
        playlist_id = self.find_playlist(title)
        if playlist_id:
            logger.debug("Found existing playlist: %s (%s)", title, playlist_id)
            return playlist_id
        return self.create_playlist(title, description, privacy)

    @retry_on_rate_limit
    def playlist_contains_video(self, playlist_id: str, video_id: str) -> bool:
        """Check if a video is already in a playlist."""
        _rate_limit_delay()
        logger.debug("API CALL: playlistItems.list")
        try:
            request = self._service.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=50,
            )
            while request:
                response = request.execute()
                for item in response.get("items", []):
                    if item["contentDetails"].get("videoId") == video_id:
                        return True
                request = self._service.playlistItems().list_next(request, response)
            return False
        except HttpError as e:
            if e.resp.status == 404:
                # Playlist not found or not yet available (eventual consistency)
                logger.debug("Playlist %s not found, assuming empty", playlist_id)
                return False
            raise

    @retry_on_rate_limit
    def add_video_to_playlist(self, playlist_id: str, video_id: str) -> str:
        """Add a video to a playlist. Returns the playlist_item_id."""
        _rate_limit_delay()
        logger.info("API CALL: playlistItems.insert (%s)", video_id)
        try:
            response = (
                self._service.playlistItems()
                .insert(
                    part="snippet",
                    body={
                        "snippet": {
                            "playlistId": playlist_id,
                            "resourceId": {"kind": "youtube#video", "videoId": video_id},
                        }
                    },
                )
                .execute()
            )
            playlist_item_id = response["id"]
            self._added_videos.add((playlist_id, video_id))
            logger.info(
                "Added video %s to playlist %s (item %s)", video_id, playlist_id, playlist_item_id
            )
            return playlist_item_id
        except HttpError as e:
            if e.resp.status == 409:
                raise YouTubeAPIError(f"Video {video_id} already in playlist {playlist_id}") from e
            if e.resp.status == 429:
                raise
            raise YouTubeAPIError(f"Failed to add video {video_id} to playlist: {e}") from e

    def add_video_if_missing(self, playlist_id: str, video_id: str) -> str | None:
        """Add video to playlist only if not already present.

        Returns the playlist_item_id if added, None if already present.
        """
        if (playlist_id, video_id) in self._added_videos:
            logger.debug("Video %s already added in this session, skipping", video_id)
            return None

        if self.playlist_contains_video(playlist_id, video_id):
            logger.debug("Video %s already in playlist, skipping", video_id)
            self._added_videos.add((playlist_id, video_id))
            return None

        return self.add_video_to_playlist(playlist_id, video_id)

    @retry_on_rate_limit
    def get_playlist_contents(self, playlist_id: str) -> dict[str, str]:
        """Fetch all items in a playlist. Returns {video_id: playlist_item_id}.

        playlist_item_id (not video_id) is required by playlistItems.delete.
        """
        result: dict[str, str] = {}
        page_token: str | None = None
        while True:
            _rate_limit_delay()
            kwargs: dict = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
            if page_token:
                kwargs["pageToken"] = page_token
            try:
                response = self._service.playlistItems().list(**kwargs).execute()
            except HttpError as e:
                if e.resp.status == 404:
                    return result
                raise
            for item in response.get("items", []):
                vid = item["contentDetails"].get("videoId")
                if vid:
                    result[vid] = item["id"]  # item["id"] is the playlist_item_id
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return result

    @retry_on_rate_limit
    def remove_playlist_item(self, playlist_item_id: str) -> None:
        """Remove a video from a playlist by playlist_item_id.

        Uses playlistItems.delete — requires the item ID, not the video ID.
        Silently ignores 404 (item already removed or never existed).
        """
        _rate_limit_delay()
        logger.info("API CALL: playlistItems.delete (%s)", playlist_item_id)
        try:
            self._service.playlistItems().delete(id=playlist_item_id).execute()
        except HttpError as e:
            if e.resp.status == 404:
                logger.debug("Playlist item %s already gone", playlist_item_id)
                return
            if e.resp.status == 429:
                raise
            raise YouTubeAPIError(f"Failed to remove playlist item {playlist_item_id}: {e}") from e

    # ── Playlist membership ───────────────────────────────────────────────────

    @retry_on_rate_limit
    def list_my_playlists(self) -> list[dict]:
        """Return all playlists owned by the authenticated user.

        Each entry: {id, title, item_count}.
        Costs 1 quota unit per page (50 playlists per page).
        """
        results = []
        request = self._service.playlists().list(
            part="snippet,contentDetails", mine=True, maxResults=50
        )
        while request:
            _rate_limit_delay()
            logger.debug("API CALL: playlists.list")
            response = request.execute()
            for item in response.get("items", []):
                pid = item["id"]
                title = item["snippet"]["title"]
                item_count = item.get("contentDetails", {}).get("itemCount", 0)
                self._playlist_cache[title] = pid
                results.append({"id": pid, "title": title, "item_count": item_count})
            request = self._service.playlists().list_next(request, response)
        return results

    def load_all_membership(self) -> dict:
        """Bulk-fetch membership across all user playlists.

        Returns {"playlists": {id: title}, "membership": {video_id: [playlist_id, ...]}}.
        Quota cost: ~1 unit (playlists.list) + 1 per playlist page of items.
        """
        playlists = self.list_my_playlists()
        playlist_titles = {p["id"]: p["title"] for p in playlists}
        membership: dict[str, list[str]] = {}
        for p in playlists:
            pid = p["id"]
            logger.info("Fetching playlist items: %s (%d videos)", p["title"], p["item_count"])
            contents = self.get_playlist_contents(pid)
            for video_id in contents:
                membership.setdefault(video_id, []).append(pid)
        return {"playlists": playlist_titles, "membership": membership}

    # ── Video fetching ────────────────────────────────────────────────────────

    @retry_on_rate_limit
    def resolve_channel_id(self, url: str) -> str:
        """
        Resolve a channel URL or @handle to a channel ID (UCxxx).

        Costs 1 quota unit for handle lookups; UCxxx IDs are returned as-is.
        Results are cached in-session.
        """
        if url in self._channel_id_cache:
            return self._channel_id_cache[url]

        # Already a channel ID
        if re.match(r"^UC[\w-]{22}$", url):
            self._channel_id_cache[url] = url
            return url

        # Extract handle: @joerogan or https://www.youtube.com/@joerogan/videos
        handle_match = re.search(r"@([\w.-]+)", url)
        if handle_match:
            handle = handle_match.group(1)
            _rate_limit_delay()
            logger.debug("API CALL: channels.list(forHandle=%s)", handle)
            response = self._service.channels().list(part="id", forHandle=handle).execute()
            items = response.get("items", [])
            if not items:
                raise YouTubeAPIError(f"Channel not found for handle: @{handle}")
            channel_id = items[0]["id"]
            self._channel_id_cache[url] = channel_id
            logger.debug("Resolved @%s → %s", handle, channel_id)
            return channel_id

        raise YouTubeAPIError(f"Cannot resolve channel ID from URL: {url}")

    @retry_on_rate_limit
    def get_uploads_playlist_id(self, channel_id: str) -> str:
        """
        Get the uploads playlist ID for a channel. Costs 1 quota unit.
        Results are cached in-session.
        """
        if channel_id in self._uploads_playlist_cache:
            return self._uploads_playlist_cache[channel_id]

        _rate_limit_delay()
        logger.debug("API CALL: channels.list(contentDetails, id=%s)", channel_id)
        response = self._service.channels().list(part="contentDetails", id=channel_id).execute()
        items = response.get("items", [])
        if not items:
            raise YouTubeAPIError(f"Channel not found: {channel_id}")
        uploads_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        self._uploads_playlist_cache[channel_id] = uploads_id
        return uploads_id

    @retry_on_rate_limit
    def list_playlist_videos(
        self, playlist_id: str, max_results: int = 50, page_token: str | None = None
    ) -> tuple[list[str], str | None]:
        """
        List video IDs from a playlist page. Costs 1 quota unit per call.

        Returns:
            (video_ids, next_page_token)
        """
        _rate_limit_delay()
        logger.debug("API CALL: playlistItems.list(%s)", playlist_id)
        kwargs: dict = {
            "part": "contentDetails",
            "playlistId": playlist_id,
            "maxResults": min(max_results, 50),
        }
        if page_token:
            kwargs["pageToken"] = page_token
        response = self._service.playlistItems().list(**kwargs).execute()
        video_ids = [
            item["contentDetails"]["videoId"]
            for item in response.get("items", [])
            if item.get("contentDetails", {}).get("videoId")
        ]
        next_token = response.get("nextPageToken")
        return video_ids, next_token

    @retry_on_rate_limit
    def get_videos_metadata(self, video_ids: list[str]) -> "list[VideoMetadata]":
        """
        Fetch full metadata for up to 50 videos in one API call. Costs 1 quota unit.
        """
        if not video_ids:
            return []

        _rate_limit_delay()
        logger.debug("API CALL: videos.list(%d ids)", len(video_ids))
        response = (
            self._service.videos()
            .list(
                part="snippet,contentDetails,statistics",
                id=",".join(video_ids[:50]),
            )
            .execute()
        )

        results = []
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            content = item.get("contentDetails", {})
            stats = item.get("statistics", {})
            results.append(
                VideoMetadata(
                    video_id=item["id"],
                    title=snippet.get("title", ""),
                    description=snippet.get("description", ""),
                    channel_name=snippet.get("channelTitle", ""),
                    channel_id=snippet.get("channelId", ""),
                    upload_date=_parse_upload_date(snippet.get("publishedAt", "")),
                    duration=_parse_iso8601_duration(content.get("duration", "")),
                    view_count=int(stats["viewCount"]) if stats.get("viewCount") else None,
                    tags=snippet.get("tags", []),
                )
            )
        return results
