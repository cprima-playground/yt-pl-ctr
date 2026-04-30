"""YouTube Data API client for playlist management."""

import logging
import os
import time
from dataclasses import dataclass, field
from functools import wraps

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube"]

# Retry settings for rate limits
MAX_RETRIES = 5
BASE_DELAY = 60.0  # seconds (YouTube rate limits need longer waits)
MAX_DELAY = 900.0  # 15 minutes max

# Minimum delay between API calls (seconds)
API_CALL_DELAY = 5.0
_last_api_call = 0.0


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
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    hours = delay / 3600
                    if delay >= 3600:
                        logger.warning(
                            "Rate limited (attempt %d/%d), waiting %.1f hours...",
                            attempt + 1, MAX_RETRIES, hours,
                        )
                    else:
                        logger.warning(
                            "Rate limited (attempt %d/%d), waiting %.0f seconds...",
                            attempt + 1, MAX_RETRIES, delay,
                        )
                    time.sleep(delay)
                    last_error = e
                else:
                    raise
        raise RateLimitError(
            f"Rate limit exceeded after {MAX_RETRIES} retries"
        ) from last_error

    return wrapper


@dataclass
class YouTubeClient:
    """Client for YouTube Data API operations."""

    _service: object = field(repr=False)
    _playlist_cache: dict[str, str] = field(default_factory=dict, repr=False)
    _added_videos: set[tuple[str, str]] = field(default_factory=set, repr=False)  # (playlist_id, video_id)

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
            request = self._service.playlists().list_next(request, response)

        return None

    @retry_on_rate_limit
    def create_playlist(
        self, title: str, description: str = "", privacy: str = "public"
    ) -> str:
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

    def ensure_playlist(
        self, title: str, description: str = "", privacy: str = "public"
    ) -> str:
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
    def add_video_to_playlist(self, playlist_id: str, video_id: str) -> bool:
        """
        Add a video to a playlist.

        Args:
            playlist_id: Target playlist ID
            video_id: Video ID to add

        Returns:
            True if added, False if already exists
        """
        _rate_limit_delay()
        logger.info("API CALL: playlistItems.insert (%s)", video_id)
        try:
            self._service.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {"kind": "youtube#video", "videoId": video_id},
                    }
                },
            ).execute()
            # Track in session cache to prevent duplicates
            self._added_videos.add((playlist_id, video_id))
            logger.info("Added video %s to playlist %s", video_id, playlist_id)
            return True
        except HttpError as e:
            if e.resp.status == 409:
                logger.debug("Video %s already in playlist %s", video_id, playlist_id)
                return False
            if e.resp.status == 429:
                raise  # Let decorator handle quota wait
            raise YouTubeAPIError(
                f"Failed to add video {video_id} to playlist: {e}"
            ) from e

    def add_video_if_missing(self, playlist_id: str, video_id: str) -> bool:
        """
        Add video to playlist only if not already present.

        Args:
            playlist_id: Target playlist ID
            video_id: Video ID to add

        Returns:
            True if added, False if already present
        """
        # Check session cache first (handles API eventual consistency)
        if (playlist_id, video_id) in self._added_videos:
            logger.debug("Video %s already added in this session, skipping", video_id)
            return False

        # Check YouTube API
        if self.playlist_contains_video(playlist_id, video_id):
            logger.debug("Video %s already in playlist, skipping", video_id)
            self._added_videos.add((playlist_id, video_id))  # Cache for future checks
            return False

        return self.add_video_to_playlist(playlist_id, video_id)

