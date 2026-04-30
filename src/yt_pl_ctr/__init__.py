"""YouTube Playlist Controller - Classify videos and manage playlists automatically."""

__version__ = "0.1.0"

from .config import load_config
from .models import Category, ChannelConfig, Config, PlaylistSettings, VideoMetadata
from .classifier import ClassificationResult, VideoClassifier
from .fetcher import fetch_channel_videos, fetch_video_metadata
from .youtube import YouTubeClient, YouTubeAPIError
from .sync import classify_channel_videos, sync_all_channels, sync_channel
from .wikipedia import WikipediaInfo, lookup_person, get_primary_topic
from .queue import VideoQueue, QueueItem

__all__ = [
    "__version__",
    # Config
    "load_config",
    "Config",
    "ChannelConfig",
    "Category",
    "PlaylistSettings",
    # Models
    "VideoMetadata",
    # Classification
    "VideoClassifier",
    "ClassificationResult",
    # Fetching
    "fetch_channel_videos",
    "fetch_video_metadata",
    # YouTube API
    "YouTubeClient",
    "YouTubeAPIError",
    # Sync
    "classify_channel_videos",
    "sync_channel",
    "sync_all_channels",
    # Wikipedia
    "WikipediaInfo",
    "lookup_person",
    "get_primary_topic",
    # Queue
    "VideoQueue",
    "QueueItem",
]
