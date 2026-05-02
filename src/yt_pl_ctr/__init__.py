"""YouTube Playlist Controller - Classify videos and manage playlists automatically."""

__version__ = "0.1.0"

from .classifier import ClassificationResult, VideoClassifier
from .config import load_config
from .fetcher import (
    VideoFetcherProtocol,
    YouTubeAPIFetcher,
    YtDlpFetcher,
    fetch_channel_videos,
    fetch_video_metadata,
)
from .models import ChannelConfig, Config, PlaylistSettings, TaxonomyNode, VideoMetadata
from .queue import QueueItem, VideoQueue
from .sync import classify_channel_videos, sync_all_channels, sync_channel
from .wikipedia import WikipediaInfo, get_primary_topic, lookup_person
from .youtube import YouTubeAPIError, YouTubeClient

__all__ = [
    "__version__",
    # Config
    "load_config",
    "Config",
    "ChannelConfig",
    "TaxonomyNode",
    "PlaylistSettings",
    # Models
    "VideoMetadata",
    # Classification
    "VideoClassifier",
    "ClassificationResult",
    # Fetching
    "fetch_channel_videos",
    "fetch_video_metadata",
    "VideoFetcherProtocol",
    "YtDlpFetcher",
    "YouTubeAPIFetcher",
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
