"""Sync orchestration for multi-channel playlist management."""

import logging
from dataclasses import dataclass, field

from .classifier import ClassificationResult, VideoClassifier
from .fetcher import VideoFetcherProtocol, YtDlpFetcher
from .models import ChannelConfig, Config, VideoMetadata
from .youtube import YouTubeClient

logger = logging.getLogger(__name__)


@dataclass
class ChannelSyncStats:
    """Statistics for a single channel sync."""

    channel_url: str
    videos_processed: int = 0
    videos_added: int = 0
    videos_skipped: int = 0
    videos_skipped_default: int = 0  # Videos skipped due to skip_default setting
    errors: int = 0
    classifications: dict[str, int] = field(default_factory=dict)

    def record_classification(self, category: str) -> None:
        self.classifications[category] = self.classifications.get(category, 0) + 1


@dataclass
class SyncStats:
    """Statistics for full sync run."""

    channels: list[ChannelSyncStats] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        return sum(c.videos_processed for c in self.channels)

    @property
    def total_added(self) -> int:
        return sum(c.videos_added for c in self.channels)

    @property
    def total_skipped(self) -> int:
        return sum(c.videos_skipped for c in self.channels)

    @property
    def total_errors(self) -> int:
        return sum(c.errors for c in self.channels)


@dataclass
class ClassifiedVideo:
    """A video with its classification result."""

    video: VideoMetadata
    classification: ClassificationResult
    playlist_name: str


def classify_channel_videos(
    channel_config: ChannelConfig,
    limit: int = 30,
    use_wikipedia: bool | None = None,
    fetcher: VideoFetcherProtocol | None = None,
) -> list[ClassifiedVideo]:
    """
    Fetch and classify videos from a channel.

    Args:
        channel_config: Channel configuration
        limit: Max videos to fetch
        use_wikipedia: Override channel's use_wikipedia setting
        fetcher: VideoFetcherProtocol implementation (defaults to YtDlpFetcher)

    Returns:
        List of classified videos
    """
    if fetcher is None:
        fetcher = YtDlpFetcher()
    wiki_enabled = use_wikipedia if use_wikipedia is not None else channel_config.use_wikipedia
    classifier = VideoClassifier(channel_config, use_wikipedia=wiki_enabled)
    results = []

    for video in fetcher.fetch_channel_videos(channel_config.url, limit):
        classification = classifier.classify(video)
        playlist_name = classifier.get_playlist_name(classification.category_key)
        results.append(
            ClassifiedVideo(
                video=video,
                classification=classification,
                playlist_name=playlist_name,
            )
        )

    return results


def sync_channel(
    channel_config: ChannelConfig,
    youtube: YouTubeClient,
    playlist_settings: "PlaylistSettings",
    fetcher: VideoFetcherProtocol,
    limit: int = 30,
    dry_run: bool = False,
    use_wikipedia: bool | None = None,
) -> ChannelSyncStats:
    """
    Sync a single channel's videos to playlists.

    Args:
        channel_config: Channel configuration
        youtube: YouTube API client
        playlist_settings: Playlist creation settings
        limit: Max videos to process
        dry_run: If True, don't make changes
        use_wikipedia: Override channel's use_wikipedia setting

    Returns:
        Sync statistics for this channel
    """
    stats = ChannelSyncStats(channel_url=channel_config.url)
    wiki_enabled = use_wikipedia if use_wikipedia is not None else channel_config.use_wikipedia
    classifier = VideoClassifier(channel_config, use_wikipedia=wiki_enabled)

    # Cache for playlist IDs
    playlist_ids: dict[str, str] = {}

    logger.info("Processing channel: %s", channel_config.url)

    for video in fetcher.fetch_channel_videos(channel_config.url, limit):
        stats.videos_processed += 1

        try:
            # Classify video
            result = classifier.classify(video)
            stats.record_classification(result.category_key)

            # Skip if marked as skipped (default category + skip_default enabled)
            if result.skipped:
                logger.info(
                    "[SKIP] '%s' -> default category (skip_default enabled)",
                    video.title[:50],
                )
                stats.videos_skipped_default += 1
                continue

            playlist_name = classifier.get_playlist_name(result.category_key)

            if dry_run:
                logger.info(
                    "[DRY RUN] '%s' -> %s (%s: %s)",
                    video.title[:50],
                    playlist_name,
                    result.match_reason,
                    result.matched_value or "n/a",
                )
                stats.videos_added += 1
                continue

            # Get or create playlist
            if playlist_name not in playlist_ids:
                playlist_ids[playlist_name] = youtube.ensure_playlist(
                    playlist_name,
                    description=playlist_settings.description_template,
                    privacy=playlist_settings.privacy,
                )

            # Add video to playlist
            playlist_id = playlist_ids[playlist_name]
            if youtube.add_video_if_missing(playlist_id, video.video_id):
                logger.info(
                    "Added '%s' -> %s (%s)",
                    video.title[:50],
                    playlist_name,
                    result.match_reason,
                )
                stats.videos_added += 1
            else:
                stats.videos_skipped += 1

        except Exception as e:
            logger.error("Error processing video %s: %s", video.video_id, e)
            stats.errors += 1

    return stats


def sync_all_channels(
    config: Config,
    youtube: YouTubeClient,
    fetcher: VideoFetcherProtocol,
    limit: int | None = None,
    dry_run: bool = False,
    channels: list[str] | None = None,
) -> SyncStats:
    """
    Sync all configured channels.

    Args:
        config: Full configuration
        youtube: YouTube API client
        limit: Override default limit (None = use config default)
        dry_run: If True, don't make changes
        channels: Optional list of channel URLs to sync (None = all)

    Returns:
        Combined sync statistics
    """
    stats = SyncStats()
    effective_limit = limit if limit is not None else config.limit

    for channel_config in config.channels:
        # Filter channels if specified
        if channels and channel_config.url not in channels:
            continue

        channel_stats = sync_channel(
            channel_config=channel_config,
            youtube=youtube,
            playlist_settings=config.playlist_settings,
            fetcher=fetcher,
            limit=effective_limit,
            dry_run=dry_run,
        )
        stats.channels.append(channel_stats)

        logger.info(
            "Channel %s: processed=%d, added=%d, skipped=%d, errors=%d",
            channel_config.playlist_prefix or channel_config.url,
            channel_stats.videos_processed,
            channel_stats.videos_added,
            channel_stats.videos_skipped,
            channel_stats.errors,
        )

    return stats
