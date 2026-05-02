"""Sync orchestration for multi-channel playlist management."""

import logging
from dataclasses import dataclass, field

from .classifier import ClassificationResult, VideoClassifier
from .fetcher import VideoFetcherProtocol, YtDlpFetcher, is_ci
from .models import ChannelConfig, Config, PlaylistSettings, VideoMetadata
from .youtube import YouTubeClient

logger = logging.getLogger(__name__)


@dataclass
class ChannelSyncStats:
    """Statistics for a single channel sync."""

    channel_url: str
    videos_processed: int = 0
    videos_added: int = 0
    videos_skipped: int = 0
    videos_skipped_default: int = 0
    videos_reclassified: int = 0  # Moved from one playlist to another
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
    fetcher: VideoFetcherProtocol | None = None,
) -> list[ClassifiedVideo]:
    """
    Fetch and classify videos from a channel.

    Args:
        channel_config: Channel configuration
        limit: Max videos to fetch
        fetcher: VideoFetcherProtocol implementation (defaults to YtDlpFetcher)

    Returns:
        List of classified videos
    """
    if fetcher is None:
        fetcher = YtDlpFetcher()
    classifier = VideoClassifier(channel_config, use_transcripts=not is_ci())
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
    playlist_settings: PlaylistSettings,
    fetcher: VideoFetcherProtocol,
    limit: int = 30,
    dry_run: bool = False,
) -> ChannelSyncStats:
    """Sync a single channel's videos to playlists, reclassifying when needed.

    At the start of each run, fetches all current playlist contents to build
    an in-memory index {video_id: (category_key, playlist_item_id)}. This lets
    us detect reclassifications and move videos without a persistent state file
    — safe for ephemeral CI runners.
    """
    stats = ChannelSyncStats(channel_url=channel_config.url)
    classifier = VideoClassifier(channel_config, use_transcripts=not is_ci())

    # playlist name → playlist_id
    playlist_ids: dict[str, str] = {}

    logger.info("Processing channel: %s", channel_config.url)

    # ── Build in-memory placement index ──────────────────────────────────────
    # {video_id: {"category_key": ..., "playlist_item_id": ..., "playlist_id": ...}}
    # Fetched from the actual YouTube playlists — no local state file needed.
    placement: dict[str, dict] = {}

    if not dry_run:
        logger.info("Building placement index from existing playlists...")
        for slug in channel_config.playlists:
            pl_name = classifier.get_playlist_name(slug)
            pl_id = youtube.find_playlist(pl_name)
            if pl_id:
                playlist_ids[pl_name] = pl_id
                contents = youtube.get_playlist_contents(pl_id)
                for vid_id, item_id in contents.items():
                    placement[vid_id] = {
                        "category_key": slug,
                        "playlist_item_id": item_id,
                        "playlist_id": pl_id,
                    }
                logger.debug("Indexed %d videos from %s", len(contents), pl_name)
        logger.info(
            "Placement index: %d videos across %d playlists", len(placement), len(playlist_ids)
        )

    # ── Process fetched videos ────────────────────────────────────────────────
    cutoff = channel_config.min_upload_date_str()  # None if max_age_days not set
    for video in fetcher.fetch_channel_videos(channel_config.url, limit):
        # Videos arrive newest-first; stop as soon as we pass the age window
        if cutoff and video.upload_date and video.upload_date < cutoff:
            logger.info("Reached age limit (%s < %s) — stopping fetch", video.upload_date, cutoff)
            break
        stats.videos_processed += 1

        try:
            result = classifier.classify(video)
            stats.record_classification(result.category_key)

            if result.skipped:
                logger.info("[SKIP] '%s' -> default (skip_default)", video.title[:50])
                stats.videos_skipped_default += 1
                continue

            playlist_name = classifier.get_playlist_name(result.category_key)
            prev = placement.get(video.video_id)

            if dry_run:
                if prev and prev["category_key"] != result.category_key:
                    logger.info(
                        "[DRY RUN] RECLASSIFY '%s': %s → %s (%s: %s)",
                        video.title[:50],
                        prev["category_key"],
                        result.category_key,
                        result.match_reason,
                        result.matched_value or "n/a",
                    )
                elif prev:
                    logger.debug(
                        "[DRY RUN] SKIP '%s' already in %s", video.title[:50], playlist_name
                    )
                else:
                    logger.info(
                        "[DRY RUN] ADD '%s' -> %s (%s: %s)",
                        video.title[:50],
                        playlist_name,
                        result.match_reason,
                        result.matched_value or "n/a",
                    )
                stats.videos_added += 1
                continue

            # Already in the correct playlist — nothing to do
            if prev and prev["category_key"] == result.category_key:
                logger.debug("'%s' already in %s, skipping", video.title[:50], playlist_name)
                stats.videos_skipped += 1
                continue

            # Ensure target playlist exists
            if playlist_name not in playlist_ids:
                playlist_ids[playlist_name] = youtube.ensure_playlist(
                    playlist_name,
                    description=playlist_settings.description_template,
                    privacy=playlist_settings.privacy,
                )
            target_playlist_id = playlist_ids[playlist_name]

            # Remove from old playlist if reclassified
            if prev and prev["category_key"] != result.category_key:
                try:
                    youtube.remove_playlist_item(prev["playlist_item_id"])
                    logger.info(
                        "Removed '%s' from %s (reclassified → %s)",
                        video.title[:50],
                        prev["category_key"],
                        result.category_key,
                    )
                    stats.videos_reclassified += 1
                except Exception as e:
                    logger.warning("Could not remove %s from old playlist: %s", video.video_id, e)

            # Add to target playlist
            playlist_item_id = youtube.add_video_if_missing(target_playlist_id, video.video_id)
            if playlist_item_id:
                placement[video.video_id] = {
                    "category_key": result.category_key,
                    "playlist_item_id": playlist_item_id,
                    "playlist_id": target_playlist_id,
                }
                logger.info(
                    "Added '%s' -> %s (%s: %s)",
                    video.title[:50],
                    playlist_name,
                    result.match_reason,
                    result.matched_value or "n/a",
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
