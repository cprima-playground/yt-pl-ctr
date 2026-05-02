"""Queue processor - watches queue and adds videos to playlists."""

import logging
from pathlib import Path

from .classifier import VideoClassifier
from .config import load_config
from .models import ChannelConfig, Config
from .queue import QueueItem, VideoQueue
from .youtube import YouTubeAPIError, YouTubeClient

logger = logging.getLogger(__name__)


def get_channel_config(config: Config, channel_url: str) -> ChannelConfig | None:
    """Find channel config by URL."""
    for ch in config.channels:
        if ch.url == channel_url:
            return ch
    return None


def process_item(
    item: QueueItem,
    config: Config,
    youtube: YouTubeClient,
    dry_run: bool = False,
) -> bool:
    """
    Process a single queue item.

    Returns:
        True if processed successfully, False otherwise
    """
    channel_config = get_channel_config(config, item.channel_url)
    if not channel_config:
        logger.error("No config for channel: %s", item.channel_url)
        return False

    # Skip videos shorter than min_duration
    if channel_config.min_duration > 0:
        duration = item.video.duration or 0
        if duration < channel_config.min_duration:
            logger.info(
                "Skipped (too short: %ds < %ds): %s",
                duration,
                channel_config.min_duration,
                item.video.title[:50],
            )
            return True

    classifier = VideoClassifier(channel_config)
    result = classifier.classify(item.video)

    if result.skipped:
        logger.info("Skipped (default): %s", item.video.title[:50])
        return True

    playlist_name = classifier.get_playlist_name(result.category_key)

    if dry_run:
        logger.info("[DRY] Would add to '%s': %s", playlist_name, item.video.title[:50])
        return True

    try:
        playlist_id = youtube.ensure_playlist(playlist_name)
        added = youtube.add_video_if_missing(playlist_id, item.video.video_id)
        if added:
            logger.info("Added to '%s': %s", playlist_name, item.video.title[:50])
        else:
            logger.info("Already in '%s': %s", playlist_name, item.video.title[:50])
        return True
    except YouTubeAPIError as e:
        logger.error("API error: %s", e)
        return False


def process_pending(
    config: Config,
    queue: VideoQueue,
    youtube: YouTubeClient | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> tuple[int, int, int]:
    """
    Process all pending items in queue.

    Returns:
        Tuple of (processed, succeeded, failed)
    """
    if not dry_run and youtube is None:
        youtube = YouTubeClient.from_env()

    processed = 0
    succeeded = 0
    failed = 0

    for item in queue.iter_pending():
        if limit and processed >= limit:
            break

        try:
            if process_item(item, config, youtube, dry_run=dry_run):
                queue.mark_done(item)
                succeeded += 1
            else:
                queue.mark_failed(item, "Processing failed")
                failed += 1
        except Exception as e:
            logger.exception("Error processing %s", item.video.video_id)
            queue.mark_failed(item, str(e))
            failed += 1

        processed += 1

    return processed, succeeded, failed


def watch_and_process(
    config: Config,
    queue: VideoQueue,
    youtube: YouTubeClient | None = None,
    dry_run: bool = False,
    interval: float = 2.0,
) -> None:
    """
    Watch queue and process items as they arrive.

    Runs indefinitely until interrupted.
    """
    if not dry_run and youtube is None:
        youtube = YouTubeClient.from_env()

    logger.info("Watching queue for new items (interval: %.1fs)...", interval)

    try:
        for item in queue.watch(interval=interval):
            try:
                if process_item(item, config, youtube, dry_run=dry_run):
                    queue.mark_done(item)
                else:
                    queue.mark_failed(item, "Processing failed")
            except Exception as e:
                logger.exception("Error processing %s", item.video.video_id)
                queue.mark_failed(item, str(e))
    except KeyboardInterrupt:
        logger.info("Stopped watching")


def run_processor(
    config_path: Path,
    queue_dir: Path = Path("queue"),
    watch: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
) -> None:
    """
    Run the processor.

    Args:
        config_path: Path to config YAML
        queue_dir: Directory for queue files
        watch: If True, watch continuously; if False, process pending and exit
        dry_run: Don't actually add to playlists
        limit: Max items to process (only for non-watch mode)
    """
    config = load_config(config_path)
    queue = VideoQueue(queue_dir)

    youtube = None
    if not dry_run:
        youtube = YouTubeClient.from_env()
        info = youtube.get_channel_info()
        logger.info("YouTube: %s", info.get("title"))

    logger.info("Queue dir: %s", queue_dir)
    logger.info("Pending: %d", queue.pending_count())

    if watch:
        watch_and_process(config, queue, youtube, dry_run=dry_run)
    else:
        processed, succeeded, failed = process_pending(
            config, queue, youtube, dry_run=dry_run, limit=limit
        )
        logger.info("Processed: %d, Succeeded: %d, Failed: %d", processed, succeeded, failed)
