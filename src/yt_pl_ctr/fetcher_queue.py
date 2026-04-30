"""Fetcher that writes to the filesystem queue."""

import json
import logging
import time
from pathlib import Path

from .config import load_config
from .fetcher import VideoFetcherProtocol, YtDlpFetcher
from .models import Config
from .queue import VideoQueue

logger = logging.getLogger(__name__)

STATE_FILE = ".yt-pl-ctr-state.json"


def load_state(state_file: Path = Path(STATE_FILE)) -> dict:
    """Load state from dotfile."""
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"offsets": {}}


def save_state(state: dict, state_file: Path = Path(STATE_FILE)) -> None:
    """Save state to dotfile."""
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def get_channel_offset(state: dict, channel_url: str) -> int:
    """Get last offset for a channel."""
    return state.get("offsets", {}).get(channel_url, 0)


def set_channel_offset(state: dict, channel_url: str, offset: int) -> None:
    """Set offset for a channel."""
    if "offsets" not in state:
        state["offsets"] = {}
    state["offsets"][channel_url] = offset


def fetch_to_queue(
    config: Config,
    queue: VideoQueue,
    fetcher: VideoFetcherProtocol | None = None,
    limit: int | None = None,
    offset: int | None = None,
    delay: float = 1.0,
    resume: bool = True,
    state_file: Path = Path(STATE_FILE),
) -> int:
    """
    Fetch videos from all configured channels and add to queue.

    Args:
        config: Configuration with channel definitions
        queue: Queue to write to
        limit: Max videos per channel (None = use config default)
        offset: Number of videos to skip (None = use saved offset if resume=True)
        delay: Delay between fetches in seconds
        resume: If True and offset is None, resume from last saved offset
        state_file: Path to state file for tracking offsets

    Returns:
        Number of videos queued
    """
    if fetcher is None:
        fetcher = YtDlpFetcher()
    total_queued = 0
    state = load_state(state_file) if resume else {"offsets": {}}

    for channel in config.channels:
        # Determine offset for this channel
        if offset is not None:
            channel_offset = offset
        elif resume:
            channel_offset = get_channel_offset(state, channel.url)
        else:
            channel_offset = 0

        logger.info("Fetching from: %s (offset: %d)", channel.url, channel_offset)

        try:
            videos = fetcher.fetch_channel_videos(
                url=channel.url,
                limit=limit or 50,
                offset=channel_offset,
            )

            channel_fetched = 0
            prev_video = None
            for video in videos:
                # Delay after previous video (not before first)
                if prev_video is not None and delay > 0:
                    time.sleep(delay)

                queue.enqueue(video, channel.url)
                total_queued += 1
                channel_fetched += 1
                prev_video = video

            # Update offset in state
            new_offset = channel_offset + channel_fetched
            set_channel_offset(state, channel.url, new_offset)
            save_state(state, state_file)
            logger.info("Saved offset %d for %s", new_offset, channel.url)

        except Exception as e:
            logger.error("Error fetching %s: %s", channel.url, e)

    return total_queued


def run_fetcher(
    config_path: Path,
    queue_dir: Path = Path("queue"),
    limit: int | None = None,
    delay: float = 1.0,
) -> None:
    """
    Run the fetcher process.

    Args:
        config_path: Path to config YAML
        queue_dir: Directory for queue files
        limit: Max videos per channel
        delay: Delay between fetches
    """
    config = load_config(config_path)
    queue = VideoQueue(queue_dir)

    logger.info("Starting fetcher, queue dir: %s", queue_dir)
    logger.info("Channels: %d", len(config.channels))

    count = fetch_to_queue(config, queue, limit=limit, delay=delay)
    logger.info("Fetched %d videos to queue", count)
    logger.info("Pending: %d", queue.pending_count())
