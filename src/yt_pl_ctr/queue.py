"""Filesystem-based queue for async video processing."""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .models import VideoMetadata

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_DIR = Path("queue")


@dataclass
class QueueItem:
    """A queued video item."""

    video: VideoMetadata
    channel_url: str
    fetched_at: str
    file_path: Path | None = None


class VideoQueue:
    """Filesystem-based queue for video processing."""

    def __init__(self, base_dir: Path = DEFAULT_QUEUE_DIR):
        self.base_dir = base_dir
        self.pending_dir = base_dir / "pending"
        self.done_dir = base_dir / "done"
        self.failed_dir = base_dir / "failed"

        # Create directories
        for d in [self.pending_dir, self.done_dir, self.failed_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _make_filename(self, video: VideoMetadata) -> str:
        """
        Create filename with YYYY-MM-DD prefix from upload date.

        Format: YYYY-MM-DD_video_id.json
        """
        if video.upload_date and len(video.upload_date) == 8:
            # upload_date is YYYYMMDD format
            date_str = f"{video.upload_date[:4]}-{video.upload_date[4:6]}-{video.upload_date[6:8]}"
        else:
            # Fallback to current date
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
        return f"{date_str}_{video.video_id}.json"

    def enqueue(self, video: VideoMetadata, channel_url: str) -> Path:
        """
        Add a video to the pending queue.

        Returns:
            Path to the created queue file
        """
        item = {
            "video_id": video.video_id,
            "title": video.title,
            "description": video.description,
            "channel_name": video.channel_name,
            "channel_id": video.channel_id,
            "upload_date": video.upload_date,
            "duration": video.duration,
            "view_count": video.view_count,
            "channel_url": channel_url,
            "fetched_at": datetime.utcnow().isoformat(),
        }

        filename = self._make_filename(video)
        file_path = self.pending_dir / filename

        # Skip if already exists (in any state)
        if self._exists_anywhere(video.video_id):
            logger.debug("Video %s already in queue, skipping", video.video_id)
            return file_path

        with open(file_path, "w") as f:
            json.dump(item, f, indent=2)

        logger.info("Queued: %s", video.title[:50])
        return file_path

    def _exists_anywhere(self, video_id: str) -> bool:
        """Check if video exists in any queue state."""
        # Check for files ending with _{video_id}.json (new format)
        # or just {video_id}.json (old format)
        patterns = [f"*_{video_id}.json", f"{video_id}.json"]
        for pattern in patterns:
            for d in [self.pending_dir, self.done_dir, self.failed_dir]:
                if list(d.glob(pattern)):
                    return True
        return False

    def pending_count(self) -> int:
        """Return number of pending items."""
        return len(list(self.pending_dir.glob("*.json")))

    def iter_pending(self) -> Iterator[QueueItem]:
        """Iterate over pending items (sorted by filename = date order)."""
        files = sorted(self.pending_dir.glob("*.json"), key=lambda p: p.name)
        for file_path in files:
            try:
                with open(file_path) as f:
                    data = json.load(f)
                yield QueueItem(
                    video=VideoMetadata(
                        video_id=data["video_id"],
                        title=data["title"],
                        description=data.get("description", ""),
                        channel_name=data.get("channel_name", ""),
                        channel_id=data.get("channel_id", ""),
                        upload_date=data.get("upload_date"),
                        duration=data.get("duration"),
                        view_count=data.get("view_count"),
                    ),
                    channel_url=data["channel_url"],
                    fetched_at=data["fetched_at"],
                    file_path=file_path,
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("Invalid queue file %s: %s", file_path, e)
                self._move_to_failed(file_path)

    def mark_done(self, item: QueueItem) -> None:
        """Move item to done queue."""
        if item.file_path and item.file_path.exists():
            dest = self.done_dir / item.file_path.name
            item.file_path.rename(dest)
            logger.debug("Marked done: %s", item.video.video_id)

    def mark_failed(self, item: QueueItem, error: str = "") -> None:
        """Move item to failed queue."""
        if item.file_path and item.file_path.exists():
            # Add error info to the file
            with open(item.file_path) as f:
                data = json.load(f)
            data["error"] = error
            data["failed_at"] = datetime.utcnow().isoformat()

            dest = self.failed_dir / item.file_path.name
            with open(dest, "w") as f:
                json.dump(data, f, indent=2)
            item.file_path.unlink()
            logger.debug("Marked failed: %s - %s", item.video.video_id, error)

    def _move_to_failed(self, file_path: Path) -> None:
        """Move a file to failed directory."""
        dest = self.failed_dir / file_path.name
        file_path.rename(dest)

    def watch(self, interval: float = 2.0) -> Iterator[QueueItem]:
        """
        Watch for new pending items.

        Yields items as they appear, polling at the given interval.
        """
        seen: set[str] = set()
        while True:
            for item in self.iter_pending():
                if item.video.video_id not in seen:
                    seen.add(item.video.video_id)
                    yield item
            time.sleep(interval)
