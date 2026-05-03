"""Persistent store of video → playlist classification decisions.

Maps video_id → {category_key, playlist_id, playlist_item_id, title, classified_at}
so that reclassification can remove the video from its old playlist before adding
it to the new one. playlist_item_id (not playlist_id) is what YouTube's
playlistItems.delete endpoint requires.
"""

import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILE = ".yt-pl-ctr-classifications.json"


def load(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not load classification state from %s: %s", path, e)
    return {"classifications": {}}


def save(state: dict, path: Path) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Could not save classification state to %s: %s", path, e)


def get(state: dict, video_id: str) -> dict | None:
    return state.get("classifications", {}).get(video_id)


def put(
    state: dict,
    video_id: str,
    category_key: str,
    playlist_id: str,
    playlist_item_id: str,
    title: str = "",
) -> None:
    state.setdefault("classifications", {})[video_id] = {
        "category_key": category_key,
        "playlist_id": playlist_id,
        "playlist_item_id": playlist_item_id,
        "title": title,
        "classified_at": date.today().isoformat(),
    }
