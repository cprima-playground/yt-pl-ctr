"""Structured episode cache helpers.

Layout under cache_dir:
  index.json              — lightweight list of all episodes for fast enumeration
  episodes/{video_id}/
    metadata.json         — full VideoMetadata fields
    transcript.txt        — auto-generated transcript text (fetched on demand)
"""

import datetime
import json
from pathlib import Path

MEMBERSHIP_MAX_AGE_HOURS = 12
_MEMBERSHIP_FILE = "playlist_membership.json"


def episode_dir(cache_dir: Path, video_id: str) -> Path:
    return cache_dir / "episodes" / video_id


# ── Index ──────────────────────────────────────────────────────────────────────


def read_index(cache_dir: Path) -> list[dict]:
    f = cache_dir / "index.json"
    return json.loads(f.read_text()) if f.exists() else []


def write_index(cache_dir: Path, entries: list[dict]) -> None:
    (cache_dir / "index.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False))


def index_entry(video: dict) -> dict:
    """Lightweight summary stored in index.json."""
    return {
        "video_id": video["video_id"],
        "title": video.get("title", ""),
        "upload_date": video.get("upload_date"),
        "duration": video.get("duration"),
        "has_transcript": False,
    }


# ── Metadata ───────────────────────────────────────────────────────────────────


def write_metadata(cache_dir: Path, video: dict) -> None:
    d = episode_dir(cache_dir, video["video_id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps(video, indent=2, ensure_ascii=False))


def read_metadata(cache_dir: Path, video_id: str) -> dict | None:
    f = episode_dir(cache_dir, video_id) / "metadata.json"
    return json.loads(f.read_text()) if f.exists() else None


# ── Wikipedia ─────────────────────────────────────────────────────────────────


def write_wikipedia(cache_dir: Path, video_id: str, data: dict) -> None:
    """Store raw Wikipedia data — no category mapping, just text for ML features."""
    d = episode_dir(cache_dir, video_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "wikipedia.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))


def read_wikipedia(cache_dir: Path, video_id: str) -> dict | None:
    f = episode_dir(cache_dir, video_id) / "wikipedia.json"
    return json.loads(f.read_text()) if f.exists() else None


def has_wikipedia(cache_dir: Path, video_id: str) -> bool:
    return (episode_dir(cache_dir, video_id) / "wikipedia.json").exists()


# ── Transcript ─────────────────────────────────────────────────────────────────


def write_transcript(cache_dir: Path, video_id: str, text: str) -> None:
    d = episode_dir(cache_dir, video_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "transcript.txt").write_text(text, encoding="utf-8")


def read_transcript(cache_dir: Path, video_id: str) -> str | None:
    f = episode_dir(cache_dir, video_id) / "transcript.txt"
    return f.read_text(encoding="utf-8") if f.exists() else None


def has_transcript(cache_dir: Path, video_id: str) -> bool:
    return (episode_dir(cache_dir, video_id) / "transcript.txt").exists()


# ── Convenience ────────────────────────────────────────────────────────────────


def known_ids(cache_dir: Path) -> set[str]:
    return {e["video_id"] for e in read_index(cache_dir)}


# ── LLM response cache ────────────────────────────────────────────────────────


def read_llm_response(
    cache_dir: Path, video_id: str, model: str, prompt_key: str | None = None
) -> dict | None:
    """Return cached LLM result for this episode+model+prompt_key, or None on miss.

    prompt_key is a short hash of the taxonomy leaf slugs used when the response was
    generated. When the taxonomy changes the key changes, forcing a fresh API call.
    """
    f = episode_dir(cache_dir, video_id) / "llm_response.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    if data.get("model") != model:
        return None
    cached_pk = data.get("prompt_key")
    if prompt_key and cached_pk and cached_pk != prompt_key:
        return None  # both sides have a key and they differ — taxonomy changed
    return data.get("result")


def write_llm_response(
    cache_dir: Path,
    video_id: str,
    model: str,
    result: dict,
    prompt_key: str | None = None,
    user_prompt: str | None = None,
    system_prompt: str | None = None,
    taxonomy_snapshot: list[dict] | None = None,
) -> None:
    """Persist full LLM request + response for this episode.

    Stores the exact prompts sent and taxonomy used so the call is fully auditable
    and reproducible without re-reading config files.
    """
    d = episode_dir(cache_dir, video_id)
    d.mkdir(parents=True, exist_ok=True)
    data = {
        "model": model,
        "prompt_key": prompt_key,
        "cached_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "request": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
        "taxonomy_snapshot": taxonomy_snapshot,  # [{"slug": ..., "label": ...}, ...]
        "result": result,
    }
    (d / "llm_response.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ── Playlist membership ────────────────────────────────────────────────────────


def _membership_path(cache_dir: Path) -> Path:
    return cache_dir / _MEMBERSHIP_FILE


def read_playlist_membership(
    cache_dir: Path, max_age_hours: float = MEMBERSHIP_MAX_AGE_HOURS
) -> dict | None:
    """Load cached membership. Returns None if missing or older than max_age_hours."""
    f = _membership_path(cache_dir)
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    try:
        fetched_at = datetime.datetime.fromisoformat(data["fetched_at"])
        age = datetime.datetime.now(datetime.UTC) - fetched_at
        if age > datetime.timedelta(hours=max_age_hours):
            return None
    except (KeyError, ValueError):
        return None
    return data


def write_playlist_membership(cache_dir: Path, playlists: dict, membership: dict) -> None:
    """Write membership snapshot with current UTC timestamp."""
    data = {
        "fetched_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "playlists": playlists,
        "membership": membership,
    }
    _membership_path(cache_dir).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def invalidate_playlist_membership(cache_dir: Path) -> bool:
    """Delete cached membership file. Returns True if it existed."""
    f = _membership_path(cache_dir)
    if f.exists():
        f.unlink()
        return True
    return False
