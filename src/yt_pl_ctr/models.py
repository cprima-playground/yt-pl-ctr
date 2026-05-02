"""Pydantic models for data structures."""

from __future__ import annotations

import datetime

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Metadata for a YouTube video."""

    video_id: str
    title: str
    description: str = ""
    channel_name: str = ""
    channel_id: str = ""
    upload_date: str | None = None
    duration: int | None = None
    view_count: int | None = None
    tags: list[str] = Field(default_factory=list, description="YouTube topic tags")


class TaxonomyNode(BaseModel):
    """A node in the topic taxonomy tree (domain or leaf)."""

    slug: str
    label: str
    children: list[TaxonomyNode] = Field(default_factory=list)

    def leaf_nodes(self) -> list[TaxonomyNode]:
        """Return all leaf nodes (nodes with no children) in this subtree."""
        if not self.children:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.leaf_nodes())
        return result


TaxonomyNode.model_rebuild()


class PlaylistConfig(BaseModel):
    """Configuration for a topic that maps to a YouTube playlist."""

    title: str = Field(..., description="Exact YouTube playlist title")


class KeywordPlaylistConfig(BaseModel):
    """A playlist populated by keyword/entity matching rather than ML classification."""

    title: str = Field(..., description="Exact YouTube playlist title")
    keywords: list[str] = Field(
        default_factory=list,
        description="Case-insensitive terms; a video matches if any term appears in title, description, or transcript.",
    )


class ChannelConfig(BaseModel):
    """Configuration for a single YouTube channel."""

    name: str = Field(default="", description="Channel display name")
    url: str = Field(..., description="Channel URL (e.g. https://www.youtube.com/@handle/videos)")
    channel_id: str | None = Field(
        default=None,
        description="YouTube channel ID (UCxxx). If None, resolved from url at runtime.",
    )
    playlist_prefix: str = Field(..., description="Prefix for playlist names")
    min_duration: int = Field(
        default=0,
        description="Minimum video duration in seconds. Videos shorter than this are skipped.",
    )
    max_age_days: int | None = Field(
        default=None,
        description="Ignore videos older than this many days. None means no limit.",
    )
    ml_confidence_threshold: float = Field(
        default=0.5, description="Minimum ML confidence to assign a topic"
    )
    taxonomy: list[TaxonomyNode] = Field(default_factory=list)
    playlists: dict[str, PlaylistConfig] = Field(
        default_factory=dict,
        description="Leaf slugs → ML-classified YouTube playlists",
    )
    keyword_playlists: dict[str, KeywordPlaylistConfig] = Field(
        default_factory=dict,
        description="Slug → keyword-matched YouTube playlists (entity/mention-based)",
    )

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier derived from name, e.g. 'candace_owens'."""
        return self.name.lower().replace(" ", "_").replace("-", "_")

    def min_upload_date_str(self) -> str | None:
        """YYYYMMDD cutoff for max_age_days, or None if unconstrained."""
        if self.max_age_days is None:
            return None
        cutoff = datetime.date.today() - datetime.timedelta(days=self.max_age_days)
        return cutoff.strftime("%Y%m%d")

    def all_leaf_slugs(self) -> list[str]:
        """Return all leaf topic slugs from the taxonomy."""
        result = []
        for node in self.taxonomy:
            result.extend(n.slug for n in node.leaf_nodes())
        return result

    def playlist_title(self, slug: str) -> str | None:
        """Return playlist title for a slug, or None if not a playlist topic."""
        pc = self.playlists.get(slug)
        return pc.title if pc else None


class PlaylistSettings(BaseModel):
    """Settings for playlist creation."""

    privacy: str = Field(
        default="unlisted", description="Privacy status: public, unlisted, private"
    )
    description_template: str = Field(
        default="Auto-managed by yt-pl-ctr", description="Description for new playlists"
    )


class Config(BaseModel):
    """Root configuration model."""

    channels: list[ChannelConfig] = Field(default_factory=list)
    playlist_settings: PlaylistSettings = Field(default_factory=PlaylistSettings)
    limit: int = Field(default=30, description="Default number of videos to fetch per channel")
