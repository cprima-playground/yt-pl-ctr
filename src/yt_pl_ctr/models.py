"""Pydantic models for data structures."""

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

    @property
    def guest_name(self) -> str | None:
        """Extract guest name from JRE-style titles (e.g., 'JRE #1234 - Guest Name')."""
        if " - " in self.title:
            return self.title.split(" - ", 1)[1].strip()
        return None


class Category(BaseModel):
    """A category for classifying videos."""

    name: str = Field(..., description="Display name for the category")
    keywords: list[str] = Field(default_factory=list, description="Keywords to match in title/description")
    guests: list[str] = Field(default_factory=list, description="Known guest names for this category")
    description_patterns: list[str] = Field(
        default_factory=list, description="Regex patterns to match in description"
    )
    priority: int | None = Field(
        default=None,
        description="Priority order for keyword matching. Lower values are checked first before Wikipedia lookup."
    )


class ChannelConfig(BaseModel):
    """Configuration for a single YouTube channel."""

    url: str = Field(..., description="yt-dlp compatible URL")
    playlist_prefix: str = Field(..., description="Prefix for playlist names")
    categories: dict[str, Category] = Field(default_factory=dict)
    default_category: str = Field(default="other", description="Fallback category")
    skip_default: bool = Field(
        default=False,
        description="If True, videos that fall into default category are not added to any playlist",
    )
    use_wikipedia: bool = Field(
        default=True,
        description="If True, use Wikipedia to help classify guests",
    )
    guest_pattern: str = Field(
        default=r"\s-\s(.+)$", description="Regex to extract guest name from title"
    )
    min_duration: int = Field(
        default=0,
        description="Minimum video duration in seconds. Videos shorter than this are skipped.",
    )


class PlaylistSettings(BaseModel):
    """Settings for playlist creation."""

    privacy: str = Field(default="unlisted", description="Privacy status: public, unlisted, private")
    description_template: str = Field(
        default="Auto-managed by yt-pl-ctr", description="Description for new playlists"
    )


class Config(BaseModel):
    """Root configuration model."""

    channels: list[ChannelConfig] = Field(default_factory=list)
    playlist_settings: PlaylistSettings = Field(default_factory=PlaylistSettings)
    limit: int = Field(default=30, description="Default number of videos to fetch per channel")
