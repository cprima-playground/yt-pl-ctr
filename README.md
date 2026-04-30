# yt-pl-ctr

YouTube Playlist Controller - Classify videos and automatically add them to categorized playlists.

## Features

- **Automatic classification** of videos based on:
  - Guest name matching (e.g., "Dave Chappelle" → comedian)
  - Keyword matching in title and description
  - Configurable categories with fallback to default
- **YAML configuration** for easy customization of categories, keywords, and guest mappings
- **Dry-run mode** to preview changes before applying
- **GitHub Actions** integration for scheduled automation
- **Idempotent operation** - safe to run multiple times

## Installation

```bash
# Using pip
pip install .

# Or with uv
uv sync
```

## Configuration

Create a YAML config file (see `configs/jre.yaml` for example):

```yaml
channel_url: "https://www.youtube.com/@yourchannel/videos"
playlist_prefix: "MyChannel"
playlist_privacy: "unlisted"

categories:
  category_name:
    keywords:
      - "keyword1"
      - "keyword2"
    guests:
      - "Known Guest Name"

default_category: "other"
guest_pattern: "\\s-\\s(.+)$"
```

## Usage

### Command Line

```bash
# Sync playlists (dry run first!)
yt-pl-ctr sync --config configs/jre.yaml --dry-run

# Apply changes
yt-pl-ctr sync --config configs/jre.yaml --limit 30

# List categories from config
yt-pl-ctr list --config configs/jre.yaml
```

### Environment Variables

Required for YouTube API access:

```bash
export YT_CLIENT_ID="your-client-id"
export YT_CLIENT_SECRET="your-client-secret"
export YT_REFRESH_TOKEN="your-refresh-token"
```

### GitHub Actions

The included workflow (`.github/workflows/sync_playlists.yml`) runs daily at 03:23 UTC.

Configure these secrets in your repository:
- `YT_CLIENT_ID`
- `YT_CLIENT_SECRET`
- `YT_REFRESH_TOKEN`

Manual trigger available via workflow_dispatch with options for config file, limit, and dry-run mode.

## Getting YouTube API Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the YouTube Data API v3
4. Create OAuth 2.0 credentials (Desktop app type)
5. Download credentials and note the Client ID and Secret
6. Use the OAuth flow to get a refresh token with `youtube` scope

## Classification Logic

Videos are classified in this order of priority:

1. **Guest match**: If the extracted guest name matches a known guest in any category
2. **Title keywords**: If any category keyword appears in the video title
3. **Description keywords**: If any category keyword appears in the description
4. **Default**: Falls back to configured `default_category`

## License

CC BY 4.0
