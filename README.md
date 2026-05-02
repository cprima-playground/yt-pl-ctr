# yt-pl-ctr

Automated YouTube playlist curator. Fetches recent videos from configured channels, classifies them, and adds them to curated playlists — on a schedule or on demand.

## How it works

Each channel in `configs/channels.yaml` defines a hierarchical topic taxonomy and a set of target playlists. Two classification strategies are supported and can be mixed per channel:

**ML classification** — a TF-IDF + logistic regression model trained on LLM-labeled episodes. The trained model is committed to `models/<channel_slug>/` so CI can classify without retraining. A per-channel `ml_confidence_threshold` controls how certain the model must be before assigning a category.

**Keyword mention matching** — scans title, description, and transcript for configurable entity names or phrases. A `min_mentions` threshold per playlist filters passing references from substantive coverage.

## Automated sync

A GitHub Actions workflow (`sync_playlists.yml`) runs on a daily schedule. It fetches recent videos via the YouTube Data API (OAuth2 with a long-lived refresh token), classifies them, and writes to playlists — no yt-dlp, no bot detection issues.

Manual trigger available via `workflow_dispatch` with `limit` and `dry_run` inputs.

## Local workflows

```bash
# Metadata + transcript ingest
just fetch-metadata "Channel Name"
just ingest-channel "Channel Name"

# ML pipeline: label → train → backfill
just retrain-channel "Channel Name"
just backfill-plan                  # classify cache, save plan
just backfill-execute-plan          # write plan to YouTube

# Keyword mention search
just mentions-plan "Channel Name"   # scan cache, save plan
just mentions-execute-plan          # write plan to YouTube

# Shortcuts
just jre                            # trigger GH Actions workflow (dry-run)
just jre false                      # trigger live run
just candace                        # run keyword scan locally
```

## Configuration

`configs/channels.yaml` — one file, all channels. Per-channel settings:

```yaml
channels:
  - name: "My Channel"
    url: "https://www.youtube.com/@handle/videos"
    channel_id: "UCxxxxx"           # optional; resolved from handle if omitted
    playlist_prefix: "MC"
    min_duration: 600               # skip shorts (seconds)
    max_age_days: 365               # ingest window
    ingest_limit: 1500              # max videos to fetch per ingest run
    ml_confidence_threshold: 0.08   # ML classifier cutoff

    taxonomy:
      - slug: parent_topic
        label: "Parent Topic"
        children:
          - slug: leaf_topic
            label: "leaf topic label"

    # ML-classified playlists (require trained model)
    playlists:
      leaf_topic:
        title: "MC – Parent – Leaf Topic"

    # Keyword-matched playlists (no model required)
    keyword_playlists:
      named_entity:
        title: "MC – Category – Named Entity"
        min_mentions: 2
        keywords:
          - "Full Name"
          - "Alternate Name"
```

## Environment variables

```bash
YT_CLIENT_ID=...
YT_CLIENT_SECRET=...
YT_REFRESH_TOKEN=...
YT_CACHE_DIR=/path/to/cache      # default: ./cache
```

Copy `.env.example` to `.env` for local development.

## Getting YouTube API credentials

1. [Google Cloud Console](https://console.cloud.google.com/) → create project → enable YouTube Data API v3
2. Create OAuth 2.0 credentials (Desktop app type)
3. Run `uv run python scripts/get_refresh_token.py` to obtain a refresh token

Store `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, and `YT_REFRESH_TOKEN` as repository secrets for GitHub Actions.

## Installation

```bash
uv sync
uv run yt-pl-ctr --help
```

Requires Python 3.11+.

## License

CC BY 4.0
