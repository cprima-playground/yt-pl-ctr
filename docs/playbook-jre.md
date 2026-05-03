# Playbook: Joe Rogan Experience (JRE)

## Channel details

| Field | Value |
|---|---|
| Channel ID | `UCzQUP1qoWDoEbmsQxvdjxgQ` |
| URL | `https://www.youtube.com/@joerogan/videos` |
| Prefix | `JRE` |
| Min duration | 3600 s (1 h) |
| Confidence threshold | 0.08 |

## Playlists

| Slug | Title |
|---|---|
| `ancient_civilizations` | JRE: History & Unexplained / Ancient Civilizations & Alternative History |
| `ufo_extraterrestrial` | JRE: History & Unexplained / UFO, Paranormal & Cryptozoology |
| `paranormal_cryptozoology` | JRE: History & Unexplained / UFO, Paranormal & Cryptozoology |
| `psychology_mental_health` | JRE: Science & Mind / Psychology & Mental Health |

## Regular update sequence

```powershell
# 1. Sync latest episodes into playlists (runs daily via CI at 03:23 UTC)
uv run yt-pl-ctr sync --config configs/channels.yaml --channel "https://www.youtube.com/@joerogan/videos" --limit 30 --verbose

# 2. Manual trigger via GitHub Actions
gh workflow run sync_playlists.yml --repo cprima-playground/yt-pl-ctr --ref main -f dry_run=false -f limit=30
```

## Retraining sequence

Run when taxonomy changes or labeled data grows significantly.

```powershell
# 1. LLM-label all episodes (idempotent)
uv run python scripts/llm_label.py --channel "Joe Rogan Experience" --all --cache-dir T:\yt-pl-ctr

# 2. Train classifier
uv run python scripts/train_classifier.py --channel "Joe Rogan Experience" --cache-dir T:\yt-pl-ctr

# 3. Commit updated model
git add models/joe_rogan_experience/
git commit -m "feat: retrain JRE classifier"
```

## Full ingest sequence (new back-catalogue)

```powershell
# 1. Fetch video metadata
uv run python scripts/download_test_data.py --channel "Joe Rogan Experience" --cache-dir T:\yt-pl-ctr

# 2. Fetch transcripts + enrich
uv run python scripts/pipeline.py --only ingest --channel "Joe Rogan Experience" --cache-dir T:\yt-pl-ctr

# 3. Fetch Wikipedia summaries
uv run python scripts/fetch_wikipedia.py --cache-dir T:\yt-pl-ctr

# 4. Build NLP features
uv run python scripts/build_features.py --cache-dir T:\yt-pl-ctr

# 5. LLM label sample, review, then all
uv run python scripts/llm_label.py --channel "Joe Rogan Experience" --limit 20 --cache-dir T:\yt-pl-ctr
uv run python scripts/llm_label.py --channel "Joe Rogan Experience" --all --cache-dir T:\yt-pl-ctr

# 6. Train classifier
uv run python scripts/train_classifier.py --channel "Joe Rogan Experience" --cache-dir T:\yt-pl-ctr

# 7. Backfill
uv run python scripts/backfill.py --channel "Joe Rogan Experience" --save-plan --cache-dir T:\yt-pl-ctr
uv run python scripts/backfill.py --channel "Joe Rogan Experience" --execute-plan --cache-dir T:\yt-pl-ctr
```

## Notes

- yt-dlp is bot-blocked on GitHub Actions IPs — CI uses YouTube Data API v3 for metadata fetch
- Model location: `models/joe_rogan_experience/`
- LLM labeled data: `T:\yt-pl-ctr\llm_labeled.json` (legacy name, no channel slug)
