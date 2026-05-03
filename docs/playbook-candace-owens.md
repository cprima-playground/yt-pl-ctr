# Playbook: Candace Owens (CO)

## Channel details

| Field | Value |
|---|---|
| Channel ID | `UCL0u5uz7KZ9q-pe-VC8TY-w` |
| URL | `https://www.youtube.com/@RealCandaceO/videos` |
| Prefix | `CO` |
| Min duration | 600 s (10 min) |
| Max age | 730 days |
| Ingest limit | 1500 |
| Confidence threshold | 0.08 |

## Playlists

### ML-based

| Slug | Title |
|---|---|
| `us_elections_politics` | CO: Politics / US Elections & Domestic Politics |
| `deep_state_censorship` | CO: Politics / Deep State & Censorship |
| `media_celebrity` | CO: Media / Celebrity & Culture |
| `lawsuits_courts` | CO: Society & Law / Lawsuits & Courts |

### Keyword-based

| Slug | Title | Keywords | Min mentions |
|---|---|---|---|
| `france_europe` | CO: International / France & Europe | Brigitte Macron, Emmanuel Macron, Macron | 5 |
| `celebrity_controversy` | CO: Media / Celebrity Controversy & Hollywood | Blake Lively, Justin Baldoni, It Ends with Us, Ryan Reynolds | 5 |

## Regular update sequence

```powershell
# Sync latest episodes (runs daily via CI at 04:17 UTC)
gh workflow run sync_candace_owens.yml --repo cprima-playground/yt-pl-ctr --ref main -f dry_run=false -f limit=30
```

## Retraining sequence

```powershell
# 1. LLM-label all episodes (idempotent)
uv run python scripts/llm_label.py --channel "Candace Owens" --all --cache-dir T:\yt-pl-ctr

# 2. Train classifier
uv run python scripts/train_classifier.py --channel "Candace Owens" --cache-dir T:\yt-pl-ctr

# 3. Commit updated model
git add models/candace_owens/
git commit -m "feat: retrain CO classifier"
```

## Taxonomy migration

If taxonomy classes are consolidated or renamed, remap existing labels before retraining:

```powershell
uv run python scripts/migrate_labels.py `
    --input T:\yt-pl-ctr\llm_labeled_candace_owens.json `
    --output T:\yt-pl-ctr\llm_labeled_candace_owens.json
```

Edit `REMAP` dict in `scripts/migrate_labels.py` to define old→new slug mappings.

## Keyword mention backfill

```powershell
# Preview matches
uv run python scripts/search_mentions.py --channel "Candace Owens" --cache-dir T:\yt-pl-ctr

# Save plan, review keyword_plan.json, then execute
uv run python scripts/search_mentions.py --channel "Candace Owens" --save-plan --cache-dir T:\yt-pl-ctr
uv run python scripts/search_mentions.py --channel "Candace Owens" --execute-plan --cache-dir T:\yt-pl-ctr
```

## Full ingest sequence

```powershell
# 1. Fetch video metadata
uv run python scripts/download_test_data.py --channel "Candace Owens" --cache-dir T:\yt-pl-ctr

# 2. Fetch transcripts + enrich
uv run python scripts/pipeline.py --only ingest --channel "Candace Owens" --cache-dir T:\yt-pl-ctr

# 3. Fetch Wikipedia summaries
uv run python scripts/fetch_wikipedia.py --cache-dir T:\yt-pl-ctr

# 4. Build NLP features
uv run python scripts/build_features.py --cache-dir T:\yt-pl-ctr

# 5. LLM label sample, review, then all
uv run python scripts/llm_label.py --channel "Candace Owens" --limit 20 --cache-dir T:\yt-pl-ctr
uv run python scripts/llm_label.py --channel "Candace Owens" --all --cache-dir T:\yt-pl-ctr

# 6. Train classifier
uv run python scripts/train_classifier.py --channel "Candace Owens" --cache-dir T:\yt-pl-ctr

# 7. Backfill
uv run python scripts/backfill.py --channel "Candace Owens" --save-plan --cache-dir T:\yt-pl-ctr
uv run python scripts/backfill.py --channel "Candace Owens" --execute-plan --cache-dir T:\yt-pl-ctr
```

## Notes

- Model location: `models/candace_owens/`
- LLM labeled data: `T:\yt-pl-ctr\llm_labeled_candace_owens.json`
- Taxonomy consolidated from 13 → 7 classes in May 2026 (accuracy was 0.432)
