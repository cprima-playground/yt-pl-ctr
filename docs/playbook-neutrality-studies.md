# Playbook: Neutrality Studies (NS)

## Channel details

| Field | Value |
|---|---|
| Channel ID | `UCHdLVKdAeG6zAeZMGZh91bg` |
| URL | `https://www.youtube.com/@neutralitystudies/videos` |
| Prefix | `NS` |
| Min duration | 300 s (5 min) |
| Ingest limit | 5000 |
| Confidence threshold | 0.08 |

## Playlists

| Slug | Title |
|---|---|
| `caucasus` | NS: Eastern Europe / Caucasus Region |
| `intelligence_covert_ops` | NS: Media & Governance / Intelligence & Covert Operations |
| `china_taiwan_japan` | NS: East Asia / China, Taiwan & Japan |

## Regular update sequence

No CI workflow yet. Run manually:

```powershell
uv run yt-pl-ctr sync --config configs/channels.yaml --channel "https://www.youtube.com/@neutralitystudies/videos" --limit 30 --verbose
```

## Retraining sequence

```powershell
# 1. LLM-label all episodes (idempotent)
uv run python scripts/llm_label.py --channel "Neutrality Studies" --all --cache-dir T:\yt-pl-ctr

# 2. Train classifier
uv run python scripts/train_classifier.py --channel "Neutrality Studies" --cache-dir T:\yt-pl-ctr

# 3. Commit updated model
git add models/neutrality_studies/
git commit -m "feat: retrain NS classifier"
```

## Backfill sequence

```powershell
# 1. Refresh playlist membership cache
uv run python scripts/fetch_playlist_membership.py --force --cache-dir T:\yt-pl-ctr

# 2. Generate and review plan
uv run python scripts/backfill.py --channel "Neutrality Studies" --save-plan --cache-dir T:\yt-pl-ctr

# 3. Execute (quota-consuming)
uv run python scripts/backfill.py --channel "Neutrality Studies" --execute-plan --cache-dir T:\yt-pl-ctr
```

If quota is hit mid-run, re-run `--execute-plan` the next day — it is idempotent.

## Full ingest sequence

```powershell
# 1. Fetch video metadata
uv run python scripts/download_test_data.py --channel "Neutrality Studies" --cache-dir T:\yt-pl-ctr

# 2. Fetch transcripts + enrich
uv run python scripts/pipeline.py --only ingest --channel "Neutrality Studies" --cache-dir T:\yt-pl-ctr

# 3. Fetch Wikipedia summaries
uv run python scripts/fetch_wikipedia.py --cache-dir T:\yt-pl-ctr

# 4. Build NLP features
uv run python scripts/build_features.py --cache-dir T:\yt-pl-ctr

# 5. LLM label sample, review, then all
uv run python scripts/llm_label.py --channel "Neutrality Studies" --limit 20 --cache-dir T:\yt-pl-ctr
uv run python scripts/llm_label.py --channel "Neutrality Studies" --all --cache-dir T:\yt-pl-ctr

# 6. Train classifier
uv run python scripts/train_classifier.py --channel "Neutrality Studies" --cache-dir T:\yt-pl-ctr

# 7. Backfill
uv run python scripts/backfill.py --channel "Neutrality Studies" --save-plan --cache-dir T:\yt-pl-ctr
uv run python scripts/backfill.py --channel "Neutrality Studies" --execute-plan --cache-dir T:\yt-pl-ctr
```

## Notes

- Model location: `models/neutrality_studies/`
- LLM labeled data: `T:\yt-pl-ctr\llm_labeled_neutrality_studies.json`
- 952 qualifying episodes (>= 300 s), 947 labeled, 15 classes
- Taxonomy defined from BERTopic discovery (NMF engine) in May 2026
- YouTube quota limit (10,000 units/day) may require splitting backfill across days
