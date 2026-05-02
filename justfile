# yt-pl-ctr development tasks
# Install just: https://github.com/casey/just

# Show available recipes
default:
    @just --list

# ── Pipeline ──────────────────────────────────────────────────────────────────

# Run the full pipeline (all phases, LLM skipped unless --llm-limit provided separately)
pipeline *ARGS:
    uv run python scripts/pipeline.py {{ ARGS }}

# Run only a specific phase
pipeline-only PHASE:
    uv run python scripts/pipeline.py --only {{ PHASE }}

# Show pipeline status without running anything
pipeline-status:
    uv run python scripts/pipeline.py --status

# ── Cache management ──────────────────────────────────────────────────────────

# Invalidate the playlist membership cache (forces re-fetch on next run)
invalidate-membership:
    uv run python scripts/fetch_playlist_membership.py --invalidate

# Force-refresh the playlist membership cache from YouTube API
refresh-membership:
    uv run python scripts/fetch_playlist_membership.py --force

# Show current playlist membership cache summary
membership:
    uv run python scripts/fetch_playlist_membership.py

# Delete all derived cache files (membership, labels) — keep raw episode data
clean-derived:
    rm -f "${YT_CACHE_DIR:-cache}/playlist_membership.json"
    rm -f "${YT_CACHE_DIR:-cache}/llm_labeled.json"
    rm -f "${YT_CACHE_DIR:-cache}/unlabeled.json"
    @echo "Derived cache cleared."

# ── Per-channel ingest (metadata → transcripts + enrichment) ──────────────────

# Fetch video metadata for a channel into cache (step 1 of ingest)
# Usage: just fetch-metadata "Candace Owens"
fetch-metadata CHANNEL:
    uv run python scripts/download_test_data.py --channel "{{ CHANNEL }}"

# Fetch transcripts + enrich for a channel (step 2 of ingest)
# Usage: just ingest-channel "Candace Owens"
ingest-channel CHANNEL:
    uv run python scripts/pipeline.py --only ingest --channel "{{ CHANNEL }}"

# ── Retraining sequence (manual, run when taxonomy or data changes) ───────────

# Full retraining: LLM label all episodes → train classifier (default: first channel)
retrain:
    uv run python scripts/llm_label.py --all
    uv run python scripts/train_classifier.py

# Retrain for a specific channel
# Usage: just retrain-channel "Candace Owens"
retrain-channel CHANNEL:
    uv run python scripts/llm_label.py --all --channel "{{ CHANNEL }}"
    uv run python scripts/train_classifier.py --channel "{{ CHANNEL }}"

# ── Labeling pipeline ─────────────────────────────────────────────────────────

# Run LLM labeling on all qualifying episodes (idempotent — skips cached)
llm-label-all:
    uv run python scripts/llm_label.py --all

# Run LLM labeling on specific video IDs
# Usage: just llm-label MtoPEub7XwA znQB0FumtV8
llm-label *VIDEO_IDS:
    uv run python scripts/llm_label.py {{ VIDEO_IDS }}

# Run LLM labeling on first N episodes
llm-label-n N='20':
    uv run python scripts/llm_label.py --limit {{ N }}

# ── Feature pipeline ──────────────────────────────────────────────────────────

# Build NLP features for all cached episodes with transcripts
build-features:
    uv run python scripts/build_features.py

# Fetch Wikipedia summaries for all episode guests
fetch-wikipedia:
    uv run python scripts/fetch_wikipedia.py

# ── Topic discovery ───────────────────────────────────────────────────────────

# Discover topic clusters across all episodes (outputs topic_summary.txt)
discover-topics:
    uv run python scripts/discover_topics.py

# Discover topics only for episodes labeled 'other' by llm_label.py
discover-topics-other:
    uv run python scripts/discover_topics.py --only-other

# ── Keyword mention search (no ML required) ───────────────────────────────────

# Preview keyword matches for a channel (no API writes)
# Usage: just mentions "Candace Owens"
mentions CHANNEL:
    uv run python scripts/search_mentions.py --channel "{{ CHANNEL }}"

# Save keyword plan for review
# Usage: just mentions-plan "Candace Owens"
mentions-plan CHANNEL:
    uv run python scripts/search_mentions.py --channel "{{ CHANNEL }}" --save-plan

# Execute saved keyword plan against YouTube API
mentions-execute-plan:
    uv run python scripts/search_mentions.py --execute-plan

# ── Code quality ──────────────────────────────────────────────────────────────

# Run linter
lint:
    uv run ruff check src/ scripts/

# Auto-fix lint issues
lint-fix:
    uv run ruff check --fix src/ scripts/

# Format code
fmt:
    uv run ruff format src/ scripts/

# Run all code quality checks
check: lint
    uv run ruff format --check src/ scripts/

# ── Development ───────────────────────────────────────────────────────────────

# Install dependencies
install:
    uv sync

# Dry-run sync (no API writes)
sync-dry:
    uv run yt-pl-ctr sync --config configs/channels.yaml --limit 5 --dry-run --verbose

# Live sync — writes to YouTube
sync-live:
    uv run yt-pl-ctr sync --config configs/channels.yaml

# Classify all cached episodes and print plan (no YouTube writes, safe on any IP)
backfill-classify:
    uv run python scripts/backfill.py

# Classify and save plan to backfill_plan.json (no YouTube writes)
backfill-plan:
    uv run python scripts/backfill.py --save-plan

# Execute saved plan against YouTube API (quota-consuming, run after reviewing plan)
backfill-execute-plan:
    uv run python scripts/backfill.py --execute-plan

# Classify and execute in one step (legacy convenience)
backfill:
    uv run python scripts/backfill.py --execute

# Full pipeline dry-run (safe, no YouTube writes)
run:
    uv run python scripts/pipeline.py --skip llm

# Full pipeline live (writes to YouTube)
run-live:
    uv run python scripts/pipeline.py --skip llm --execute
