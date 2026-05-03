#!/usr/bin/env python3
"""Remap old taxonomy slugs to new ones in an llm_labeled JSON file.

Usage:
    uv run python scripts/migrate_labels.py
        --input T:/yt-pl-ctr/llm_labeled_candace_owens.json
        --output T:/yt-pl-ctr/llm_labeled_candace_owens.json
"""

import argparse
import json
from pathlib import Path

REMAP = {
    # media consolidation
    "celebrity_controversy":      "media_celebrity",
    "mainstream_media_criticism": "media_celebrity",
    "entertainment_news":         "media_celebrity",
    # politics consolidation
    "deep_state_establishment":   "deep_state_censorship",
    "censorship_free_speech":     "deep_state_censorship",
    # society consolidation
    "gender_feminism":            "society_values",
    "family_values_religion":     "society_values",
    "crime_public_safety":        "society_values",
    # removed — no data, not worth a class
    "government_policy":          "other",
    "africa_diaspora":            "other",
    "race_identity":              "other",
    "corporate_woke_agenda":      "other",
    "economy_inflation":          "other",
    "big_tech_censorship":        "other",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))

    counts: dict[str, int] = {}
    for record in data:
        old = record.get("category", "other")
        new = REMAP.get(old, old)
        if old != new:
            counts[f"{old} -> {new}"] = counts.get(f"{old} -> {new}", 0) + 1
        record["category"] = new

    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Migrated {len(data)} records")
    if counts:
        print("Remapped:")
        for mapping, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {mapping}: {n}")
    else:
        print("No remapping needed.")


if __name__ == "__main__":
    main()
