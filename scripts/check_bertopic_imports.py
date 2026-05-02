#!/usr/bin/env python3
"""Quick smoke test: verify BERTopic engine imports work before running discovery.

Exit 0 = all imports OK.
Exit 1 = an import failed (missing dep or SIGBUS).

Usage:
    uv run python scripts/check_bertopic_imports.py
    uv run python scripts/check_bertopic_imports.py --umap   # also test umap
"""

import os
import sys

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_bertopic_cache")

CHECKS = [
    ("numpy",               "import numpy"),
    ("bertopic",            "from bertopic import BERTopic"),
    ("sentence_transformers","from sentence_transformers import SentenceTransformer"),
    ("sklearn.cluster",     "from sklearn.cluster import KMeans"),
    ("sklearn.decomposition","from sklearn.decomposition import PCA"),
]

UMAP_CHECK = ("umap",  "from umap import UMAP")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--umap", action="store_true", help="Also test umap import")
args = parser.parse_args()

checks = CHECKS + ([UMAP_CHECK] if args.umap else [])

ok = True
for label, stmt in checks:
    try:
        exec(stmt)
        print(f"  {label:30s} ok")
    except Exception as e:
        print(f"  {label:30s} FAIL: {e}", file=sys.stderr)
        ok = False

if ok:
    print("\nAll imports OK — BERTopic engine should work.")
    sys.exit(0)
else:
    print("\nSome imports failed.", file=sys.stderr)
    print("Run: uv sync --extra topic-discovery", file=sys.stderr)
    sys.exit(1)
