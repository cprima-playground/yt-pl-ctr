"""Measure classifier accuracy against the manually labeled golden set.

To build the golden set:
    uv run python scripts/download_test_data.py --limit 100
    uv run python scripts/classify_corpus.py
    # Edit tests/fixtures/jre_to_label.json — set correct_category for each entry
    cp tests/fixtures/jre_to_label.json tests/fixtures/jre_labeled.json

Then run:
    uv run pytest tests/test_classifier_accuracy.py -v
"""

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS_FILE = FIXTURES / "jre_videos.json"
LABELED_FILE = FIXTURES / "jre_labeled.json"

ACCURACY_THRESHOLD = 0.80  # fail if overall accuracy drops below 80%


def load_corpus() -> dict[str, dict]:
    """Load corpus keyed by video_id."""
    if not CORPUS_FILE.exists():
        return {}
    with open(CORPUS_FILE) as f:
        videos = json.load(f)
    return {v["video_id"]: v for v in videos}


def load_labeled() -> list[dict]:
    """Load manually labeled golden set."""
    if not LABELED_FILE.exists():
        return []
    with open(LABELED_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def classifier():
    from yt_pl_ctr.config import load_config
    from yt_pl_ctr.classifier import VideoClassifier

    config_path = Path(__file__).parent.parent / "configs" / "channels.yaml"
    config = load_config(config_path)
    # No Wikipedia in tests — deterministic and fast
    return VideoClassifier(config.channels[0], use_wikipedia=False)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def labeled():
    return load_labeled()


def pytest_configure(config):
    config.addinivalue_line("markers", "golden: test against manually labeled golden set")


@pytest.mark.skipif(not CORPUS_FILE.exists(), reason="corpus not built — run download_test_data.py")
@pytest.mark.skipif(not LABELED_FILE.exists(), reason="golden set not built — run classify_corpus.py and label jre_to_label.json")
class TestClassifierAccuracy:

    def test_labeled_set_not_empty(self, labeled):
        assert len(labeled) > 0, "Golden set is empty — label some entries in jre_labeled.json"

    def test_overall_accuracy(self, classifier, corpus, labeled):
        """Overall accuracy must meet threshold."""
        from yt_pl_ctr.models import VideoMetadata

        correct = 0
        wrong = []

        for entry in labeled:
            video_data = corpus.get(entry["video_id"])
            if not video_data:
                continue
            video = VideoMetadata(**{k: v for k, v in video_data.items() if k in VideoMetadata.model_fields})
            result = classifier.classify(video)

            if result.category_key == entry["correct_category"]:
                correct += 1
            else:
                wrong.append({
                    "title": entry["title"],
                    "expected": entry["correct_category"],
                    "got": result.category_key,
                    "reason": result.match_reason,
                    "matched": result.matched_value,
                })

        total = len([e for e in labeled if e["video_id"] in corpus])
        accuracy = correct / total if total > 0 else 0.0

        if wrong:
            print(f"\nMisclassified ({len(wrong)}/{total}):")
            for w in wrong[:20]:
                print(f"  [{w['expected']} → {w['got']}] {w['title'][:60]}")
                print(f"    reason={w['reason']}, matched={w['matched']}")

        assert accuracy >= ACCURACY_THRESHOLD, (
            f"Accuracy {accuracy:.1%} below threshold {ACCURACY_THRESHOLD:.0%} "
            f"({correct}/{total} correct)"
        )

    def test_per_category_recall(self, classifier, corpus, labeled):
        """Each category should have reasonable recall (not all mapped to wrong bucket)."""
        from yt_pl_ctr.models import VideoMetadata

        by_category: dict[str, dict[str, int]] = {}

        for entry in labeled:
            video_data = corpus.get(entry["video_id"])
            if not video_data:
                continue
            expected = entry["correct_category"]
            video = VideoMetadata(**{k: v for k, v in video_data.items() if k in VideoMetadata.model_fields})
            result = classifier.classify(video)

            if expected not in by_category:
                by_category[expected] = {"correct": 0, "total": 0}
            by_category[expected]["total"] += 1
            if result.category_key == expected:
                by_category[expected]["correct"] += 1

        print("\nPer-category recall:")
        for cat, counts in sorted(by_category.items()):
            recall = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            print(f"  {cat:<20} {counts['correct']}/{counts['total']}  {recall:.0%}")
            if counts["total"] >= 3:
                assert recall >= 0.50, (
                    f"Category '{cat}' recall {recall:.0%} below 50% "
                    f"({counts['correct']}/{counts['total']})"
                )


@pytest.mark.skipif(not CORPUS_FILE.exists(), reason="corpus not built")
class TestClassifierSmokeTests:
    """Fast deterministic tests — no Wikipedia, no network."""

    @pytest.mark.parametrize("title,expected_category", [
        ("Joe Rogan Experience #2492 - Ari Shaffir", "comedy"),
        ("Joe Rogan Experience #1169 - Elon Musk", "science_tech"),
        ("Joe Rogan Experience #1315 - Bob Lazar & Jeremy Corbell", "ufo_aliens"),
        ("Joe Rogan Experience #1683 - Andrew Huberman", "science_tech"),
        ("Joe Rogan Experience #1908 - Jon Jones", "mma_martial_arts"),
        ("Joe Rogan Experience #1805 - Conor McGregor", "mma_martial_arts"),
        ("Joe Rogan Experience #1284 - Graham Hancock", "ancient_history"),
    ])
    def test_known_guests(self, classifier, title, expected_category):
        from yt_pl_ctr.models import VideoMetadata

        video = VideoMetadata(video_id="test", title=title, description="")
        result = classifier.classify(video)
        assert result.category_key == expected_category, (
            f"'{title}' → got '{result.category_key}' (reason: {result.match_reason}), "
            f"expected '{expected_category}'"
        )
