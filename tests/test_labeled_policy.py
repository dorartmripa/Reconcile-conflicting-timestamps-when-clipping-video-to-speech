import json
from pathlib import Path

from src.reconciler import reconcile


ROOT = Path(__file__).resolve().parents[1]


def test_policy_beats_always_stt_on_labeled_sample(sample_aligned):
    labels = json.loads((ROOT / "sample" / "labeled_conflicts.json").read_text())
    expected = {row["word"]: row["expected"] for row in labels["cases"]}
    result = reconcile(sample_aligned)
    got = {row.word: row.decision for row in result.conflicts if row.word in expected}

    policy_hits = sum(got[word] == winner for word, winner in expected.items())
    always_stt_hits = sum(winner == "STT" for winner in expected.values())
    assert len(got) == len(expected)
    assert policy_hits == len(expected)
    assert policy_hits > always_stt_hits
