"""Shared data structures for timestamp sources, conflicts, and decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CONFLICT_THRESHOLD_SECONDS = 0.5


@dataclass
class WordTimestamp:
    """A single word aligned to a timestamp from one source."""

    word: str
    start: float
    end: float
    confidence: float | None = None
    source: str = ""
    trusted: bool = True
    interpolated: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class AlignedWord:
    """The same transcript word observed by both timestamp sources."""

    index: int
    word: str
    metadata: WordTimestamp
    stt: WordTimestamp

    @property
    def difference(self) -> float:
        return abs(self.stt.start - self.metadata.start)

    @property
    def end_difference(self) -> float:
        return abs(self.stt.end - self.metadata.end)

    @property
    def is_conflict(self) -> bool:
        if self.difference > CONFLICT_THRESHOLD_SECONDS:
            return True
        if self.end_difference > CONFLICT_THRESHOLD_SECONDS:
            return True
        return _interval_containment_conflict(self.stt, self.metadata)


def _interval_containment_conflict(
    left: WordTimestamp, right: WordTimestamp
) -> bool:
    """True when one span swallows the other by more than the conflict threshold."""

    if left.duration < 0.05 or right.duration < 0.05:
        return False
    left_in_right = left.start >= right.start - 1e-9 and left.end <= right.end + 1e-9
    right_in_left = right.start >= left.start - 1e-9 and right.end <= left.end + 1e-9
    if not (left_in_right or right_in_left):
        return False
    return abs(left.duration - right.duration) > CONFLICT_THRESHOLD_SECONDS


@dataclass
class ConflictLog:
    """Explainable record for one metadata-vs-STT conflict."""

    number: int
    word: str
    metadata_timestamp: float
    stt_timestamp: float
    difference: float
    stt_confidence: float | None
    factors: dict[str, Any]
    decision: str
    reason: str
    final_timestamp: float


@dataclass
class ReconciliationResult:
    """Full output of the decision engine."""

    aligned: list[AlignedWord]
    conflicts: list[ConflictLog]
    final_words: list[WordTimestamp]
    estimated_offset: float
    offset_mad: float
    drift_detected: bool
    estimated_slope: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def trusted_words(self) -> list[WordTimestamp]:
        trusted = [word for word in self.final_words if word.trusted]
        return trusted or list(self.final_words)

    @property
    def clip_start(self) -> float:
        words = self.trusted_words
        if not words:
            return 0.0
        return max(0.0, words[0].start)

    @property
    def clip_end(self) -> float:
        words = self.trusted_words
        if not words:
            return 0.0
        return max(words[-1].end, self.clip_start + 0.05)
