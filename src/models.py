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
    def is_conflict(self) -> bool:
        return self.difference > CONFLICT_THRESHOLD_SECONDS


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
    notes: list[str] = field(default_factory=list)

    @property
    def clip_start(self) -> float:
        if not self.final_words:
            return 0.0
        return max(0.0, self.final_words[0].start)

    @property
    def clip_end(self) -> float:
        if not self.final_words:
            return 0.0
        return self.final_words[-1].end
