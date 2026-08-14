"""Deterministic timestamp reconciliation.

The engine never blindly prefers one source. Each conflict is scored using:

- STT confidence
- Raw timestamp difference
- Neighbor consistency (does this time sit cleanly between neighbors?)
- Monotonic ordering
- Systematic offset / drift between the two sources

A source that would make the word sequence go backwards is rejected.
"""

from __future__ import annotations

from statistics import median

from src.models import (
    CONFLICT_THRESHOLD_SECONDS,
    AlignedWord,
    ConflictLog,
    ReconciliationResult,
    WordTimestamp,
)


HIGH_CONFIDENCE = 0.80
LOW_CONFIDENCE = 0.45
DRIFT_MAD_LIMIT = 0.25
DRIFT_OFFSET_MIN = 0.15


def align_words(
    transcript_words: list[str],
    metadata: list[WordTimestamp],
    stt: list[WordTimestamp],
) -> list[AlignedWord]:
    if not (len(transcript_words) == len(metadata) == len(stt)):
        raise ValueError("Transcript, metadata, and STT must have the same length.")
    aligned = []
    for i, word in enumerate(transcript_words):
        aligned.append(
            AlignedWord(
                index=i,
                word=word,
                metadata=metadata[i],
                stt=stt[i],
            )
        )
    return aligned


def detect_conflicts(aligned: list[AlignedWord]) -> list[AlignedWord]:
    return [item for item in aligned if item.is_conflict]


def estimate_offset(aligned: list[AlignedWord]) -> tuple[float, float]:
    """Return (median STT-minus-metadata offset, median absolute deviation)."""

    high_conf = [
        item.stt.start - item.metadata.start
        for item in aligned
        if (item.stt.confidence or 0.0) >= HIGH_CONFIDENCE
    ]
    deltas = high_conf or [item.stt.start - item.metadata.start for item in aligned]
    if not deltas:
        return 0.0, 0.0
    center = median(deltas)
    mad = median([abs(delta - center) for delta in deltas])
    return float(center), float(mad)


def neighbor_consistency(times: list[float], index: int) -> float:
    """Score in [0, 1] for how well times[index] sits among its neighbors."""

    score = 1.0
    current = times[index]
    if index > 0:
        previous = times[index - 1]
        if current < previous:
            score -= 0.6
        elif current - previous > 3.0:
            score -= 0.25
        elif current - previous < 0.02:
            score -= 0.15
    if index < len(times) - 1:
        following = times[index + 1]
        if current > following:
            score -= 0.6
        elif following - current > 3.0:
            score -= 0.25
        elif following - current < 0.02:
            score -= 0.15
    return max(0.0, min(1.0, score))


def is_monotonic(times: list[float], index: int, candidate: float) -> bool:
    if index > 0 and candidate + 1e-9 < times[index - 1]:
        return False
    if index < len(times) - 1 and candidate - 1e-9 > times[index + 1]:
        return False
    return True


def _score_stt(
    confidence: float,
    neighbor: float,
    monotonic: bool,
    residual_after_drift: float,
    difference: float,
) -> float:
    score = 0.0
    score += confidence * 40.0
    score += neighbor * 25.0
    score += 15.0 if monotonic else 0.0
    if residual_after_drift <= CONFLICT_THRESHOLD_SECONDS:
        score += 10.0
    # Very large unexplained gaps slightly penalize STT (possible hallucination).
    if difference > 2.0:
        score -= 5.0
    return score


def _score_metadata(
    confidence: float,
    neighbor: float,
    monotonic: bool,
    residual_after_drift: float,
) -> float:
    score = 0.0
    score += (1.0 - confidence) * 35.0
    score += neighbor * 25.0
    score += 15.0 if monotonic else 0.0
    if residual_after_drift <= CONFLICT_THRESHOLD_SECONDS:
        score += 12.0
    return score


def reconcile(
    aligned: list[AlignedWord],
    padding: float = 0.0,
) -> ReconciliationResult:
    stt_times = [item.stt.start for item in aligned]
    meta_times = [item.metadata.start for item in aligned]
    offset, offset_mad = estimate_offset(aligned)
    drift_detected = offset_mad <= DRIFT_MAD_LIMIT and abs(offset) >= DRIFT_OFFSET_MIN

    notes = [
        f"Estimated systematic offset (STT - metadata): {offset:+.3f}s "
        f"(MAD {offset_mad:.3f}s)."
    ]
    if drift_detected:
        notes.append(
            "Systematic drift detected. Drift-corrected metadata is used as a "
            "factor, but raw |STT-metadata| > 0.5s still counts as a conflict."
        )

    conflicts: list[ConflictLog] = []
    final_words: list[WordTimestamp] = []
    conflict_number = 0

    for item in aligned:
        confidence = 0.0 if item.stt.confidence is None else item.stt.confidence
        residual = abs(item.stt.start - (item.metadata.start + offset))
        stt_neighbor = neighbor_consistency(stt_times, item.index)
        meta_neighbor = neighbor_consistency(meta_times, item.index)
        stt_mono = is_monotonic(stt_times, item.index, item.stt.start)
        meta_mono = is_monotonic(meta_times, item.index, item.metadata.start)

        if not item.is_conflict:
            chosen_start = item.stt.start if confidence >= HIGH_CONFIDENCE else (
                (item.stt.start + item.metadata.start) / 2.0
            )
            chosen_end = item.stt.end if confidence >= HIGH_CONFIDENCE else (
                (item.stt.end + item.metadata.end) / 2.0
            )
            source = "stt" if confidence >= HIGH_CONFIDENCE else "blend"
            final_words.append(
                WordTimestamp(
                    word=item.word,
                    start=chosen_start,
                    end=max(chosen_end, chosen_start + 0.05),
                    confidence=item.stt.confidence,
                    source=source,
                )
            )
            continue

        stt_score = _score_stt(
            confidence, stt_neighbor, stt_mono, residual, item.difference
        )
        meta_score = _score_metadata(
            confidence, meta_neighbor, meta_mono, residual
        )

        decision = "STT" if stt_score >= meta_score else "metadata"
        chosen_start = item.stt.start if decision == "STT" else item.metadata.start
        chosen_end = item.stt.end if decision == "STT" else item.metadata.end
        reason_parts = []

        if decision == "STT":
            if confidence >= HIGH_CONFIDENCE and stt_neighbor >= 0.7:
                reason_parts.append(
                    "High STT confidence and neighboring STT timestamps are consistent"
                )
            elif stt_mono and not meta_mono:
                reason_parts.append(
                    "Metadata violates chronological order, so STT was selected"
                )
            elif drift_detected and residual <= CONFLICT_THRESHOLD_SECONDS:
                reason_parts.append(
                    "Conflict is explained by systematic drift; STT is treated as "
                    "the audio ground truth"
                )
            else:
                reason_parts.append(
                    f"STT score {stt_score:.1f} beat metadata score {meta_score:.1f}"
                )
        else:
            if confidence <= LOW_CONFIDENCE and meta_neighbor >= 0.7:
                reason_parts.append(
                    "Low STT confidence and metadata neighbors are consistent"
                )
            elif meta_mono and not stt_mono:
                reason_parts.append(
                    "STT timestamp violates chronological order, so metadata was selected"
                )
            else:
                reason_parts.append(
                    f"Metadata score {meta_score:.1f} beat STT score {stt_score:.1f}"
                )

        # Reject a winner that would go backwards relative to the previous *final* time.
        if final_words and chosen_start + 1e-9 < final_words[-1].start:
            fallback = "metadata" if decision == "STT" else "STT"
            fallback_start = (
                item.metadata.start if fallback == "metadata" else item.stt.start
            )
            fallback_end = (
                item.metadata.end if fallback == "metadata" else item.stt.end
            )
            if fallback_start + 1e-9 >= final_words[-1].start:
                reason_parts.append(
                    f"{decision} was rejected because it would go backwards; "
                    f"using {fallback} instead"
                )
                decision = fallback
                chosen_start = fallback_start
                chosen_end = fallback_end
            else:
                chosen_start = final_words[-1].start + 0.05
                chosen_end = max(chosen_end, chosen_start + 0.05)
                decision = "ordering_repair"
                reason_parts.append(
                    "Both sources violated monotonic order versus the previous "
                    "final timestamp; a +0.05s repair was applied"
                )

        factors = {
            "stt_confidence": round(confidence, 4),
            "difference_seconds": round(item.difference, 4),
            "stt_neighbor_consistency": round(stt_neighbor, 4),
            "metadata_neighbor_consistency": round(meta_neighbor, 4),
            "stt_monotonic": stt_mono,
            "metadata_monotonic": meta_mono,
            "estimated_offset_seconds": round(offset, 4),
            "drift_detected": drift_detected,
            "residual_after_drift_seconds": round(residual, 4),
            "stt_score": round(stt_score, 2),
            "metadata_score": round(meta_score, 2),
        }

        conflict_number += 1
        conflicts.append(
            ConflictLog(
                number=conflict_number,
                word=item.word,
                metadata_timestamp=item.metadata.start,
                stt_timestamp=item.stt.start,
                difference=item.difference,
                stt_confidence=item.stt.confidence,
                factors=factors,
                decision=decision,
                reason="; ".join(reason_parts) + ".",
                final_timestamp=chosen_start,
            )
        )
        final_words.append(
            WordTimestamp(
                word=item.word,
                start=chosen_start,
                end=max(chosen_end, chosen_start + 0.05),
                confidence=item.stt.confidence,
                source=decision.lower(),
            )
        )

    if padding:
        if final_words:
            final_words[0].start = max(0.0, final_words[0].start - padding)
            final_words[-1].end = final_words[-1].end + padding

    return ReconciliationResult(
        aligned=aligned,
        conflicts=conflicts,
        final_words=final_words,
        estimated_offset=offset,
        offset_mad=offset_mad,
        drift_detected=drift_detected,
        notes=notes,
    )


def format_conflict_logs(conflicts: list[ConflictLog]) -> str:
    blocks = []
    for conflict in conflicts:
        confidence = (
            "n/a"
            if conflict.stt_confidence is None
            else f"{conflict.stt_confidence:.2f}"
        )
        factor_lines = "\n".join(
            f"  - {key}: {value}" for key, value in conflict.factors.items()
        )
        blocks.append(
            "\n".join(
                [
                    f"Conflict #{conflict.number}",
                    f'Word: "{conflict.word}"',
                    f"Metadata: {conflict.metadata_timestamp:.2f}s",
                    f"STT: {conflict.stt_timestamp:.2f}s",
                    f"Difference: {conflict.difference:.2f}s",
                    f"STT confidence: {confidence}",
                    "Factors considered:",
                    factor_lines,
                    f"Decision: {conflict.decision}",
                    f"Reason: {conflict.reason}",
                    f"Final timestamp: {conflict.final_timestamp:.2f}s",
                ]
            )
        )
    return "\n\n".join(blocks)
