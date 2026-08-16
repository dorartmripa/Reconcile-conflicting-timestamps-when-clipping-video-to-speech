"""Deterministic timestamp reconciliation.

Score weights are not learned. They were chosen so STT confidence is the
largest term, then local consistency, then order, then drift residual.

Evaluated on sample/labeled_conflicts.json (hand-labeled winners):
the policy beats always-STT because low-confidence rows can choose metadata.
See tests/test_labeled_policy.py.
"""

from __future__ import annotations

from statistics import median

from backend.models import (
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
DRIFT_SLOPE_MIN = 0.02
SCORE_TIE_MARGIN = 3.0

# STT: confidence, neighbors, monotonicity, drift residual, huge-gap penalty
STT_CONFIDENCE_WEIGHT = 40.0
STT_NEIGHBOR_WEIGHT = 25.0
STT_MONOTONIC_BONUS = 15.0
STT_DRIFT_RESIDUAL_BONUS = 10.0
STT_HUGE_GAP_PENALTY = 5.0

# Metadata: (1 - STT confidence), neighbors, monotonicity, drift residual
META_LOW_CONF_WEIGHT = 35.0
META_NEIGHBOR_WEIGHT = 25.0
META_MONOTONIC_BONUS = 15.0
META_DRIFT_RESIDUAL_BONUS = 12.0


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


def estimate_linear_drift(
    aligned: list[AlignedWord],
) -> tuple[float, float, float]:
    """Least squares: (STT - metadata) ≈ offset + slope * metadata_time."""

    points = [
        (item.metadata.start, item.stt.start - item.metadata.start)
        for item in aligned
        if (item.stt.confidence or 0.0) >= HIGH_CONFIDENCE
    ]
    if len(points) < 3:
        offset, mad = estimate_offset(aligned)
        return offset, 0.0, mad

    times = [point[0] for point in points]
    deltas = [point[1] for point in points]
    mean_t = sum(times) / len(times)
    mean_d = sum(deltas) / len(deltas)
    denom = sum((time - mean_t) ** 2 for time in times)
    if denom < 1e-12:
        offset, mad = estimate_offset(aligned)
        return offset, 0.0, mad

    slope = sum((time - mean_t) * (delta - mean_d) for time, delta in points) / denom
    offset = mean_d - slope * mean_t
    residuals = [delta - (offset + slope * time) for time, delta in points]
    mad = float(median([abs(item) for item in residuals]))
    return float(offset), float(slope), mad


def _corrected_metadata_start(meta_start: float, offset: float, slope: float) -> float:
    return meta_start + offset + slope * meta_start


def neighbor_consistency(words: list[WordTimestamp], index: int) -> float:
    """Score in [0, 1] using order, gaps, and overlap with neighbor durations."""

    score = 1.0
    current = words[index]
    if index > 0:
        previous = words[index - 1]
        if current.start < previous.start:
            score -= 0.6
        elif current.start < previous.end - 1e-6:
            score -= 0.25
        elif current.start - previous.start > 3.0:
            score -= 0.25
        elif current.start - previous.start < 0.02:
            score -= 0.15
    if index < len(words) - 1:
        following = words[index + 1]
        if current.start > following.start:
            score -= 0.6
        elif current.end > following.start + 1e-6:
            score -= 0.25
        elif following.start - current.start > 3.0:
            score -= 0.25
        elif following.start - current.start < 0.02:
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
    score += confidence * STT_CONFIDENCE_WEIGHT
    score += neighbor * STT_NEIGHBOR_WEIGHT
    score += STT_MONOTONIC_BONUS if monotonic else 0.0
    if residual_after_drift <= CONFLICT_THRESHOLD_SECONDS:
        score += STT_DRIFT_RESIDUAL_BONUS
    if difference > 2.0:
        score -= STT_HUGE_GAP_PENALTY
    return score


def _score_metadata(
    confidence: float,
    neighbor: float,
    monotonic: bool,
    residual_after_drift: float,
) -> float:
    score = 0.0
    score += (1.0 - confidence) * META_LOW_CONF_WEIGHT
    score += neighbor * META_NEIGHBOR_WEIGHT
    score += META_MONOTONIC_BONUS if monotonic else 0.0
    if residual_after_drift <= CONFLICT_THRESHOLD_SECONDS:
        score += META_DRIFT_RESIDUAL_BONUS
    return score


def _apply_padding(words: list[WordTimestamp], padding: float) -> None:
    trusted = [word for word in words if word.trusted] or words
    if not trusted or padding <= 0:
        return
    trusted[0].start = max(0.0, trusted[0].start - padding)
    trusted[-1].end = trusted[-1].end + padding
    if trusted[-1].end <= trusted[0].start:
        trusted[-1].end = trusted[0].start + 0.05


def _fill_untrusted(final_words: list[WordTimestamp]) -> None:
    trusted_indexes = [i for i, word in enumerate(final_words) if word.trusted]
    for i, word in enumerate(final_words):
        if word.trusted:
            continue
        prev_i = next((idx for idx in reversed(trusted_indexes) if idx < i), None)
        next_i = next((idx for idx in trusted_indexes if idx > i), None)
        if prev_i is not None and next_i is not None:
            gap = next_i - prev_i
            frac = (i - prev_i) / gap
            span = final_words[next_i].start - final_words[prev_i].start
            word.start = final_words[prev_i].start + frac * span
            word.end = max(word.start + 0.05, word.start + word.duration)
        elif prev_i is not None:
            word.start = final_words[prev_i].end + 0.04
            word.end = word.start + 0.25
        elif next_i is not None:
            word.end = final_words[next_i].start
            word.start = max(0.0, word.end - 0.25)
        word.source = "untrusted"


def reconcile(
    aligned: list[AlignedWord],
    padding: float = 0.0,
) -> ReconciliationResult:
    stt_words = [item.stt for item in aligned]
    meta_words = [item.metadata for item in aligned]
    stt_times = [item.start for item in stt_words]
    meta_times = [item.start for item in meta_words]
    offset, slope, offset_mad = estimate_linear_drift(aligned)
    drift_detected = offset_mad <= DRIFT_MAD_LIMIT and (
        abs(offset) >= DRIFT_OFFSET_MIN or abs(slope) >= DRIFT_SLOPE_MIN
    )

    notes = [
        f"Estimated drift (STT - metadata) ≈ {offset:+.3f}s + {slope:+.4f} * t "
        f"(residual MAD {offset_mad:.3f}s)."
    ]
    if drift_detected:
        notes.append(
            "Systematic drift detected. Drift-corrected metadata is used as a "
            "factor, but raw conflicts still count."
        )

    conflicts: list[ConflictLog] = []
    final_words: list[WordTimestamp] = []
    conflict_number = 0

    for item in aligned:
        confidence = 0.0 if item.stt.confidence is None else item.stt.confidence
        corrected = _corrected_metadata_start(item.metadata.start, offset, slope)
        residual = abs(item.stt.start - corrected)
        stt_neighbor = neighbor_consistency(stt_words, item.index)
        meta_neighbor = neighbor_consistency(meta_words, item.index)
        stt_mono = is_monotonic(stt_times, item.index, item.stt.start)
        meta_mono = is_monotonic(meta_times, item.index, item.metadata.start)
        interpolated = item.stt.interpolated or item.stt.source == "whisper_interpolated"

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
                    interpolated=interpolated,
                )
            )
            continue

        stt_score = _score_stt(
            confidence, stt_neighbor, stt_mono, residual, item.difference
        )
        meta_score = _score_metadata(
            confidence, meta_neighbor, meta_mono, residual
        )

        if abs(stt_score - meta_score) < SCORE_TIE_MARGIN:
            decision = "blend"
            chosen_start = (item.stt.start + item.metadata.start) / 2.0
            chosen_end = (item.stt.end + item.metadata.end) / 2.0
            reason_parts = [
                f"Near-tie (STT {stt_score:.1f} vs metadata {meta_score:.1f}); blended"
            ]
        elif stt_score > meta_score:
            decision = "STT"
            chosen_start = item.stt.start
            chosen_end = item.stt.end
            reason_parts = []
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
            decision = "metadata"
            chosen_start = item.metadata.start
            chosen_end = item.metadata.end
            reason_parts = []
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

        trusted = True
        if final_words and chosen_start + 1e-9 < final_words[-1].start:
            fallback = "metadata" if decision in {"STT", "blend"} else "STT"
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
                decision = "untrusted"
                trusted = False
                reason_parts.append(
                    "Both sources violate order versus the previous final timestamp; "
                    "this word is untrusted and interpolated from trusted neighbors"
                )

        factors = {
            "stt_confidence": round(confidence, 4),
            "difference_seconds": round(item.difference, 4),
            "end_difference_seconds": round(item.end_difference, 4),
            "stt_neighbor_consistency": round(stt_neighbor, 4),
            "metadata_neighbor_consistency": round(meta_neighbor, 4),
            "stt_monotonic": stt_mono,
            "metadata_monotonic": meta_mono,
            "estimated_offset_seconds": round(offset, 4),
            "estimated_slope": round(slope, 6),
            "drift_detected": drift_detected,
            "residual_after_drift_seconds": round(residual, 4),
            "stt_score": round(stt_score, 2),
            "metadata_score": round(meta_score, 2),
            "interpolated_stt": interpolated,
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
                final_timestamp=0.0 if not trusted else chosen_start,
            )
        )
        final_words.append(
            WordTimestamp(
                word=item.word,
                start=chosen_start if trusted else (
                    final_words[-1].start if final_words else 0.0
                ),
                end=max(chosen_end, (chosen_start if trusted else 0.0) + 0.05),
                confidence=item.stt.confidence,
                source=decision.lower(),
                trusted=trusted,
                interpolated=interpolated,
            )
        )

    _fill_untrusted(final_words)
    conflict_iter = iter(conflicts)
    for item in aligned:
        if not item.is_conflict:
            continue
        conflict = next(conflict_iter)
        conflict.final_timestamp = final_words[item.index].start

    _apply_padding(final_words, padding)

    return ReconciliationResult(
        aligned=aligned,
        conflicts=conflicts,
        final_words=final_words,
        estimated_offset=offset,
        offset_mad=offset_mad,
        drift_detected=drift_detected,
        estimated_slope=slope,
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
