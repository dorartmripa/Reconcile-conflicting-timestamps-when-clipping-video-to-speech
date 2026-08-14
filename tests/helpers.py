from __future__ import annotations

from src.models import WordTimestamp


def word(
    token: str,
    start: float,
    end: float | None = None,
    confidence: float | None = None,
    source: str = "test",
) -> WordTimestamp:
    return WordTimestamp(
        word=token,
        start=start,
        end=start + 0.3 if end is None else end,
        confidence=confidence,
        source=source,
    )


def paired(
    tokens: list[str],
    meta_starts: list[float],
    stt_starts: list[float],
    stt_conf: list[float],
) -> tuple[list[WordTimestamp], list[WordTimestamp]]:
    metadata = [
        word(token, start, source="metadata")
        for token, start in zip(tokens, meta_starts)
    ]
    stt = [
        word(token, start, confidence=conf, source="stt")
        for token, start, conf in zip(tokens, stt_starts, stt_conf)
    ]
    return metadata, stt
