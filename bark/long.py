"""Auto-split long text and generate audio via batched parallel pipeline."""
from __future__ import annotations

import logging
import re
from typing import Generator, Union

import numpy as np
from transformers import BertTokenizer

from .batching import generate_audio_batched
from .generation import (
    generate_text_semantic_batched,
    generate_coarse_batched,
    generate_fine_batched,
    codec_decode_batched,
)

logger = logging.getLogger(__name__)

_tokenizer: BertTokenizer | None = None

# Default number of semantic tokens per streaming sub-segment.
# At 49.9 Hz this is ~3 seconds of audio — short enough for low TTFA,
# long enough for decent prosody within each segment.
DEFAULT_SEGMENT_TOKENS = 150


def _get_tokenizer() -> BertTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    return _tokenizer


def _token_len(text: str) -> int:
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


def split_text(text: str, max_tokens: int = 75) -> list[str]:
    """Split *text* into chunks that each fit within *max_tokens* BERT tokens.

    Splits on sentence boundaries (`.!?` followed by whitespace) greedily,
    packing as many sentences as possible per chunk. If a single sentence
    exceeds the limit, it is split at the last whitespace before the limit.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _token_len(sent)

        if sent_tokens > max_tokens:
            # flush anything accumulated so far
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            # split oversized sentence at whitespace boundaries
            words = sent.split()
            part: list[str] = []
            for word in words:
                candidate = " ".join(part + [word])
                if _token_len(candidate) > max_tokens and part:
                    chunks.append(" ".join(part))
                    part = [word]
                else:
                    part.append(word)
            if part:
                chunks.append(" ".join(part))
            continue

        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_semantic_tokens(
    tokens: np.ndarray, max_tokens: int = DEFAULT_SEGMENT_TOKENS,
) -> list[np.ndarray]:
    """Split a 1-D semantic token array into sub-segments."""
    if len(tokens) <= max_tokens:
        return [tokens]
    return [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]


def _log_chunks(chunks: list[str]) -> None:
    logger.info("Long-form: %d chunk(s) from input text", len(chunks))
    for i, c in enumerate(chunks):
        logger.info("  chunk %d: %r", i, c[:80] + ("…" if len(c) > 80 else ""))


def generate_audio_long(
    text: str,
    history_prompt=None,
    *,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
    max_tokens: int = 75,
    stream: bool = False,
    segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
) -> Union[np.ndarray, Generator[np.ndarray, None, None]]:
    """Generate audio from arbitrarily long text.

    The text is automatically split into chunks that fit within Bark's
    256-token semantic input window.

    Parameters
    ----------
    stream : bool
        If False (default), returns a single concatenated float32 numpy
        array at 24 kHz.

        If True, returns a **generator** that yields one ``np.ndarray``
        per audio segment.  Semantic tokens are generated for all text
        chunks in one batched pass, then split into sub-segments of
        *segment_tokens* semantic tokens (~3 s each at default 150).
        The first sub-segment is decoded immediately for low
        time-to-first-audio; remaining sub-segments are batched through
        coarse → fine → codec and yielded in order.

    segment_tokens : int
        Maximum semantic tokens per streaming sub-segment (default 150,
        ~3 s of audio at 49.9 Hz).  Only used when ``stream=True``.
    """
    chunks = split_text(text, max_tokens)
    if not chunks:
        raise ValueError("Text is empty after normalisation")

    _log_chunks(chunks)

    if stream:
        return _stream_chunks(
            chunks,
            history_prompt=history_prompt,
            text_temp=text_temp,
            waveform_temp=waveform_temp,
            silent=silent,
            segment_tokens=segment_tokens,
        )

    audio_list = generate_audio_batched(
        chunks,
        history_prompt=history_prompt,
        text_temp=text_temp,
        waveform_temp=waveform_temp,
        silent=silent,
    )
    if len(audio_list) == 1:
        return audio_list[0]
    return np.concatenate(audio_list)


def _stream_chunks(
    chunks: list[str],
    *,
    history_prompt,
    text_temp: float,
    waveform_temp: float,
    silent: bool,
    segment_tokens: int,
) -> Generator[np.ndarray, None, None]:
    """Yield audio with sub-segment streaming.

    1. Semantic runs batched for ALL text chunks (one text-model load).
    2. The resulting semantic token arrays are split into sub-segments
       of *segment_tokens* tokens each (~3 s of audio).
    3. The first sub-segment is pushed through coarse → fine → codec
       immediately and yielded for lowest time-to-first-audio.
    4. Remaining sub-segments are batched through coarse → fine → codec
       and yielded in order.
    """
    B = len(chunks)
    history_lst = [history_prompt] * B

    # ── Stage 1: semantic for ALL text chunks (one model load) ───────
    logger.info("Streaming 1/4: semantic (batch=%d)", B)
    sem_toks_per_chunk = generate_text_semantic_batched(
        chunks, history_prompts=history_lst, temp=text_temp, silent=silent,
    )

    # ── Split semantic tokens into sub-segments ──────────────────────
    segments: list[np.ndarray] = []
    for sem in sem_toks_per_chunk:
        segments.extend(_split_semantic_tokens(sem, segment_tokens))

    total = len(segments)
    logger.info(
        "Streaming: %d semantic segment(s) from %d text chunk(s) "
        "(~%d tokens each)",
        total, B, segment_tokens,
    )

    # ── First segment: solo decode for lowest TTFA ───────────────────
    logger.info("Streaming 2–4/4: coarse→fine→codec for segment 0/%d", total)
    coarse = generate_coarse_batched(
        [segments[0]], history_prompts=[history_prompt],
        temp=waveform_temp, silent=silent,
    )
    fine = generate_fine_batched(
        coarse, history_prompts=[history_prompt],
        temp=waveform_temp, silent=silent,
    )
    audio = codec_decode_batched(fine)
    yield audio[0]

    # ── Remaining segments: batched decode ───────────────────────────
    if total > 1:
        rest = total - 1
        logger.info(
            "Streaming 2–4/4: coarse→fine→codec for segments 1–%d (batch=%d)",
            total - 1, rest,
        )
        hp_rest = [history_prompt] * rest
        coarse = generate_coarse_batched(
            segments[1:], history_prompts=hp_rest,
            temp=waveform_temp, silent=silent,
        )
        fine = generate_fine_batched(
            coarse, history_prompts=hp_rest,
            temp=waveform_temp, silent=silent,
        )
        audio_rest = codec_decode_batched(fine)
        for a in audio_rest:
            yield a
