"""Auto-split long text and generate audio via batched parallel pipeline."""
from __future__ import annotations

import logging
import re

import numpy as np
from transformers import BertTokenizer

from .batching import generate_audio_batched

logger = logging.getLogger(__name__)

_tokenizer: BertTokenizer | None = None


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


def generate_audio_long(
    text: str,
    history_prompt=None,
    *,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
    max_tokens: int = 75,
) -> np.ndarray:
    """Generate audio from arbitrarily long text.

    The text is automatically split into chunks that fit within Bark's
    256-token semantic input window, all chunks are processed in parallel
    via ``generate_audio_batched``, and the resulting waveforms are
    concatenated in order.

    Returns a single float32 numpy array at 24 kHz.
    """
    chunks = split_text(text, max_tokens)
    if not chunks:
        raise ValueError("Text is empty after normalisation")

    logger.info("Long-form: %d chunk(s) from input text", len(chunks))
    for i, c in enumerate(chunks):
        logger.info("  chunk %d: %r", i, c[:80] + ("…" if len(c) > 80 else ""))

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
