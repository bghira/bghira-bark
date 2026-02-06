from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .generation import (
    generate_text_semantic_batched,
    generate_coarse_batched,
    generate_fine_batched,
    codec_decode_batched,
)

logger = logging.getLogger(__name__)

__all__ = ["generate_audio_batched"]


# ─────────────────────────────────────────────────────────────────────────────
def generate_audio_batched(
    prompts: List[str],
    history_prompt: Optional[Union[str, Dict, Sequence[Union[str, Dict]]]] = None,
    *,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    sliding_window_len: int = 60,
    min_eos_p: float = 0.2,
    allow_early_stop: bool = True,
    silent: bool = False,
    output_full: bool = False,
) -> list[np.ndarray] | list[tuple[Dict[str, np.ndarray], np.ndarray]]:
    """
    Truly batched audio generation for a list of prompts.

    Every stage (semantic, coarse, fine, codec) runs with batch > 1 when
    possible, giving much better GPU utilisation than sequential calls.

    Parameters
    ----------
    prompts
        List of prompt strings.
    history_prompt
        Either a single history preset (applied to all), **or** a list
        matching ``prompts``.
    text_temp / waveform_temp
        Temperature for text→semantic and coarse/fine stages respectively.
    output_full
        If True, return ``(full_token_dict, audio)`` tuples.

    Returns
    -------
    List of numpy float32 waveforms (24 kHz) — or list of
    ``(full, audio)`` tuples when ``output_full=True``.
    """
    if not prompts:
        return []

    B = len(prompts)

    # normalise history to a list of length B ─────────────────────────────
    if isinstance(history_prompt, (str, dict)) or history_prompt is None:
        history_lst = [history_prompt] * B
    else:
        if len(history_prompt) != B:
            raise ValueError("`history_prompt` length must match `prompts`")
        history_lst = list(history_prompt)

    # ── Stage 1: batched semantic ────────────────────────────────────────
    logger.info("Stage 1/4: semantic (batch=%d)", B)
    sem_toks = generate_text_semantic_batched(
        prompts,
        history_prompts=history_lst,
        temp=text_temp,
        silent=silent,
        min_eos_p=min_eos_p,
        allow_early_stop=allow_early_stop,
    )

    # ── Stage 2: batched coarse ──────────────────────────────────────────
    logger.info("Stage 2/4: coarse (batch=%d)", B)
    coarse_toks = generate_coarse_batched(
        sem_toks,
        history_prompts=history_lst,
        temp=waveform_temp,
        silent=silent,
        sliding_window_len=sliding_window_len,
    )

    # ── Stage 3: batched fine ────────────────────────────────────────────
    logger.info("Stage 3/4: fine (batch=%d)", B)
    fine_toks = generate_fine_batched(
        coarse_toks,
        history_prompts=history_lst,
        temp=waveform_temp,
        silent=silent,
    )

    # ── Stage 4: batched codec decode ────────────────────────────────────
    logger.info("Stage 4/4: codec decode (batch=%d)", B)
    audio_list = codec_decode_batched(fine_toks)

    # ── assemble output ──────────────────────────────────────────────────
    if output_full:
        results = []
        for i in range(B):
            full = {
                "semantic_prompt": sem_toks[i],
                "coarse_prompt": coarse_toks[i],
                "fine_prompt": fine_toks[i],
            }
            results.append((full, audio_list[i]))
        return results

    return audio_list
