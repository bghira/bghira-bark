# examples/batch.py
"""Bark batched-TTS demo with optional interactive mode for benchmarking.

Usage
-----
# One-shot (default):
$ python examples/batch.py -t "Hello|World|Goodbye" -v en_speaker_6

# Interactive REPL – models stay loaded between runs:
$ python examples/batch.py --interactive -v en_speaker_6

Inside the REPL, type prompts separated by | and press Enter.
Type 'quit' or Ctrl-D to exit.
"""
from __future__ import annotations

import argparse, io, logging, sys, time
from pathlib import Path

import numpy as np
from pydub import AudioSegment, effects
from scipy.io.wavfile import write as write_wav

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def normalise(sound: AudioSegment, target_dbfs: float = -14.0) -> AudioSegment:
    change = target_dbfs - sound.dBFS
    return sound.apply_gain(change)


def compress(sound: AudioSegment) -> AudioSegment:
    return effects.compress_dynamic_range(
        sound, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0,
    )


def export_batch(audio_np_batch, args, *, run_label=""):
    """Write a list of numpy waveforms to disk."""
    for idx, audio_np in enumerate(audio_np_batch, 1):
        wav_buf = io.BytesIO()
        write_wav(wav_buf, 24_000, (audio_np * 32767).astype(np.int16))
        wav_buf.seek(0)
        snd = AudioSegment.from_wav(wav_buf)

        if args.normalize is not None:
            snd = normalise(snd, args.normalize)
        if args.compress:
            snd = compress(snd)

        fname, fext = Path(args.out).stem, Path(args.out).suffix
        suffix = f"_{run_label}" if run_label else ""
        out_path = Path(f"{fname}{suffix}_{idx}{fext}")
        fmt = "wav" if out_path.suffix.lower() == ".wav" else "mp3"
        snd.export(out_path, format=fmt)
        log.info("Saved %s (%.2f s)", out_path, snd.duration_seconds)


def run_batch(prompts, voice, args):
    """Run one batched generation and print timing info. Returns the audio list."""
    import torch
    from bark import generate_audio_batched, generate_audio_long

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start = time.time()
    if len(prompts) == 1:
        # Single prompt: use generate_audio_long which auto-splits long text
        audio_arr = generate_audio_long(
            prompts[0],
            history_prompt=voice,
            max_tokens=args.max_tokens,
        )
        audio_np_batch = [audio_arr]
    else:
        # Multiple explicit prompts: keep existing batched behaviour
        audio_np_batch = generate_audio_batched(
            prompts,
            history_prompt=voice,
            allow_early_stop=not args.no_early_stop,
            min_eos_p=args.min_eos_p,
        )
    elapsed = time.time() - start

    for idx, audio_np in enumerate(audio_np_batch, 1):
        dur_s = len(audio_np) / 24_000
        log.info(
            "  [%d/%d] %.2f s audio, %d samples",
            idx, len(audio_np_batch), dur_s, len(audio_np),
        )
    log.info(
        "Batch of %d done in %.2f s (%.2f s/item)",
        len(prompts), elapsed, elapsed / len(prompts),
    )
    return audio_np_batch


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Bark batched TTS demo")
    p.add_argument("-t", "--text", default=None, help="Prompts separated by |")
    p.add_argument("-v", "--voice", default="en_speaker_1", help="Bark voice preset")
    p.add_argument("--out", default="out.wav", help="Output file (.mp3 or .wav)")
    p.add_argument("--normalize", type=float, default=None, metavar="LUFS")
    p.add_argument("--compress", action="store_true")
    p.add_argument("--seed", type=int, default=1, help="RNG seed")
    p.add_argument(
        "--interactive", action="store_true",
        help="Enter a REPL after loading models. Keeps compiled models warm.",
    )
    p.add_argument(
        "--no-early-stop", action="store_true",
        help="Disable early stopping so generation runs until max steps.",
    )
    p.add_argument(
        "--min-eos-p", type=float, default=0.2,
        help="Min P(eos) to trigger early stop (default 0.2). Lower = longer.",
    )
    p.add_argument(
        "--max-tokens", type=int, default=75,
        help="Max BERT tokens per chunk for long-form splitting (default 75).",
    )
    args = p.parse_args()

    if not args.interactive and args.text is None:
        p.error("--text is required unless --interactive is set")

    # ── load & compile models once ────────────────────────────────────────
    from bark.generation import preload_models
    log.info("Loading models …")
    t0 = time.time()
    preload_models()
    log.info("Models loaded in %.1f s", time.time() - t0)

    # ── warmup run (triggers torch.compile) ───────────────────────────────
    # Use 2 prompts so compiled graphs are warm for batch>1 (avoids
    # recompilation on the first real batched run).
    warmup_prompts = ["warmup one.", "warmup two."]
    log.info("Warmup run (triggers torch.compile) …")
    t0 = time.time()
    run_batch(warmup_prompts, args.voice, args)
    log.info("Warmup done in %.1f s", time.time() - t0)

    if args.interactive:
        # ── REPL ──────────────────────────────────────────────────────────
        run_num = 0
        print("\n── Interactive mode ──")
        print("Enter prompts separated by |   (e.g. Hello|World|Goodbye)")
        print("Type 'quit' or press Ctrl-D to exit.\n")
        while True:
            try:
                line = input("bark> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line.lower() in ("quit", "exit", "q"):
                break

            prompts = [s.strip() for s in line.split("|") if s.strip()]
            if not prompts:
                continue

            run_num += 1
            log.info("Run %d: %d prompt(s)", run_num, len(prompts))
            audio = run_batch(prompts, args.voice, args)
            export_batch(audio, args, run_label=f"r{run_num}")
    else:
        # ── one-shot mode ─────────────────────────────────────────────────
        prompts = [s.strip() for s in args.text.split("|") if s.strip()]
        if not prompts:
            log.error("No prompts provided")
            return 1
        audio = run_batch(prompts, args.voice, args)
        export_batch(audio, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
