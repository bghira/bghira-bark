# examples/batch.py
"""Bark batched-TTS demo with optional interactive mode for benchmarking.

Usage
-----
# One-shot (default):
$ python examples/batch.py -t "Hello|World|Goodbye" -v en_speaker_6

# One-shot with live playback:
$ python examples/batch.py -t "Hello world." -v en_speaker_6 --play

# Interactive REPL – models stay loaded between runs:
$ python examples/batch.py --interactive --play -v en_speaker_6

Inside the REPL, type prompts separated by | and press Enter.
Type 'quit' or Ctrl-D to exit.
"""
from __future__ import annotations

import argparse, io, logging, struct, sys, time
from pathlib import Path

import numpy as np
from pydub import AudioSegment, effects
from scipy.io.wavfile import write as write_wav

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
SAMPLE_WIDTH = 2  # 16-bit PCM


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


def _postprocess(audio_np, args):
    """Convert a float32 waveform to an AudioSegment with optional processing."""
    wav_buf = io.BytesIO()
    write_wav(wav_buf, SAMPLE_RATE, (audio_np * 32767).astype(np.int16))
    wav_buf.seek(0)
    snd = AudioSegment.from_wav(wav_buf)
    if args.normalize is not None:
        snd = normalise(snd, args.normalize)
    if args.compress:
        snd = compress(snd)
    return snd


# ---------------------------------------------------------------------------
# StreamingWavWriter
# ---------------------------------------------------------------------------

class StreamingWavWriter:
    """Append PCM segments to a WAV output, supporting both files and pipes.

    ``append()`` enqueues audio and returns immediately.  A background
    thread handles all file I/O (including blocking FIFO writes) so the
    generation loop is never stalled by slow consumers.

    For seekable files the RIFF/data sizes are patched after each write.
    For non-seekable outputs (FIFOs, pipes) the header uses
    ``0xFFFFFFFF`` sizes; most players handle this correctly.
    """

    def __init__(self, path: Path, sample_rate: int = SAMPLE_RATE, sample_width: int = SAMPLE_WIDTH):
        import os, stat, queue, threading
        self.path = path
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.n_channels = 1
        self.data_bytes_written = 0
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._finished = threading.Event()
        self._error: Exception | None = None

        # Open FIFOs with O_NONBLOCK so we fail fast if no reader is
        # connected instead of blocking the whole process indefinitely.
        if path.exists() and stat.S_ISFIFO(path.stat().st_mode):
            try:
                fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:
                raise RuntimeError(
                    f"{path} is a FIFO with no reader. Start a reader first "
                    f"(e.g. mplayer {path}) or remove the FIFO (rm {path})."
                )
            os.set_blocking(fd, True)
            self._fp = os.fdopen(fd, "wb")
        else:
            self._fp = open(path, "wb")
        self._seekable = self._fp.seekable()
        self._write_header()

        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()

    def _write_header(self):
        """Write a RIFF/WAV header."""
        size_placeholder = 0xFFFFFFFF if not self._seekable else 0
        self._fp.write(b"RIFF")
        self._fp.write(struct.pack("<I", size_placeholder))
        self._fp.write(b"WAVE")
        self._fp.write(b"fmt ")
        self._fp.write(struct.pack("<I", 16))
        self._fp.write(struct.pack("<H", 1))   # PCM
        self._fp.write(struct.pack("<H", self.n_channels))
        self._fp.write(struct.pack("<I", self.sample_rate))
        byte_rate = self.sample_rate * self.n_channels * self.sample_width
        self._fp.write(struct.pack("<I", byte_rate))
        block_align = self.n_channels * self.sample_width
        self._fp.write(struct.pack("<H", block_align))
        self._fp.write(struct.pack("<H", self.sample_width * 8))
        self._fp.write(b"data")
        self._fp.write(struct.pack("<I", size_placeholder))
        self._fp.flush()

    def _patch_sizes(self):
        if not self._seekable:
            return
        self._fp.seek(40)
        self._fp.write(struct.pack("<I", self.data_bytes_written))
        self._fp.seek(4)
        self._fp.write(struct.pack("<I", 36 + self.data_bytes_written))
        self._fp.seek(0, 2)
        self._fp.flush()

    def _drain_loop(self):
        """Background thread: pull from queue and write to file."""
        try:
            while True:
                chunk = self._q.get()
                if chunk is None:
                    break
                pcm = (chunk * 32767).astype(np.int16).tobytes()
                self._fp.write(pcm)
                self._fp.flush()
                self.data_bytes_written += len(pcm)
                self._patch_sizes()
        except Exception as exc:
            self._error = exc
        finally:
            self._finished.set()

    def append(self, audio_np: np.ndarray):
        self._q.put(audio_np)

    @property
    def duration(self) -> float:
        return self.data_bytes_written / (self.sample_rate * self.sample_width * self.n_channels)

    def close(self):
        self._q.put(None)
        self._finished.wait()
        self._patch_sizes()
        self._fp.close()
        if self._error is not None:
            raise self._error


# ---------------------------------------------------------------------------
# StreamingAudioPlayer
# ---------------------------------------------------------------------------

class StreamingAudioPlayer:
    """Play audio segments to the desktop speakers via sounddevice.

    ``append()`` enqueues samples into an internal buffer and returns
    immediately.  A callback-driven ``OutputStream`` pulls from the
    buffer in the background.  Call ``drain()`` to block until all
    queued audio has finished playing.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        import queue, threading
        import sounddevice as sd
        self._sd = sd
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._finished = threading.Event()
        self._done_feeding = False
        self.samples_written = 0

        def _callback(outdata, frames, _time, status):
            with self._lock:
                # pull from queue into local buffer as needed
                while len(self._buf) < frames:
                    try:
                        chunk = self._q.get_nowait()
                    except queue.Empty:
                        break
                    if chunk is None:
                        # sentinel: no more data coming
                        self._done_feeding = True
                        break
                    self._buf = np.concatenate([self._buf, chunk])

                if len(self._buf) >= frames:
                    outdata[:, 0] = self._buf[:frames]
                    self._buf = self._buf[frames:]
                else:
                    # underrun: play what we have, zero-pad the rest
                    n = len(self._buf)
                    outdata[:n, 0] = self._buf
                    outdata[n:, 0] = 0.0
                    self._buf = np.zeros(0, dtype=np.float32)
                    if self._done_feeding:
                        self._finished.set()

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=2048,
            callback=_callback,
        )
        self._stream.start()

    def append(self, audio_np: np.ndarray):
        self._q.put(audio_np.astype(np.float32).ravel())
        self.samples_written += len(audio_np)

    def drain(self):
        """Block until all queued audio has finished playing."""
        self._q.put(None)  # sentinel
        self._finished.wait()

    @property
    def duration(self) -> float:
        return self.samples_written / SAMPLE_RATE

    def close(self):
        self.drain()
        self._stream.stop()
        self._stream.close()


# ---------------------------------------------------------------------------
# export helpers
# ---------------------------------------------------------------------------

def _out_path(args, label: str = "") -> Path:
    fname, fext = Path(args.out).stem, Path(args.out).suffix
    suffix = f"_{label}" if label else ""
    return Path(f"{fname}{suffix}{fext}")


def export_one(audio_np, args, *, label=""):
    snd = _postprocess(audio_np, args)
    out_path = _out_path(args, label)
    fmt = "wav" if out_path.suffix.lower() == ".wav" else "mp3"
    snd.export(out_path, format=fmt)
    log.info("Saved %s (%.2f s)", out_path, snd.duration_seconds)
    return out_path


def export_batch(audio_np_batch, args, *, run_label=""):
    for idx, audio_np in enumerate(audio_np_batch, 1):
        label = f"{run_label}_{idx}" if run_label else str(idx)
        export_one(audio_np, args, label=label)


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------

def run_batch(prompts, voice, args, *, run_label="", player=None):
    """Run one generation pass and return the audio list.

    When a single prompt is given, uses streaming long-form generation:
    audio segments are appended to the output as they finish — both to
    the WAV file and (with --play) directly to the speakers.

    An optional *player* (``StreamingAudioPlayer``) can be passed in so
    it persists across calls — audio from successive runs queues up
    without blocking generation.
    """
    import torch
    from bark import generate_audio_batched, generate_audio_long

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start = time.time()
    if len(prompts) == 1:
        # Single prompt: stream segments
        out_path = Path(args.out)
        writer = StreamingWavWriter(out_path)
        log.info("Streaming long-form generation → %s%s",
                 out_path, " (+play)" if player else "")
        gen = generate_audio_long(
            prompts[0],
            history_prompt=voice,
            max_tokens=args.max_tokens,
            stream=True,
            segment_tokens=args.segment_tokens,
        )
        audio_np_batch = []
        total_samples = 0
        for idx, audio_np in enumerate(gen):
            audio_np_batch.append(audio_np)
            writer.append(audio_np)
            if player is not None:
                player.append(audio_np)
            total_samples += len(audio_np)
            dur_s = len(audio_np) / SAMPLE_RATE
            log.info(
                "  segment %d ready: %.2f s audio → %s (%.2f s total)",
                idx, dur_s, out_path, total_samples / SAMPLE_RATE,
            )
        writer.close()
        log.info("Saved %s (%.2f s)", out_path, total_samples / SAMPLE_RATE)
    else:
        # Multiple explicit prompts: existing batched behaviour
        audio_np_batch = generate_audio_batched(
            prompts,
            history_prompt=voice,
            allow_early_stop=not args.no_early_stop,
            min_eos_p=args.min_eos_p,
        )
    elapsed = time.time() - start

    total_dur = sum(len(a) / SAMPLE_RATE for a in audio_np_batch)
    log.info(
        "Done: %d piece(s), %.2f s total audio in %.2f s wall-clock",
        len(audio_np_batch), total_dur, elapsed,
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
        "--play", action="store_true",
        help="Stream audio to speakers via sounddevice as it generates.",
    )
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
    p.add_argument(
        "--segment-tokens", type=int, default=150,
        help="Max semantic tokens per streaming sub-segment (default 150, ~3 s).",
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

    # ── pre-warm the BERT tokenizer used by long-form text splitting ────
    from bark.long import _get_tokenizer
    _get_tokenizer()

    # ── warmup runs (trigger torch.compile) ──────────────────────────────
    # Call generate_audio_batched directly for batch=1 and batch=2 so
    # compiled graphs are warm for both sizes.  This avoids the streaming
    # path (files, audio player) that run_batch uses for single prompts.
    from bark import generate_audio_batched
    log.info("Warmup (triggers torch.compile) …")
    t0 = time.time()
    log.info("  warmup batch=1 …")
    generate_audio_batched(["warmup."], history_prompt=args.voice)
    log.info("  warmup batch=2 …")
    generate_audio_batched(["warmup one.", "warmup two."], history_prompt=args.voice)
    log.info("Warmup done in %.1f s", time.time() - t0)

    # ── create a shared player (lives across all runs) ──────────────────
    player = StreamingAudioPlayer() if args.play else None

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
            audio = run_batch(
                prompts, args.voice, args,
                run_label=f"r{run_num}", player=player,
            )
            if len(prompts) > 1:
                export_batch(audio, args, run_label=f"r{run_num}")
    else:
        # ── one-shot mode ─────────────────────────────────────────────────
        prompts = [s.strip() for s in args.text.split("|") if s.strip()]
        if not prompts:
            log.error("No prompts provided")
            return 1
        audio = run_batch(prompts, args.voice, args, player=player)
        if len(prompts) > 1:
            export_batch(audio, args)

    if player is not None:
        player.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
