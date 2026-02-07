# 🐶 Bark
## 🚀 Updates

**2026.02**
- **Long-form generation** (`generate_audio_long`): pass arbitrarily long text and it auto-splits at sentence boundaries (using BERT tokenizer counts), then runs all chunks through the batched pipeline
- **Streaming output**: `generate_audio_long(stream=True)` returns a generator that yields audio segments as they decode, giving low time-to-first-audio
  - Semantic tokens are generated for all text chunks in one batched pass, split into ~3 s sub-segments, then streamed through coarse/fine/codec
- **Native audio playback**: `--play` flag streams audio directly to desktop speakers via `sounddevice` as it generates (no external tools needed)
- **Interactive REPL**: `--interactive` keeps compiled models warm between runs for fast iteration
- **Non-blocking I/O**: file writes and audio playback run on background threads; generation is never stalled by slow consumers (including FIFOs)
- **True batched pipeline**: all four stages (semantic, coarse, fine, codec) now run with batch > 1 natively instead of sequential-per-prompt
- **Performance**: top-p filtering runs entirely on GPU (no CPU round-trip), removed unnecessary `cuda.synchronize()` calls, vectorised codebook flattening

**2025.05.17**
- MultiGPU inference example added to run a large script and combine the outputs at the end
- Uses torch.compile with max-autotune across all models by default
- Uses padded attention masks to support batched inference
- SageAttention will be used automatically if installed
  - Compatible with batch inference and torch compile

On a 3x 4090 system, this can bring long inference job runtimes down from 2-5 minutes to 30-60 seconds.

Bark is licensed under the MIT License, meaning it's now available for commercial use!

## Installation

```bash
git clone https://github.com/bghira/bghira-bark
cd bghira-bark
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e .
pip install sounddevice  # optional, for --play
```

## Usage

### Single-GPU batched TTS

```bash
# One-shot generation (long text auto-splits):
python examples/batch.py -t "Your long text here..." -v en_speaker_6

# With live audio playback:
python examples/batch.py -t "Hello world." -v en_speaker_6 --play

# Interactive REPL (models stay compiled between runs):
python examples/batch.py --interactive --play -v en_speaker_6

# Multiple explicit prompts (batched):
python examples/batch.py -t "Hello|World|Goodbye" -v en_speaker_6
```

### Long-form generation API

```python
from bark import generate_audio_long
from bark.generation import SAMPLE_RATE, preload_models

preload_models()

# Returns a single concatenated numpy array:
audio = generate_audio_long("Your long text here...", history_prompt="en_speaker_6")

# Or stream segments as they decode:
for segment in generate_audio_long("Long text...", history_prompt="en_speaker_6", stream=True):
    # each segment is ~3 s of float32 audio at 24 kHz
    pass
```

### MultiGPU inference

This will run the example across all available GPUs without invoking torch compile:

```bash
env SUNO_DISABLE_COMPILE=true accelerate launch examples/parallel.py --out out.mp3 --normalize -14 --compress
```

## ⚙️ Details

Bark is fully generative tex-to-audio model devolved for research and demo purposes. It follows a GPT style architecture similar to [AudioLM](https://arxiv.org/abs/2209.03143) and [Vall-E](https://arxiv.org/abs/2301.02111) and a quantized Audio representation from [EnCodec](https://github.com/facebookresearch/encodec). It is not a conventional TTS model, but instead a fully generative text-to-audio model capable of deviating in unexpected ways from any given script. Different to previous approaches, the input text prompt is converted directly to audio without the intermediate use of phonemes. It can therefore generalize to arbitrary instructions beyond speech such as music lyrics, sound effects or other non-speech sounds.

Below is a list of some known non-speech sounds, but we are finding more every day. Please let us know if you find patterns that work particularly well on [Discord](https://discord.gg/J2B2vsjKuE)!

- `[laughter]`
- `[laughs]`
- `[sighs]`
- `[music]`
- `[gasps]`
- `[clears throat]`
- `—` or `...` for hesitations
- `♪` for song lyrics
- CAPITALIZATION for emphasis of a word
- `[MAN]` and `[WOMAN]` to bias Bark toward male and female speakers, respectively

### Supported Languages

| Language | Status |
| --- | --- |
| English (en) | ✅ |
| German (de) | ✅ |
| Spanish (es) | ✅ |
| French (fr) | ✅ |
| Hindi (hi) | ✅ |
| Italian (it) | ✅ |
| Japanese (ja) | ✅ |
| Korean (ko) | ✅ |
| Polish (pl) | ✅ |
| Portuguese (pt) | ✅ |
| Russian (ru) | ✅ |
| Turkish (tr) | ✅ |
| Chinese, simplified (zh) | ✅ |

Requests for future language support [here](https://github.com/suno-ai/bark/discussions/111) or in the **#forums** channel on [Discord](https://discord.com/invite/J2B2vsjKuE). 

