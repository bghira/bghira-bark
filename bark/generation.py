import contextlib
import gc
import os
import re

from encodec import EncodecModel
import funcy
import logging
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download
from typing import Optional

from .model import GPTConfig, GPT
from .model_fine import FineGPT, FineGPTConfig


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """GPU top-p (nucleus) filtering. Works on 1-D (V,) or 2-D (B, V) logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Shift right so the token that pushes past top_p is kept
    remove_mask = (cumulative_probs - sorted_probs) >= top_p
    sorted_logits[remove_mask] = -float("inf")
    logits.scatter_(-1, sorted_indices, sorted_logits)
    return logits

_TORCH23_PLUS = tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 3)

COMPILE_KW = dict(
    mode="reduce-overhead",  # good default for small, latency‑sensitive batches
    fullgraph=False,  # allow custom CUDA ops (Sparge/Sage) outside graph
    dynamic=True,  # one graph for all prompt lengths
)


def maybe_compile(model: torch.nn.Module, *, tag: str = "", **kwargs):
    """Compile *model* with `torch.compile` when it makes sense.

    Set the env var `SUNO_DISABLE_COMPILE` to skip compilation without code
    changes.
    """
    if _TORCH23_PLUS and torch.cuda.is_available() and not os.getenv("SUNO_DISABLE_COMPILE"):
        try:
            logging.info(f"[torch.compile] Compiling {tag or model.__class__.__name__} …")
            kwargs = {**COMPILE_KW, **kwargs, "mode": "max-autotune-no-cudagraphs"}
            return torch.compile(model, **(kwargs or COMPILE_KW))
        except Exception as err:
            logging.warning(f"[torch.compile] Failed for {tag}: {err}. Falling back to eager.")
            return model
    return model


if (
    torch.cuda.is_available()
    and hasattr(torch, "amp")
    and hasattr(torch.amp, "autocast")
    and hasattr(torch.cuda, "is_bf16_supported")
    and torch.cuda.is_bf16_supported()
):
    autocast = funcy.partial(torch.amp.autocast, dtype=torch.bfloat16)
else:

    @contextlib.contextmanager
    def autocast():
        yield


# hold models in global scope to lazy load
global models
models = {}

global models_devices
models_devices = {}


CONTEXT_WINDOW_SIZE = 1024

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
COARSE_RATE_HZ = 75

SAMPLE_RATE = 24_000


SUPPORTED_LANGS = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]

logger = logging.getLogger(__name__)


CUR_PATH = os.path.dirname(os.path.abspath(__file__))


default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "suno", "bark_v0")


USE_SMALL_MODELS = os.environ.get("SUNO_USE_SMALL_MODELS", False)
GLOBAL_ENABLE_MPS = os.environ.get("SUNO_ENABLE_MPS", False)
OFFLOAD_CPU = os.environ.get("SUNO_OFFLOAD_CPU", True)


REMOTE_MODEL_PATHS = {
    "text_small": {
        "repo_id": "suno/bark",
        "file_name": "text.pt",
    },
    "coarse_small": {
        "repo_id": "suno/bark",
        "file_name": "coarse.pt",
    },
    "fine_small": {
        "repo_id": "suno/bark",
        "file_name": "fine.pt",
    },
    "text": {
        "repo_id": "suno/bark",
        "file_name": "text_2.pt",
    },
    "coarse": {
        "repo_id": "suno/bark",
        "file_name": "coarse_2.pt",
    },
    "fine": {
        "repo_id": "suno/bark",
        "file_name": "fine_2.pt",
    },
}

if not hasattr(torch.nn.functional, "scaled_dot_product_attention") and torch.cuda.is_available():
    logger.warning(
        "torch version does not support flash attention. You will get faster"
        + " inference speed by upgrade torch to newest nightly version."
    )


def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    elif torch.backends.mps.is_available() and use_gpu and GLOBAL_ENABLE_MPS:
        device = "mps"
    else:
        device = "cpu"
    return device


def _get_ckpt_path(model_type, use_small=False):
    key = model_type
    if use_small or USE_SMALL_MODELS:
        key += "_small"
    return os.path.join(CACHE_DIR, REMOTE_MODEL_PATHS[key]["file_name"])


def _download(from_hf_path, file_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    hf_hub_download(repo_id=from_hf_path, filename=file_name, local_dir=CACHE_DIR)


class InferenceContext:
    def __init__(self, benchmark=False):
        # we can't expect inputs to be the same length, so disable benchmarking by default
        self._chosen_cudnn_benchmark = benchmark
        self._cudnn_benchmark = None

    def __enter__(self):
        self._cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self._chosen_cudnn_benchmark

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.backends.cudnn.benchmark = self._cudnn_benchmark


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@contextlib.contextmanager
def _inference_mode():
    with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast(device_type="cuda"):
        yield


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clean_models(model_key=None):
    global models
    model_keys = [model_key] if model_key is not None else list(models.keys())
    for k in model_keys:
        if k in models:
            del models[k]
    _clear_cuda_cache()
    gc.collect()


def _load_model(ckpt_path, device, use_small=False, model_type="text"):
    """Load & (optionally) compile a GPT or FineGPT checkpoint."""

    # pick config / class based on model_type
    if model_type == "text":
        ConfigClass, ModelClass = GPTConfig, GPT
    elif model_type == "coarse":
        ConfigClass, ModelClass = GPTConfig, GPT
    elif model_type == "fine":
        ConfigClass, ModelClass = FineGPTConfig, FineGPT
    else:
        raise NotImplementedError(model_type)

    # resolve ckpt path (may download)
    model_key = f"{model_type}_small" if use_small or USE_SMALL_MODELS else model_type
    info = REMOTE_MODEL_PATHS[model_key]
    if not os.path.exists(ckpt_path):
        logger.info(f"{model_type} checkpoint missing → downloading to {CACHE_DIR} …")
        _download(info["repo_id"], info["file_name"])

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # handle old ckpts missing new keys
    model_args = checkpoint["model_args"]
    if "input_vocab_size" not in model_args:
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        del model_args["vocab_size"]

    model = ModelClass(ConfigClass(**model_args))
    model.load_state_dict(_fix_checkpoint_keys(checkpoint["model"]))

    # —> Compile here (GPU‑only; guarded)  <—
    model = maybe_compile(model, tag=model_type)

    model.eval().to(device)

    n_params = model.get_num_params()
    logger.info(
        f"{model_type} loaded: {round(n_params/1e6,1)} M params, "
        f"val‑loss={checkpoint['best_val_loss']:.3f}"
    )

    # tokenizer only for text model
    if model_type == "text":
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        return {"model": model, "tokenizer": tokenizer}

    return model


def _load_codec_model(device: str):
    """Load & compile the EnCodec 24 kHz decoder."""
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)

    # —> compile decoder too  <—
    model.decoder = maybe_compile(model.decoder, tag="encodec_decoder")

    model.eval().to(device)
    _clear_cuda_cache()
    return model


def load_model(use_gpu=True, use_small=False, force_reload=False, model_type="text"):
    _load_model_f = funcy.partial(
        _load_model, model_type=model_type, use_small=use_small,
    )
    if model_type not in ("text", "coarse", "fine"):
        raise NotImplementedError()
    global models
    global models_devices
    device = _grab_best_device(use_gpu=use_gpu)
    model_key = f"{model_type}"
    if OFFLOAD_CPU:
        models_devices[model_key] = device
        device = "cpu"
    if model_key not in models or force_reload:
        ckpt_path = _get_ckpt_path(model_type, use_small=use_small)
        clean_models(model_key=model_key)
        model = _load_model_f(ckpt_path, device)
        models[model_key] = model
    if model_type == "text":
        models[model_key]["model"].to(device)
    else:
        models[model_key].to(device)
    return models[model_key]


def load_codec_model(use_gpu=True, force_reload=False):
    global models
    global models_devices
    device = _grab_best_device(use_gpu=use_gpu)
    if device == "mps":
        # encodec doesn't support mps
        device = "cpu"
    model_key = "codec"
    if OFFLOAD_CPU:
        models_devices[model_key] = device
        device = "cpu"
    if model_key not in models or force_reload:
        clean_models(model_key=model_key)
        model = _load_codec_model(device)
        models[model_key] = model
    models[model_key].to(device)
    return models[model_key]


def _fix_checkpoint_keys(state_dict):
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    return state_dict


def preload_models(
    text_use_gpu=True,
    text_use_small=False,
    coarse_use_gpu=True,
    coarse_use_small=False,
    fine_use_gpu=True,
    fine_use_small=False,
    codec_use_gpu=True,
    force_reload=False,
):
    """Load all the necessary models for the pipeline."""
    if _grab_best_device() == "cpu" and (
        text_use_gpu or coarse_use_gpu or fine_use_gpu or codec_use_gpu
    ):
        logger.warning("No GPU being used. Careful, inference might be very slow!")
    _ = load_model(
        model_type="text", use_gpu=text_use_gpu, use_small=text_use_small,
        force_reload=force_reload,
    )
    _ = load_model(
        model_type="coarse",
        use_gpu=coarse_use_gpu,
        use_small=coarse_use_small,
        force_reload=force_reload,
    )
    _ = load_model(
        model_type="fine", use_gpu=fine_use_gpu, use_small=fine_use_small, force_reload=force_reload
    )
    _ = load_codec_model(use_gpu=codec_use_gpu, force_reload=force_reload)


####
# Generation Functionality
####


def _tokenize(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _detokenize(tokenizer, enc_text):
    return tokenizer.decode(enc_text)


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599


def _load_history_prompt(history_prompt_input):
    if isinstance(history_prompt_input, str) and history_prompt_input.endswith(".npz"):
        history_prompt = np.load(history_prompt_input)
    elif isinstance(history_prompt_input, str):
        # make sure this works on non-ubuntu
        history_prompt_input = os.path.join(*history_prompt_input.split("/"))
        history_prompt = np.load(
            os.path.join(CUR_PATH, "assets", "prompts", f"{history_prompt_input}.npz")
        )
    elif isinstance(history_prompt_input, dict):
        assert "semantic_prompt" in history_prompt_input
        assert "coarse_prompt" in history_prompt_input
        assert "fine_prompt" in history_prompt_input
        history_prompt = history_prompt_input
    else:
        raise ValueError("history prompt format unrecognized")
    return history_prompt


def generate_text_semantic(
    text: str,
    history_prompt: Optional[dict | str] = None,
    *,
    temp: float = 0.7,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    silent: bool = False,
    min_eos_p: float = 0.2,
    max_gen_duration_s: Optional[float] = None,
    allow_early_stop: bool = True,
    use_kv_caching: bool = False,
):
    """Generate semantic tokens from *text* using the patched fast path."""

    text = _normalize_whitespace(text)
    if not text:
        raise ValueError("Prompt text must be non‑empty")

    # 1. Handle history prompt ------------------------------------------------
    if history_prompt is not None:
        hp = _load_history_prompt(history_prompt)
        semantic_history = hp["semantic_prompt"].astype(np.int64)
        assert semantic_history.ndim == 1 and semantic_history.max() < SEMANTIC_VOCAB_SIZE
    else:
        semantic_history = None

    # 2. Retrieve model & tokenizer ------------------------------------------
    if "text" not in models:
        preload_models()
    container = models["text"]
    model, tokenizer = container["model"], container["tokenizer"]
    if OFFLOAD_CPU:
        model.to(models_devices["text"])
    device = next(model.parameters()).device

    # 3. Encode text & history ------------------------------------------------
    tok = np.array(_tokenize(tokenizer, text)) + TEXT_ENCODING_OFFSET
    if tok.size > 256:
        logger.warning(
            "prompt too long, truncating to 256 tokens (%.1f%% removed)",
            (tok.size - 256) / tok.size * 100,
        )
        tok = tok[:256]
    tok = np.pad(tok, (0, 256 - tok.size), constant_values=TEXT_PAD_TOKEN)

    if semantic_history is not None:
        semantic_history = semantic_history[-256:]
        semantic_history = np.pad(
            semantic_history, (0, 256 - semantic_history.size), constant_values=SEMANTIC_PAD_TOKEN
        )
    else:
        semantic_history = np.full(256, SEMANTIC_PAD_TOKEN, dtype=np.int64)

    x_np = np.hstack([tok, semantic_history, [SEMANTIC_INFER_TOKEN]]).astype(np.int64)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)

    # 4. Pre‑allocate tensor for new tokens -----------------------------------
    n_tot_steps = 768
    x_initial = x.shape[1]
    x = torch.hstack([x, torch.empty((1, n_tot_steps), dtype=torch.int32, device=device)])

    kv_cache = None
    tot_gen_dur = 0.0
    pbar = tqdm.tqdm(total=n_tot_steps, disable=silent)

    with _inference_mode():
        for n in range(n_tot_steps):
            # Input selection for KV caching
            if use_kv_caching and kv_cache is not None:
                x_input = x[:, [x_initial + n - 1]]
            else:
                x_input = x[:, : x_initial + n]

            logits, kv_cache = model(
                x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
            )
            logits = logits[0, 0]
            relevant_logits = logits[:SEMANTIC_VOCAB_SIZE]
            if allow_early_stop:
                relevant_logits = torch.concat([relevant_logits, logits[[SEMANTIC_PAD_TOKEN]]])

            # top‑p / top‑k filtering (all on-device, no CPU round-trip)
            if top_p is not None:
                relevant_logits = _top_p_filter(relevant_logits, top_p)
            if top_k is not None:
                v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.numel()))
                relevant_logits[relevant_logits < v[-1]] = -float("inf")

            probs = F.softmax(relevant_logits / temp, dim=-1)
            if probs.device.type == "mps":  # multinomial bug workaround
                probs_cpu = probs.cpu()
                sample = torch.multinomial(probs_cpu, 1).to(probs.device)
            else:
                sample = torch.multinomial(probs, 1)

            # Early‑stop check (compare on device to avoid GPU sync per step)
            if allow_early_stop and (
                (sample == SEMANTIC_VOCAB_SIZE).item() or (min_eos_p and probs[-1] >= min_eos_p)
            ):
                n -= 1  # exclude eos from slice
                pbar.update(n + 1 - pbar.n)
                break

            # Write token into slot & continue ---------------------------------
            x[0, x_initial + n] = sample
            tot_gen_dur += 1 / SEMANTIC_RATE_HZ
            pbar.update(1)

            if max_gen_duration_s and tot_gen_dur > max_gen_duration_s:
                break
        pbar.close()

    if OFFLOAD_CPU:
        model.to("cpu")

    out = x.detach().cpu().numpy().squeeze()[x_initial : x_initial + n + 1]
    assert (0 <= out).all() and (out < SEMANTIC_VOCAB_SIZE).all()
    _clear_cuda_cache()
    return out


def _flatten_codebooks(arr, offset_size=CODEBOOK_SIZE):
    assert len(arr.shape) == 2
    if offset_size is not None:
        offsets = np.arange(arr.shape[0])[:, None] * offset_size
        arr = arr + offsets
    flat_arr = arr.ravel("F")
    return flat_arr


COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050


def generate_coarse(
    x_semantic,
    history_prompt=None,
    temp=0.7,
    top_k=None,
    top_p=None,
    silent=False,
    max_coarse_history=630,  # min 60 (faster), max 630 (more context)
    sliding_window_len=60,
    use_kv_caching=True,
):
    """Generate coarse audio codes from semantic tokens."""
    assert (
        isinstance(x_semantic, np.ndarray)
        and len(x_semantic.shape) == 1
        and len(x_semantic) > 0
        and x_semantic.min() >= 0
        and x_semantic.max() <= SEMANTIC_VOCAB_SIZE - 1
    )
    assert 60 <= max_coarse_history <= 630
    assert max_coarse_history + sliding_window_len <= 1024 - 256
    semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
    max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
    if history_prompt is not None:
        history_prompt = _load_history_prompt(history_prompt)
        x_semantic_history = history_prompt["semantic_prompt"]
        x_coarse_history = history_prompt["coarse_prompt"]
        assert (
            isinstance(x_semantic_history, np.ndarray)
            and len(x_semantic_history.shape) == 1
            and len(x_semantic_history) > 0
            and x_semantic_history.min() >= 0
            and x_semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
            and isinstance(x_coarse_history, np.ndarray)
            and len(x_coarse_history.shape) == 2
            and x_coarse_history.shape[0] == N_COARSE_CODEBOOKS
            and x_coarse_history.shape[-1] >= 0
            and x_coarse_history.min() >= 0
            and x_coarse_history.max() <= CODEBOOK_SIZE - 1
            and (
                round(x_coarse_history.shape[-1] / len(x_semantic_history), 1)
                == round(semantic_to_coarse_ratio / N_COARSE_CODEBOOKS, 1)
            )
        )
        x_coarse_history = _flatten_codebooks(x_coarse_history) + SEMANTIC_VOCAB_SIZE
        # trim histories correctly
        n_semantic_hist_provided = np.min(
            [
                max_semantic_history,
                len(x_semantic_history) - len(x_semantic_history) % 2,
                int(np.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
            ]
        )
        n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
        x_semantic_history = x_semantic_history[-n_semantic_hist_provided:].astype(np.int32)
        x_coarse_history = x_coarse_history[-n_coarse_hist_provided:].astype(np.int32)
        # TODO: bit of a hack for time alignment (sounds better)
        x_coarse_history = x_coarse_history[:-2]
    else:
        x_semantic_history = np.array([], dtype=np.int32)
        x_coarse_history = np.array([], dtype=np.int32)
    # load models if not yet exist
    global models
    global models_devices
    if "coarse" not in models:
        preload_models()
    model = models["coarse"]
    if OFFLOAD_CPU:
        model.to(models_devices["coarse"])
    device = next(model.parameters()).device
    # start loop
    n_steps = int(
        round(
            np.floor(len(x_semantic) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
            * N_COARSE_CODEBOOKS
        )
    )
    assert n_steps > 0 and n_steps % N_COARSE_CODEBOOKS == 0
    x_semantic = np.hstack([x_semantic_history, x_semantic]).astype(np.int32)
    x_coarse = x_coarse_history.astype(np.int32)
    base_semantic_idx = len(x_semantic_history)
    with _inference_mode():
        x_semantic_in = torch.from_numpy(x_semantic)[None].to(device)
        # Pre-allocate x_coarse_in with room for all new tokens (avoids O(n^2) torch.cat)
        x_coarse_base = torch.from_numpy(x_coarse)[None].to(device)
        n_coarse_history = x_coarse_base.shape[1]
        x_coarse_in = torch.empty(
            (1, n_coarse_history + n_steps), dtype=x_coarse_base.dtype, device=device
        )
        x_coarse_in[:, :n_coarse_history] = x_coarse_base
        del x_coarse_base
        coarse_write_idx = n_coarse_history
        # Hoist constant token tensor outside loop
        infer_token = torch.tensor([COARSE_INFER_TOKEN], dtype=torch.long, device=device)[None]
        n_window_steps = int(np.ceil(n_steps / sliding_window_len))
        n_step = 0
        for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
            # pad from right side
            x_in = x_semantic_in[:, np.max([0, semantic_idx - max_semantic_history]) :]
            x_in = x_in[:, :256]
            x_in = F.pad(
                x_in,
                (0, 256 - x_in.shape[-1]),
                "constant",
                COARSE_SEMANTIC_PAD_TOKEN,
            )
            coarse_tail = x_coarse_in[:, max(0, coarse_write_idx - max_coarse_history):coarse_write_idx]
            # Pre-allocate x_in with room for sliding_window_len new tokens
            x_in_prefix = torch.hstack([x_in, infer_token, coarse_tail])
            prefix_len = x_in_prefix.shape[1]
            x_in = torch.empty(
                (1, prefix_len + sliding_window_len), dtype=x_in_prefix.dtype, device=device
            )
            x_in[:, :prefix_len] = x_in_prefix
            del x_in_prefix
            x_in_write_idx = prefix_len
            kv_cache = None
            for _ in range(sliding_window_len):
                if n_step >= n_steps:
                    continue
                is_major_step = n_step % N_COARSE_CODEBOOKS == 0

                if use_kv_caching and kv_cache is not None:
                    x_input = x_in[:, [x_in_write_idx - 1]]
                else:
                    x_input = x_in[:, :x_in_write_idx]

                logits, kv_cache = model(x_input, use_cache=use_kv_caching, past_kv=kv_cache)
                logit_start_idx = SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
                logit_end_idx = SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE
                relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]
                if top_p is not None:
                    relevant_logits = _top_p_filter(relevant_logits, top_p)
                if top_k is not None:
                    v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                    relevant_logits[relevant_logits < v[-1]] = -float("Inf")
                probs = F.softmax(relevant_logits / temp, dim=-1)
                # multinomial bugged on mps: shuttle to cpu if necessary
                inf_device = probs.device
                if probs.device.type == "mps":
                    probs = probs.to("cpu")
                item_next = torch.multinomial(probs, num_samples=1)
                probs = probs.to(inf_device)
                item_next = item_next.to(inf_device)
                item_next += logit_start_idx
                x_coarse_in[0, coarse_write_idx] = item_next
                x_in[0, x_in_write_idx] = item_next
                coarse_write_idx += 1
                x_in_write_idx += 1
                del logits, relevant_logits, probs, item_next
                n_step += 1
            del x_in
        del x_semantic_in
    if OFFLOAD_CPU:
        model.to("cpu")
    gen_coarse_arr = x_coarse_in.detach().cpu().numpy().squeeze()[n_coarse_history:]
    del x_coarse_in
    assert len(gen_coarse_arr) == n_steps
    gen_coarse_audio_arr = gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
    gen_coarse_audio_arr -= np.arange(N_COARSE_CODEBOOKS)[:, None] * CODEBOOK_SIZE
    _clear_cuda_cache()
    return gen_coarse_audio_arr


def generate_fine(
    x_coarse_gen,
    history_prompt=None,
    temp=0.6,
    silent=True,
):
    """Generate full audio codes from coarse audio codes."""
    assert (
        isinstance(x_coarse_gen, np.ndarray)
        and len(x_coarse_gen.shape) == 2
        and 1 <= x_coarse_gen.shape[0] <= N_FINE_CODEBOOKS - 1
        and x_coarse_gen.shape[1] > 0
        and x_coarse_gen.min() >= 0
        and x_coarse_gen.max() <= CODEBOOK_SIZE - 1
    )
    if history_prompt is not None:
        history_prompt = _load_history_prompt(history_prompt)
        x_fine_history = history_prompt["fine_prompt"]
        assert (
            isinstance(x_fine_history, np.ndarray)
            and len(x_fine_history.shape) == 2
            and x_fine_history.shape[0] == N_FINE_CODEBOOKS
            and x_fine_history.shape[1] >= 0
            and x_fine_history.min() >= 0
            and x_fine_history.max() <= CODEBOOK_SIZE - 1
        )
    else:
        x_fine_history = None
    n_coarse = x_coarse_gen.shape[0]
    # load models if not yet exist
    global models
    global models_devices
    if "fine" not in models:
        preload_models()
    model = models["fine"]
    if OFFLOAD_CPU:
        model.to(models_devices["fine"])
    device = next(model.parameters()).device
    # make input arr
    in_arr = np.vstack(
        [
            x_coarse_gen,
            np.zeros((N_FINE_CODEBOOKS - n_coarse, x_coarse_gen.shape[1]))
            + CODEBOOK_SIZE,  # padding
        ]
    ).astype(np.int32)
    # prepend history if available (max 512)
    if x_fine_history is not None:
        x_fine_history = x_fine_history.astype(np.int32)
        in_arr = np.hstack(
            [
                x_fine_history[:, -512:].astype(np.int32),
                in_arr,
            ]
        )
        n_history = x_fine_history[:, -512:].shape[1]
    else:
        n_history = 0
    n_remove_from_end = 0
    # need to pad if too short (since non-causal model)
    if in_arr.shape[1] < 1024:
        n_remove_from_end = 1024 - in_arr.shape[1]
        in_arr = np.hstack(
            [
                in_arr,
                np.zeros((N_FINE_CODEBOOKS, n_remove_from_end), dtype=np.int32) + CODEBOOK_SIZE,
            ]
        )
    # we can be lazy about fractional loop and just keep overwriting codebooks
    n_loops = np.max([0, int(np.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512))]) + 1
    with _inference_mode():
        in_arr = torch.tensor(in_arr.T).to(device)
        for n in tqdm.tqdm(range(n_loops), disable=silent):
            start_idx = np.min([n * 512, in_arr.shape[0] - 1024])
            start_fill_idx = np.min([n_history + n * 512, in_arr.shape[0] - 512])
            rel_start_fill_idx = start_fill_idx - start_idx
            in_buffer = in_arr[start_idx : start_idx + 1024, :][None]
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                logits = model(nn, in_buffer)
                if temp is None:
                    relevant_logits = logits[0, rel_start_fill_idx:, :CODEBOOK_SIZE]
                    codebook_preds = torch.argmax(relevant_logits, -1)
                else:
                    relevant_logits = logits[0, rel_start_fill_idx:, :CODEBOOK_SIZE] / temp
                    probs = F.softmax(relevant_logits, dim=-1)
                    # multinomial bugged on mps: shuttle to cpu if necessary
                    inf_device = probs.device
                    if probs.device.type == "mps":
                        probs = probs.to("cpu")
                    # Batched multinomial instead of per-row loop
                    codebook_preds = torch.multinomial(probs, num_samples=1).squeeze(-1).to(inf_device)
                in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds
                del logits, codebook_preds
            # transfer over info into model_in and convert to numpy
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                in_arr[start_fill_idx : start_fill_idx + (1024 - rel_start_fill_idx), nn] = (
                    in_buffer[0, rel_start_fill_idx:, nn]
                )
            del in_buffer
        gen_fine_arr = in_arr.detach().cpu().numpy().squeeze().T
        del in_arr
    if OFFLOAD_CPU:
        model.to("cpu")
    gen_fine_arr = gen_fine_arr[:, n_history:]
    if n_remove_from_end > 0:
        gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]
    assert gen_fine_arr.shape[-1] == x_coarse_gen.shape[-1]
    _clear_cuda_cache()
    return gen_fine_arr


def codec_decode(fine_tokens):
    """Turn quantized audio codes into audio array using encodec."""
    # load models if not yet exist
    global models
    global models_devices
    if "codec" not in models:
        preload_models()
    model = models["codec"]
    if OFFLOAD_CPU:
        model.to(models_devices["codec"])
    device = next(model.parameters()).device
    arr = torch.from_numpy(fine_tokens)[None].to(device)
    arr = arr.transpose(0, 1)
    emb = model.quantizer.decode(arr)
    # run decoder under no_grad only (not full inference_mode) so weight-norm hooks still see valid version counters
    with torch.no_grad():
        out = model.decoder(emb)
    audio_arr = out.cpu().detach().numpy().squeeze()
    del arr, emb, out
    if OFFLOAD_CPU:
        model.to("cpu")
    return audio_arr


# ── Batched generation ──────────────────────────────────────────────────────


def generate_text_semantic_batched(
    texts: list[str],
    history_prompts: list = None,
    *,
    temp: float = 0.7,
    top_k: int | None = None,
    top_p: float | None = None,
    silent: bool = False,
    min_eos_p: float = 0.2,
    max_gen_duration_s: float | None = None,
    allow_early_stop: bool = True,
    use_kv_caching: bool = True,
) -> list[np.ndarray]:
    """Batched semantic-token generation for *B* texts at once.

    All sequences share the same prefill length (513 tokens) so no
    padding masks are needed — the causal GPT handles batch>1 natively.
    """
    B = len(texts)
    if not B:
        return []
    if history_prompts is None:
        history_prompts = [None] * B

    # ── load model & tokenizer ──────────────────────────────────────────
    global models, models_devices
    if "text" not in models:
        preload_models()
    container = models["text"]
    model, tokenizer = container["model"], container["tokenizer"]
    if OFFLOAD_CPU:
        model.to(models_devices["text"])
    device = next(model.parameters()).device

    # ── build [B, 513] input ────────────────────────────────────────────
    rows = []
    for i in range(B):
        text = _normalize_whitespace(texts[i])
        if not text:
            raise ValueError(f"Prompt {i} is empty")

        tok = np.array(_tokenize(tokenizer, text)) + TEXT_ENCODING_OFFSET
        if tok.size > 256:
            tok = tok[:256]
        tok = np.pad(tok, (0, 256 - tok.size), constant_values=TEXT_PAD_TOKEN)

        if history_prompts[i] is not None:
            hp = _load_history_prompt(history_prompts[i])
            sem_hist = hp["semantic_prompt"].astype(np.int64)[-256:]
            sem_hist = np.pad(
                sem_hist, (0, 256 - sem_hist.size), constant_values=SEMANTIC_PAD_TOKEN
            )
        else:
            sem_hist = np.full(256, SEMANTIC_PAD_TOKEN, dtype=np.int64)

        rows.append(np.hstack([tok, sem_hist, [SEMANTIC_INFER_TOKEN]]).astype(np.int64))

    x = torch.from_numpy(np.stack(rows)).to(device)  # [B, 513]
    n_tot_steps = 768
    x_initial = x.shape[1]
    # pre-allocate output slots filled with pad (safe to feed for finished seqs)
    x = torch.cat(
        [x, torch.full((B, n_tot_steps), SEMANTIC_PAD_TOKEN, dtype=torch.long, device=device)],
        dim=1,
    )

    kv_cache = None
    active = torch.ones(B, dtype=torch.bool, device=device)
    gen_lens = torch.full((B,), n_tot_steps, dtype=torch.long, device=device)
    pbar = tqdm.tqdm(total=n_tot_steps, disable=silent)

    with _inference_mode():
        for n in range(n_tot_steps):
            if use_kv_caching and kv_cache is not None:
                x_input = x[:, [x_initial + n - 1]]
            else:
                x_input = x[:, : x_initial + n]

            logits, kv_cache = model(
                x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
            )
            logits = logits[:, 0, :]  # [B, vocab]
            relevant_logits = logits[:, :SEMANTIC_VOCAB_SIZE]
            if allow_early_stop:
                relevant_logits = torch.cat(
                    [relevant_logits, logits[:, SEMANTIC_PAD_TOKEN : SEMANTIC_PAD_TOKEN + 1]],
                    dim=-1,
                )

            # top-p / top-k (operates per-row for 2-D tensors)
            if top_p is not None:
                relevant_logits = _top_p_filter(relevant_logits, top_p)
            if top_k is not None:
                v, _ = torch.topk(
                    relevant_logits, min(top_k, relevant_logits.size(-1)), dim=-1
                )
                relevant_logits[relevant_logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(relevant_logits / temp, dim=-1)
            if probs.device.type == "mps":
                samples = torch.multinomial(probs.cpu(), 1).to(probs.device)
            else:
                samples = torch.multinomial(probs, 1)
            samples = samples.squeeze(-1)  # [B]

            # early-stop bookkeeping
            if allow_early_stop:
                is_eos = samples == SEMANTIC_VOCAB_SIZE
                if min_eos_p:
                    is_eos = is_eos | (probs[:, -1] >= min_eos_p)
                just_finished = active & is_eos
                gen_lens[just_finished] = n  # n tokens at indices 0..n-1
                active &= ~just_finished

            x[active, x_initial + n] = samples[active]
            pbar.update(1)

            if not active.any():
                break
            if max_gen_duration_s and (n + 1) / SEMANTIC_RATE_HZ > max_gen_duration_s:
                gen_lens[active] = n + 1
                break
    pbar.close()

    if OFFLOAD_CPU:
        model.to("cpu")

    x_cpu = x.detach().cpu().numpy()
    results = []
    for b in range(B):
        L = gen_lens[b].item()
        out = x_cpu[b, x_initial : x_initial + L]
        assert (0 <= out).all() and (out < SEMANTIC_VOCAB_SIZE).all()
        results.append(out)

    _clear_cuda_cache()
    return results


def generate_coarse_batched(
    x_semantics: list[np.ndarray],
    history_prompts: list = None,
    *,
    temp: float = 0.7,
    top_k: int | None = None,
    top_p: float | None = None,
    silent: bool = False,
    max_coarse_history: int = 630,
    sliding_window_len: int = 60,
    use_kv_caching: bool = True,
) -> list[np.ndarray]:
    """Batched coarse-code generation from semantic tokens.

    Supports heterogeneous history prompts.  Each sequence gets its own
    semantic/coarse history, padded into a common buffer so the model
    forward pass runs with batch > 1.
    """
    B = len(x_semantics)
    if not B:
        return []
    if history_prompts is None:
        history_prompts = [None] * B

    assert 60 <= max_coarse_history <= 630
    assert max_coarse_history + sliding_window_len <= 1024 - 256
    semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
    max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))

    # ── per-sequence preprocessing ──────────────────────────────────────
    all_x_semantic = []      # full semantic (history + new) per seq
    all_x_coarse_hist = []   # flattened coarse history per seq
    all_base_sem_idx = []    # where new semantic tokens start per seq
    all_n_steps = []         # coarse steps to generate per seq

    for b in range(B):
        x_sem = x_semantics[b]
        assert x_sem.ndim == 1 and x_sem.min() >= 0 and x_sem.max() <= SEMANTIC_VOCAB_SIZE - 1
        hp = history_prompts[b]
        if hp is not None:
            hp = _load_history_prompt(hp)
            x_sem_hist = hp["semantic_prompt"]
            x_coarse_hist = hp["coarse_prompt"]
            x_coarse_hist = _flatten_codebooks(x_coarse_hist) + SEMANTIC_VOCAB_SIZE
            n_sem_hist = int(np.min([
                max_semantic_history,
                len(x_sem_hist) - len(x_sem_hist) % 2,
                int(np.floor(len(x_coarse_hist) / semantic_to_coarse_ratio)),
            ]))
            n_coarse_hist = int(round(n_sem_hist * semantic_to_coarse_ratio))
            x_sem_hist = x_sem_hist[-n_sem_hist:].astype(np.int32)
            x_coarse_hist = x_coarse_hist[-n_coarse_hist:].astype(np.int32)
            x_coarse_hist = x_coarse_hist[:-2]  # time-alignment hack
        else:
            x_sem_hist = np.array([], dtype=np.int32)
            x_coarse_hist = np.array([], dtype=np.int32)

        all_base_sem_idx.append(len(x_sem_hist))
        all_x_semantic.append(np.hstack([x_sem_hist, x_sem]).astype(np.int32))
        all_x_coarse_hist.append(x_coarse_hist)

        n_steps = int(round(
            np.floor(len(x_sem) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
            * N_COARSE_CODEBOOKS
        ))
        assert n_steps > 0 and n_steps % N_COARSE_CODEBOOKS == 0
        all_n_steps.append(n_steps)

    n_steps_max = max(all_n_steps)
    max_coarse_hist_len = max(len(h) for h in all_x_coarse_hist)

    # pad semantics to common length
    max_sem_len = max(s.size for s in all_x_semantic)
    x_sem_padded = np.zeros((B, max_sem_len), dtype=np.int32)
    for i, s in enumerate(all_x_semantic):
        x_sem_padded[i, :s.size] = s

    # ── load model ──────────────────────────────────────────────────────
    global models, models_devices
    if "coarse" not in models:
        preload_models()
    model = models["coarse"]
    if OFFLOAD_CPU:
        model.to(models_devices["coarse"])
    device = next(model.parameters()).device

    with _inference_mode():
        x_semantic_in = torch.from_numpy(x_sem_padded).to(device)  # [B, max_sem_len]

        # pre-allocate coarse buffer [B, max_coarse_hist + n_steps_max]
        # each row is right-aligned: history sits at the end of the hist region
        total_coarse_cols = max_coarse_hist_len + n_steps_max
        x_coarse_in = torch.zeros(
            (B, total_coarse_cols), dtype=torch.long, device=device
        )
        # per-sequence: write history right-aligned into [0..max_coarse_hist_len)
        for b in range(B):
            h = all_x_coarse_hist[b]
            if len(h) > 0:
                ht = torch.from_numpy(h).to(device)
                x_coarse_in[b, max_coarse_hist_len - len(h) : max_coarse_hist_len] = ht

        coarse_write_idx = max_coarse_hist_len  # all new tokens start here

        infer_token = torch.tensor(
            [COARSE_INFER_TOKEN], dtype=torch.long, device=device
        ).unsqueeze(0).expand(B, -1)

        # per-sequence base_sem_idx as tensor for vectorised offset computation
        base_sem_idx_t = torch.tensor(all_base_sem_idx, device=device)

        n_window_steps = int(np.ceil(n_steps_max / sliding_window_len))
        n_step = 0

        for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            # ── build per-sequence semantic context [B, 256] ────────
            # semantic_idx per seq = base_sem_idx[b] + round(n_step / ratio)
            sem_offset = int(round(n_step / semantic_to_coarse_ratio))
            # per-seq start into x_semantic_in
            sem_idx_per_seq = [all_base_sem_idx[b] + sem_offset for b in range(B)]
            sem_ctx = torch.full(
                (B, 256), COARSE_SEMANTIC_PAD_TOKEN, dtype=torch.long, device=device
            )
            for b in range(B):
                si = sem_idx_per_seq[b]
                start = max(0, si - max_semantic_history)
                end = min(start + 256, x_semantic_in.shape[1])
                width = end - start
                sem_ctx[b, :width] = x_semantic_in[b, start:end]

            # ── coarse tail context ─────────────────────────────────
            # each sequence sees its own history region + generated tokens
            # tail = last max_coarse_history tokens of x_coarse_in[:, :coarse_write_idx]
            tail_start = max(0, coarse_write_idx - max_coarse_history)
            coarse_tail = x_coarse_in[:, tail_start:coarse_write_idx]

            x_in_prefix = torch.cat([sem_ctx, infer_token, coarse_tail], dim=1)
            prefix_len = x_in_prefix.shape[1]
            x_in = torch.empty(
                (B, prefix_len + sliding_window_len),
                dtype=x_in_prefix.dtype, device=device,
            )
            x_in[:, :prefix_len] = x_in_prefix
            del x_in_prefix
            x_in_write_idx = prefix_len

            kv_cache = None
            for _ in range(sliding_window_len):
                if n_step >= n_steps_max:
                    break
                is_major_step = n_step % N_COARSE_CODEBOOKS == 0

                if use_kv_caching and kv_cache is not None:
                    x_input = x_in[:, [x_in_write_idx - 1]]
                else:
                    x_input = x_in[:, :x_in_write_idx]

                logits, kv_cache = model(
                    x_input, use_cache=use_kv_caching, past_kv=kv_cache
                )
                logit_start_idx = (
                    SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
                )
                logit_end_idx = (
                    SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE
                )
                relevant_logits = logits[:, 0, logit_start_idx:logit_end_idx]

                if top_p is not None:
                    relevant_logits = _top_p_filter(relevant_logits, top_p)
                if top_k is not None:
                    v, _ = torch.topk(
                        relevant_logits, min(top_k, relevant_logits.size(-1)), dim=-1
                    )
                    relevant_logits[relevant_logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(relevant_logits / temp, dim=-1)
                inf_device = probs.device
                if probs.device.type == "mps":
                    probs = probs.to("cpu")
                item_next = torch.multinomial(probs, num_samples=1).squeeze(-1)
                item_next = item_next.to(inf_device) + logit_start_idx

                x_coarse_in[:, coarse_write_idx] = item_next
                x_in[:, x_in_write_idx] = item_next
                coarse_write_idx += 1
                x_in_write_idx += 1
                del logits, relevant_logits, probs, item_next
                n_step += 1
            del x_in
        del x_semantic_in

    if OFFLOAD_CPU:
        model.to("cpu")

    # extract per-sequence results, trimmed to real n_steps
    x_coarse_cpu = x_coarse_in.detach().cpu().numpy()
    del x_coarse_in
    results = []
    for b in range(B):
        ns = all_n_steps[b]
        gen = x_coarse_cpu[b, max_coarse_hist_len : max_coarse_hist_len + ns]
        gen_audio = gen.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
        gen_audio -= np.arange(N_COARSE_CODEBOOKS)[:, None] * CODEBOOK_SIZE
        results.append(gen_audio)

    _clear_cuda_cache()
    return results


def generate_fine_batched(
    x_coarse_gens: list[np.ndarray],
    history_prompts: list = None,
    *,
    temp: float = 0.6,
    silent: bool = True,
) -> list[np.ndarray]:
    """Batched fine-code generation.  FineGPT already supports pad_mask.

    Supports heterogeneous history prompts.  Each sequence's history is
    right-aligned into a common prefix region so the 1024-token sliding
    window schedule is uniform across the batch.
    """
    B = len(x_coarse_gens)
    if not B:
        return []
    if history_prompts is None:
        history_prompts = [None] * B

    # ── per-sequence input arrays ───────────────────────────────────────
    n_coarse = x_coarse_gens[0].shape[0]
    all_in_arr = []
    all_coarse_len = []
    all_n_history = []

    for i in range(B):
        x_cg = x_coarse_gens[i]
        assert x_cg.ndim == 2 and x_cg.min() >= 0 and x_cg.max() <= CODEBOOK_SIZE - 1
        all_coarse_len.append(x_cg.shape[1])

        hp = history_prompts[i]
        if hp is not None:
            hp = _load_history_prompt(hp)
            x_fine_hist = hp["fine_prompt"].astype(np.int32)[:, -512:]
            all_n_history.append(x_fine_hist.shape[1])
        else:
            x_fine_hist = None
            all_n_history.append(0)

        in_arr = np.vstack([
            x_cg,
            np.zeros((N_FINE_CODEBOOKS - n_coarse, x_cg.shape[1]), dtype=np.int32) + CODEBOOK_SIZE,
        ])
        if x_fine_hist is not None:
            in_arr = np.hstack([x_fine_hist, in_arr])
        all_in_arr.append(in_arr)

    # use max history length for the uniform window schedule
    max_n_history = max(all_n_history)

    # right-align histories: pad each sequence on the left so that
    # the boundary between history and new tokens is at column max_n_history
    all_in_arr_aligned = []
    for i in range(B):
        left_pad = max_n_history - all_n_history[i]
        if left_pad > 0:
            pad_block = np.full(
                (N_FINE_CODEBOOKS, left_pad), CODEBOOK_SIZE, dtype=np.int32
            )
            all_in_arr_aligned.append(np.hstack([pad_block, all_in_arr[i]]))
        else:
            all_in_arr_aligned.append(all_in_arr[i])

    # pad to common temporal length (at least 1024)
    max_T_raw = max(a.shape[1] for a in all_in_arr_aligned)
    max_T = max(max_T_raw, 1024)
    padded = np.full((B, N_FINE_CODEBOOKS, max_T), CODEBOOK_SIZE, dtype=np.int32)
    for i in range(B):
        T = all_in_arr_aligned[i].shape[1]
        padded[i, :, :T] = all_in_arr_aligned[i]

    # ── load model ──────────────────────────────────────────────────────
    global models, models_devices
    if "fine" not in models:
        preload_models()
    model = models["fine"]
    if OFFLOAD_CPU:
        model.to(models_devices["fine"])
    device = next(model.parameters()).device

    max_coarse_len = max(all_coarse_len)
    n_loops = max(0, int(np.ceil((max_coarse_len - (1024 - max_n_history)) / 512))) + 1

    with _inference_mode():
        # [B, max_T, n_codes]  (temporal dim first for slicing)
        in_arr_t = torch.from_numpy(padded.transpose(0, 2, 1)).to(device)

        for n in tqdm.tqdm(range(n_loops), disable=silent):
            start_idx = min(n * 512, max_T - 1024)
            start_fill_idx = min(max_n_history + n * 512, max_T - 512)
            rel_start_fill_idx = start_fill_idx - start_idx
            in_buffer = in_arr_t[:, start_idx : start_idx + 1024, :].clone()

            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                logits = model(nn, in_buffer)
                if temp is None:
                    relevant_logits = logits[:, rel_start_fill_idx:, :CODEBOOK_SIZE]
                    codebook_preds = torch.argmax(relevant_logits, -1)
                else:
                    relevant_logits = logits[:, rel_start_fill_idx:, :CODEBOOK_SIZE] / temp
                    probs = F.softmax(relevant_logits, dim=-1)
                    inf_device = probs.device
                    if probs.device.type == "mps":
                        probs = probs.to("cpu")
                    flat = probs.reshape(-1, probs.shape[-1])
                    codebook_preds = (
                        torch.multinomial(flat, 1).squeeze(-1).to(inf_device)
                        .reshape(probs.shape[0], probs.shape[1])
                    )
                in_buffer[:, rel_start_fill_idx:, nn] = codebook_preds
                del logits, codebook_preds

            in_arr_t[:, start_fill_idx : start_fill_idx + (1024 - rel_start_fill_idx), :] = (
                in_buffer[:, rel_start_fill_idx:, :]
            )
            del in_buffer

        gen_fine_all = in_arr_t.detach().cpu().numpy()  # [B, max_T, n_codes]
        del in_arr_t

    if OFFLOAD_CPU:
        model.to("cpu")

    results = []
    for b in range(B):
        g = gen_fine_all[b].T  # [n_codes, max_T]
        g = g[:, max_n_history : max_n_history + all_coarse_len[b]]
        assert g.shape[-1] == all_coarse_len[b]
        results.append(g)

    _clear_cuda_cache()
    return results


def codec_decode_batched(fine_tokens_list: list[np.ndarray]) -> list[np.ndarray]:
    """Decode a list of fine-token arrays into audio waveforms.

    Pads to common length, runs EnCodec in one forward pass, then trims each
    output to its original sample count.
    """
    if not fine_tokens_list:
        return []

    global models, models_devices
    if "codec" not in models:
        preload_models()
    model = models["codec"]
    if OFFLOAD_CPU:
        model.to(models_devices["codec"])
    device = next(model.parameters()).device

    lengths = [ft.shape[1] for ft in fine_tokens_list]
    max_len = max(lengths)
    B = len(fine_tokens_list)

    padded = np.zeros((B, fine_tokens_list[0].shape[0], max_len), dtype=fine_tokens_list[0].dtype)
    for i, ft in enumerate(fine_tokens_list):
        padded[i, :, : ft.shape[1]] = ft

    arr = torch.from_numpy(padded).to(device)  # [B, n_codes, T]
    arr = arr.transpose(0, 1)                   # [n_codes, B, T]
    emb = model.quantizer.decode(arr)
    with torch.no_grad():
        out = model.decoder(emb)
    audio_all = out.cpu().detach().numpy()  # [1?, B, samples] or [B, 1, samples]
    del arr, emb, out

    if OFFLOAD_CPU:
        model.to("cpu")

    # EnCodec output shape may vary; flatten to [B, samples]
    if audio_all.ndim == 3:
        if audio_all.shape[0] == 1:
            audio_all = audio_all[0]  # [B, samples]
        elif audio_all.shape[1] == 1:
            audio_all = audio_all.squeeze(1)  # [B, samples]

    results = []
    for i in range(B):
        # compute expected sample count from token length
        n_samples = int(lengths[i] / COARSE_RATE_HZ * SAMPLE_RATE)
        results.append(audio_all[i, :n_samples])

    _clear_cuda_cache()
    return results
