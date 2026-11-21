"""Local LLM inference with MLX (Apple silicon). Qwen2.5-7B-Instruct, 4-bit quantized."""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

from . import config

_lock = threading.Lock()  # one generation at a time on the GPU


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


@lru_cache(maxsize=1)
def _model():
    from mlx_lm import load
    return load(config.LLM_MODEL)


def chat(messages: list[dict], max_tokens: int = 400, temperature: float = 0.0) -> LLMResult:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tok = _model()
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    t0, text, last = time.perf_counter(), "", None
    with _lock:
        for r in stream_generate(model, tok, prompt, max_tokens=max_tokens,
                                 sampler=make_sampler(temp=temperature)):
            text += r.text
            last = r
    return LLMResult(text.strip(), last.prompt_tokens if last else 0,
                     last.generation_tokens if last else 0, time.perf_counter() - t0)


def chat_json(messages: list[dict], max_tokens: int = 400) -> tuple[dict | None, LLMResult]:
    """Ask for JSON and parse the first object in the reply (None if it is not valid)."""
    res = chat(messages, max_tokens=max_tokens)
    m = re.search(r"\{.*\}", res.text, re.S)
    try:
        return (json.loads(m.group(0)) if m else None), res
    except json.JSONDecodeError:
        return None, res
