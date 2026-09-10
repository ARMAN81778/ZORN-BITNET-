"""High-level Python API for ZORN BitNet inference."""
from __future__ import annotations

import os
import json
import time
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

try:
    import zorn_engine
except ImportError:
    from . import zorn_engine

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


class ZornError(RuntimeError):
    pass


def validate_model(model_dir: str) -> dict:
    p = Path(model_dir)
    required = [
        "weights.zrn", "embedding.bin", "lm_head.bin",
        "final_norm.bin", "zorn_manifest.json", "tokenizer.json"
    ]
    missing = [x for x in required if not (p / x).exists()]
    if missing:
        raise ZornError("Missing ZORN model files: " + ", ".join(missing))
    try:
        manifest = json.loads((p / "zorn_manifest.json").read_text())
    except Exception as e:
        raise ZornError(f"Invalid zorn_manifest.json: {e}") from e
    return manifest


def auto_threads(limit: int = 64) -> int:
    return max(1, min(os.cpu_count() or 1, limit))

def prepare_model(model_path: str) -> str:
    """Return a ZORN model directory, converting raw Safetensors when needed."""
    p = Path(model_path).expanduser().resolve()
    if p.is_dir() and (p / "zorn_manifest.json").exists():
        return str(p)

    if p.suffix.lower() == ".gguf":
        return str(p)

    has_raw = p.is_file() and p.suffix.lower() == ".safetensors"
    if p.is_dir():
        has_raw = any(p.glob("*.safetensors")) or (p / "model.safetensors.index.json").exists()
    if not has_raw:
        raise ZornError(f"Unsupported model input: {p}")

    out = Path(str(p) + ".zorn") if p.is_file() else p.parent / (p.name + ".zorn")
    if (out / "zorn_manifest.json").exists() and (out / "tokenizer.json").exists():
        return str(out)

    from .convert_safetensors_to_zrn import convert
    try:
        convert(str(p), str(out))
    except Exception as e:
        raise ZornError(
            f"Could not convert raw checkpoint '{p}'. "
            "ZORN currently requires BitNet ternary/I2_S tensors for native conversion."
        ) from e
    return str(out)


class ZornSession:
    def __init__(self, model_dir: str, cpu_cores: Optional[int] = None):
        prepared = prepare_model(model_dir)
        if str(prepared).lower().endswith(".gguf"):
            raise ZornError("GGUF is supported by ZornChat via the optional llama-cpp-python backend; use ZornChat for GGUF models.")
        validate_model(prepared)
        self.model_dir = str(prepared)
        self.cpu_cores = cpu_cores or auto_threads()
        self.native = zorn_engine.load_model(self.model_dir, cpu_cores=self.cpu_cores)
        self.info = zorn_engine.info(self.native)
        if not self.info.get("real_generation_ready", False):
            raise ZornError("Native model loaded but is not generation-ready; check model files, tokenizer, vocabulary, and manifest.")
        self.history_tokens: list[int] = []

    def reset(self):
        zorn_engine.reset(self.native)
        self.history_tokens.clear()

    def generate_ids(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        reset_cache: bool = True,
        seed: int = 0,
        stop_ids: Optional[Sequence[int]] = None,
    ) -> list[int]:
        return list(zorn_engine.generate(
            self.native, list(map(int, input_ids)),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            reset_cache=int(reset_cache),
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            stop_ids=list(stop_ids or []),
        ))

    def stream_ids(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        stop_ids: Optional[Sequence[int]] = None,
    ) -> Iterator[int]:
        ids = list(map(int, input_ids))
        if max_new_tokens <= 0:
            return
        first = self.generate_ids(
            ids, max_new_tokens=1, temperature=temperature, top_k=top_k,
            top_p=top_p, repetition_penalty=repetition_penalty,
            reset_cache=True, stop_ids=stop_ids
        )
        if not first:
            return
        token = first[0]
        yield token
        if stop_ids and token in stop_ids:
            return
        history = ids + [token]
        base_seed = int(time.time()) & 0xFFFFFFFF
        for index in range(max_new_tokens - 1):
            nxt = int(zorn_engine.step(
                self.native, token, reset_cache=False,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=(base_seed + index + 1) & 0xFFFFFFFF,
                history_ids=history, stop_ids=list(stop_ids or [])
            ))
            token = nxt
            history.append(token)
            yield token
            if stop_ids and token in stop_ids:
                break
            if self.native is None:
                break

    def close(self):
        self.native = None


class ZornChat:
    def __init__(
        self,
        model_dir: str,
        tokenizer_name_or_path: Optional[str] = None,
        cpu_cores: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ):
        self.gguf = str(model_dir).lower().endswith(".gguf")
        self.messages = []
        if self.gguf:
            try:
                from llama_cpp import Llama
            except Exception as e:
                raise ImportError(
                    "Install GGUF support with: pip install llama-cpp-python"
                ) from e
            self.session = None
            self.tokenizer = None
            self._llama = Llama(model_path=str(model_dir), n_ctx=4096, verbose=False)
        else:
            if AutoTokenizer is None:
                raise ImportError("Install transformers to use ZornChat: pip install transformers")
            self.session = ZornSession(model_dir, cpu_cores=cpu_cores)
            tok_path = tokenizer_name_or_path or self.session.model_dir
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def _encode_chat(self, messages):
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        text = ""
        for m in messages:
            text += f"{m['role']}: {m['content']}\n"
        text += "assistant:"
        return self.tokenizer(text, add_special_tokens=True).input_ids

    def ask(self, text: str, **generation_kwargs) -> str:
        if self.gguf:
            self.messages.append({"role": "user", "content": text})
            result = self._llama.create_chat_completion(
                messages=self.messages,
                max_tokens=int(generation_kwargs.get("max_new_tokens", 128)),
                temperature=float(generation_kwargs.get("temperature", 0.7)),
                top_p=float(generation_kwargs.get("top_p", 0.9)),
                top_k=int(generation_kwargs.get("top_k", 40)),
            )
            answer = result["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": answer})
            return answer
        self.messages.append({"role": "user", "content": text})
        ids = self._encode_chat(self.messages)
        max_ctx = int(self.session.info.get("context_length", len(ids)))
        if len(ids) >= max_ctx - 8:
            system = self.messages[:1] if self.messages and self.messages[0]["role"] == "system" else []
            rest = self.messages[1:] if system else self.messages
            while len(self._encode_chat(system + rest)) >= max_ctx - 8 and len(rest) > 2:
                rest = rest[2:]
            self.messages = system + rest
            ids = self._encode_chat(self.messages)
        remaining = max(0, int(self.session.info.get("context_length", len(ids))) - len(ids))
        requested = int(generation_kwargs.get("max_new_tokens", 128))
        generation_kwargs["max_new_tokens"] = min(requested, remaining)
        out = self.session.generate_ids(ids, reset_cache=True, **generation_kwargs)
        answer = self.tokenizer.decode(out, skip_special_tokens=True)
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def stream(self, text: str, **generation_kwargs) -> Iterator[str]:
        if self.gguf:
            self.messages.append({"role": "user", "content": text})
            pieces = []
            for chunk in self._llama.create_chat_completion(
                messages=self.messages,
                max_tokens=int(generation_kwargs.get("max_new_tokens", 128)),
                temperature=float(generation_kwargs.get("temperature", 0.7)),
                top_p=float(generation_kwargs.get("top_p", 0.9)),
                top_k=int(generation_kwargs.get("top_k", 40)),
                stream=True,
            ):
                piece = chunk["choices"][0].get("delta", {}).get("content", "")
                if piece:
                    pieces.append(piece)
                    yield piece
            self.messages.append({"role": "assistant", "content": "".join(pieces)})
            return
        self.messages.append({"role": "user", "content": text})
        ids = self._encode_chat(self.messages)
        remaining = max(0, int(self.session.info.get("context_length", len(ids))) - len(ids))
        requested = int(generation_kwargs.get("max_new_tokens", 128))
        generation_kwargs["max_new_tokens"] = min(requested, remaining)
        pieces = []
        for token in self.session.stream_ids(ids, **generation_kwargs):
            piece = self.tokenizer.decode([token], skip_special_tokens=False)
            if piece:
                pieces.append(piece)
                yield piece
        self.messages.append({"role": "assistant", "content": "".join(pieces)})

    def reset(self):
        if self.gguf:
            self.messages = []
            return
        self.session.reset()
        self.messages = []


class ZornModelManager:
    def __init__(self):
        self.models = {}

    def load(self, name: str, model_dir: str, cpu_cores: Optional[int] = None) -> ZornSession:
        if name not in self.models:
            self.models[name] = ZornSession(model_dir, cpu_cores=cpu_cores)
        return self.models[name]

    def unload(self, name: str):
        session = self.models.pop(name, None)
        if session:
            session.close()

    def list_models(self):
        return {k: v.info for k, v in self.models.items()}


def benchmark(session: ZornSession, input_ids: Sequence[int], **kwargs) -> dict:
    start = time.perf_counter()
    out = session.generate_ids(input_ids, **kwargs)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return {
        "generated_tokens": len(out),
        "seconds": elapsed,
        "tokens_per_second": len(out) / elapsed,
        "model": session.info,
    }


def generate_batch(session: ZornSession, batch_input_ids: Iterable[Sequence[int]], **kwargs) -> list[list[int]]:
    """Run multiple independent prompts sequentially without reloading the model."""
    results = []
    for ids in batch_input_ids:
        results.append(session.generate_ids(ids, reset_cache=True, **kwargs))
    return results
