# ZORN-BitNet v4.1

ZORN is a native C inference engine for ternary BitNet b1.58 / I2_S models, with Falcon3-10B-1.58bit as the primary target.

## v4.1 features

- 2-bit packed ternary weights `{-1,0,+1}` with I2_S group scales.
- Memory-mapped model weights.
- Falcon3 GQA, RoPE, RMSNorm, gated MLP and KV cache.
- Scalar/AVX2/AVX-512 SIMD detection.
- Persistent native Python sessions: load once, generate many times.
- Prompt-aware generation using token IDs from a matching Hugging Face tokenizer.
- Temperature, top-k, top-p and repetition penalty sampling.
- Seed control and stop-token support.
- Incremental `step()` API for streaming generation.
- High-level `ZornChat` API with chat templates and context-window management.
- Model validation and model manager for multiple loaded models.
- Batch inference helper (sequential, shared model, independent cache per prompt).
- Benchmark/metrics helper.
- CLI chat interface.
- Optional OpenAI-compatible `/v1/chat/completions` server.
- Existing RAG folder ingestion and JSONL logging remain available.
- Missing weight files now fail explicitly instead of silently creating placeholder weights.

## Build

```bash
python3 -m pip install -U setuptools wheel numpy
python3 setup.py build_ext --inplace
python3 -c "import zorn_engine; print('ZORN-BitNet OK')"
```

For the high-level Python API:

```bash
pip install -r requirements-extra.txt
```

## Python chat

```python
from zorn import ZornChat

chat = ZornChat(
    "zorn_model",
    cpu_cores=16,
    system_prompt="You are a helpful assistant."
)

print(chat.ask(
    "Hello!",
    max_new_tokens=128,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.05,
))
```

Streaming:

```python
for piece in chat.stream("Explain BitNet simply.", max_new_tokens=128):
    print(piece, end="", flush=True)
```

## Native API

```python
import zorn_engine
s = zorn_engine.load_model("zorn_model", cpu_cores=16)
print(zorn_engine.info(s))
ids = [/* tokenizer output */]
out = zorn_engine.generate(
    s, ids,
    max_new_tokens=128,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.05,
    reset_cache=True,
)
```

For incremental generation, call `step(session, token_id, reset_cache=False, ...)` after the initial prompt prefill.

## CLI

```bash
python zorn_cli.py zorn_model --threads 16
```

## OpenAI-compatible server

```bash
pip install fastapi uvicorn
uvicorn 'zorn_server:create_app("zorn_model")' --host 0.0.0.0 --port 8000
```

The endpoint is:

```text
POST /v1/chat/completions
```

This server is intentionally single-session by default. Add a session pool before using it for concurrent production traffic.

## Model conversion

The converter produces the ZORN model directory containing `weights.zrn`, `scales.zrn`, floating-point embedding/lm-head/norm files, layer files, `zorn_manifest.json`, and `tokenizer.json`.

## Important limitation

The current 4.1 native backend is CPU/SIMD based. It detects AVX2/AVX-512 but does **not** contain a CUDA kernel backend. A real CUDA backend requires separate GPU kernels and memory/layout work; this release does not pretend that a Colab GPU is being used by the CPU engine.

Numerical equivalence should be validated against an official/reference BitNet implementation before treating ZORN as production-equivalent.


## v4.2 platform/model support

This release includes portable Windows/macOS/Linux/Android build support, automatic raw Safetensors preparation, and an optional GGUF backend through llama.cpp.
