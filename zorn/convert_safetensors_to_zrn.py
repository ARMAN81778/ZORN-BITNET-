#!/usr/bin/env python3
"""Convert BitNet/Falcon3 safetensors checkpoints to the ZORN model format."""

import argparse
import json
import os
import shutil
import struct
import sys

try:
    import numpy as np
except ImportError:
    print("numpy is required: pip install numpy")
    sys.exit(1)

LAYER_PREFIX_PATTERNS = ["model.layers.{i}.", "transformer.h.{i}.", "layers.{i}."]
TERNARY_FIELD_SUFFIXES = {
    "q_proj": ["self_attn.q_proj.weight", "attn.q_proj.weight"],
    "k_proj": ["self_attn.k_proj.weight", "attn.k_proj.weight"],
    "v_proj": ["self_attn.v_proj.weight", "attn.v_proj.weight"],
    "o_proj": ["self_attn.o_proj.weight", "attn.o_proj.weight"],
    "gate_proj": ["mlp.gate_proj.weight"],
    "up_proj": ["mlp.up_proj.weight"],
    "down_proj": ["mlp.down_proj.weight"],
}
SCALE_SUFFIXES = {k: [s.replace(".weight", ".weight_scale") for s in v]
                  for k, v in TERNARY_FIELD_SUFFIXES.items()}
NORM_SUFFIXES = {
    "input_layernorm": ["input_layernorm.weight", "ln_1.weight"],
    "post_attention_layernorm": ["post_attention_layernorm.weight", "ln_2.weight"],
}
EMBED_NAMES = ["model.embed_tokens.weight", "transformer.wte.weight", "embed_tokens.weight"]
LM_HEAD_NAMES = ["lm_head.weight", "model.lm_head.weight"]
FINAL_NORM_NAMES = ["model.norm.weight", "transformer.ln_f.weight", "norm.weight"]
DTYPE_MAP = {
    "F32": np.float32, "F16": np.float16, "I8": np.int8, "U8": np.uint8,
    "I32": np.int32, "I64": np.int64, "BOOL": np.bool_,
}


def read_safetensors_header(path):
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header, 8 + header_len


def load_tensor_raw(default_path, data_start, meta):
    path = meta.get("_file", default_path)
    dtype_name = meta["dtype"]
    shape = meta["shape"]
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(meta["_data_start"])
        f.seek(start, os.SEEK_CUR)
        raw = f.read(end - start)
    if dtype_name == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2")
        arr = (u16.astype(np.uint32) << 16).view(np.float32)
    else:
        np_dtype = DTYPE_MAP.get(dtype_name)
        if np_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_name}")
        arr = np.frombuffer(raw, dtype=np_dtype)
    return arr.reshape(shape)


def load_model(path):
    if os.path.isfile(path):
        header, data_start = read_safetensors_header(path)
        for meta in header.values():
            if isinstance(meta, dict):
                meta["_file"] = path
                meta["_data_start"] = data_start
        return header, path

    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        header = {}
        for filename in sorted(set(index["weight_map"].values())):
            shard_path = os.path.join(path, filename)
            shard_header, data_start = read_safetensors_header(shard_path)
            for name, meta in shard_header.items():
                if name == "__metadata__":
                    continue
                meta["_file"] = shard_path
                meta["_data_start"] = data_start
                header[name] = meta
        return header, path

    candidates = sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.endswith(".safetensors")
    )
    if len(candidates) == 1:
        return load_model(candidates[0])
    if candidates:
        raise ValueError(
            "Multiple safetensors files found. Provide the model directory "
            "with model.safetensors.index.json or specify one shard."
        )
    raise FileNotFoundError("No safetensors checkpoint found")


def find_tensor_name(header, candidates):
    return next((name for name in candidates if name in header), None)


def find_layer_tensor_name(header, layer_idx, suffix_candidates):
    for pattern in LAYER_PREFIX_PATTERNS:
        prefix = pattern.format(i=layer_idx)
        for suffix in suffix_candidates:
            name = prefix + suffix
            if name in header:
                return name
    return None


def discover(path):
    header, _ = load_model(path)
    print(f"\n=== tensors: {path} ===\n")
    total = 0
    count = 0
    for name, meta in sorted(header.items()):
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        total += end - start
        count += 1
        print(f"  {name:60s} dtype={meta['dtype']:5s} shape={meta['shape']}")
    print(f"\n{count} tensors, {total / 1024 / 1024:.1f} MB\n")

    print("--- detected ---")
    print(f"  embedding: {find_tensor_name(header, EMBED_NAMES)}")
    print(f"  lm_head: {find_tensor_name(header, LM_HEAD_NAMES)}")
    print(f"  final_norm: {find_tensor_name(header, FINAL_NORM_NAMES)}")

    layer_count = 0
    for i in range(200):
        found = find_layer_tensor_name(header, i, NORM_SUFFIXES["input_layernorm"])
        if found:
            layer_count = i + 1
        elif i > 0:
            break
    print(f"  layers: {layer_count}")


def pack_ternary_2bit(int_array):
    flat = int_array.flatten()
    n = flat.size
    pad = (-n) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    code_map = {0: 0, 1: 1, -1: 2}
    codes = np.vectorize(code_map.get)(flat).astype(np.uint8).reshape(-1, 4)
    packed = (codes[:, 0] | (codes[:, 1] << 2) |
              (codes[:, 2] << 4) | (codes[:, 3] << 6)).astype(np.uint8)
    return packed, n


def get_scale_tensor(header, path, data_start, layer_idx, field):
    for suffix in SCALE_SUFFIXES[field]:
        for pattern in LAYER_PREFIX_PATTERNS:
            name = pattern.format(i=layer_idx) + suffix
            if name in header:
                meta = header[name]
                arr = np.asarray(
                    load_tensor_raw(path, data_start, meta), dtype=np.float32
                ).reshape(-1)
                return arr, name
    return np.asarray([1.0], dtype=np.float32), None


def quantize_absmean_ternary(float_arr, group_size=64):
    """BitNet-style absmean ternary quantization (per the BitNet b1.58 paper):
    for each group of `group_size` input weights, scale = mean(|w|), then
    round(clip(w/scale, -1, 1)) -> {-1, 0, 1}. Returns (int8 codes, float32
    per-(out_feature, group) scales) with the SAME shape convention the ZORN
    loader already expects via scale_offset/group_size/num_groups.
    """
    arr = np.asarray(float_arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D [out_features, in_features] tensor, got shape {arr.shape}")
    out_features, in_features = arr.shape
    num_groups = (in_features + group_size - 1) // group_size
    pad = num_groups * group_size - in_features
    if pad:
        arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant")
    grouped = arr.reshape(out_features, num_groups, group_size)

    eps = 1e-8
    scales = np.mean(np.abs(grouped), axis=2) + eps          # [out_features, num_groups]
    normalized = grouped / scales[:, :, None]
    codes = np.clip(np.round(normalized), -1, 1).astype(np.int8)

    codes = codes.reshape(out_features, num_groups * group_size)[:, :in_features]
    return codes, scales.astype(np.float32), num_groups


def normalize_ternary(arr):
    arr = np.asarray(arr)
    values = np.unique(arr)
    bad = [int(v) for v in values if int(v) not in (-1, 0, 1)]
    if bad:
        raise ValueError(f"Non-ternary weights found: {bad[:16]}")
    return arr.astype(np.int8, copy=False)


def convert(path, out_dir, hidden_size_override=None, head_dim=256,
            rope_theta=1000042.0, max_ctx=32768):
    header, _ = load_model(path)
    os.makedirs(out_dir, exist_ok=True)

    emb_name = find_tensor_name(header, EMBED_NAMES)
    lm_name = find_tensor_name(header, LM_HEAD_NAMES)
    final_norm_name = find_tensor_name(header, FINAL_NORM_NAMES)

    if not emb_name:
        raise ValueError("Embedding tensor not found")
    if not lm_name:
        raise ValueError("lm_head tensor not found")

    emb_meta = header[emb_name]
    vocab_size, hidden_size = emb_meta["shape"]
    if hidden_size_override:
        hidden_size = hidden_size_override

    emb_arr = load_tensor_raw(path, 0, emb_meta).astype(np.float32)
    emb_arr.tofile(os.path.join(out_dir, "embedding.bin"))

    lm_meta = header[lm_name]
    if list(lm_meta.get("shape", [])) != [int(vocab_size), int(hidden_size)]:
        raise ValueError(
            f"lm_head shape {lm_meta.get('shape')} does not match "
            f"[vocab_size, hidden_size]=[{vocab_size}, {hidden_size}]"
        )
    lm_arr = load_tensor_raw(path, 0, lm_meta).astype(np.float32)
    lm_arr.tofile(os.path.join(out_dir, "lm_head.bin"))

    if final_norm_name:
        fn_arr = load_tensor_raw(path, 0, header[final_norm_name]).astype(np.float32)
    else:
        fn_arr = np.ones(int(hidden_size), dtype=np.float32)
    fn_arr.tofile(os.path.join(out_dir, "final_norm.bin"))

    layer_count = 0
    for i in range(200):
        if find_layer_tensor_name(header, i, NORM_SUFFIXES["input_layernorm"]):
            layer_count = i + 1
        elif i > 0:
            break

    group_size_default = 64
    manifest = {
        "config": {
            "hidden_size": int(hidden_size),
            "num_layers": layer_count,
            "vocab_size": int(vocab_size),
        },
        "embedding": {
            "file": "embedding.bin",
            "vocab_size": int(vocab_size),
            "hidden_size": int(hidden_size),
        },
        "lm_head": {
            "file": "lm_head.bin",
            "vocab_size": int(vocab_size),
            "hidden_size": int(hidden_size),
        },
        "quantization": {
            "method": "bitnet_i2_s",
            "bits": 2,
            "group_size": group_size_default,
            "weight_values": [-1, 0, 1],
        },
        "layers": [],
    }

    weights_chunks = []
    scales_chunks = []
    cursor_weights_bytes = 0
    cursor_scales_bytes = 0

    for li in range(layer_count):
        layer_manifest = {"index": li}

        for norm_key, suffixes in NORM_SUFFIXES.items():
            tname = find_layer_tensor_name(header, li, suffixes)
            if not tname:
                raise ValueError(f"Missing layer {li} tensor: {norm_key}")
            arr = load_tensor_raw(path, 0, header[tname]).astype(np.float32)
            fname = f"layer{li}_{'input_ln' if 'input' in norm_key else 'post_ln'}.bin"
            arr.tofile(os.path.join(out_dir, fname))
            layer_manifest[norm_key] = fname

        for field, suffixes in TERNARY_FIELD_SUFFIXES.items():
            tname = find_layer_tensor_name(header, li, suffixes)
            if not tname:
                raise ValueError(f"Missing layer {li} tensor: {field}")

            meta = header[tname]
            arr = load_tensor_raw(path, 0, meta)

            if meta["dtype"] in ("I8", "U8"):
                # Already-quantized ternary checkpoint (matches the original
                # ZORN converter behavior): validate and use scale tensors
                # as-is if present.
                int_arr = normalize_ternary(arr)
                if int_arr.ndim != 2:
                    raise ValueError(f"{field} must be a 2D matrix, got shape {int_arr.shape}")
                out_features, in_features = int_arr.shape
                packed, n = pack_ternary_2bit(int_arr)
                weights_chunks.append(packed)

                scale_arr, scale_name = get_scale_tensor(header, path, 0, li, field)
                expected_groups = (in_features + group_size_default - 1) // group_size_default

                if scale_arr.size == 1:
                    scale = float(scale_arr[0])
                    scale_offset = -1
                    num_groups = 0
                    group_size = 0
                else:
                    if scale_arr.size == out_features * expected_groups:
                        scale_arr = scale_arr.reshape(out_features, expected_groups)
                    elif scale_arr.size == expected_groups:
                        scale_arr = np.broadcast_to(
                            scale_arr.reshape(1, -1),
                            (out_features, expected_groups),
                        ).copy()
                    elif scale_arr.size % out_features == 0:
                        scale_arr = scale_arr.reshape(out_features, -1)
                        expected_groups = scale_arr.shape[1]
                    else:
                        raise ValueError(
                            f"Cannot map scale tensor {scale_name}: {scale_arr.shape}"
                        )

                    flat_scales = np.asarray(scale_arr, dtype=np.float32).reshape(-1)
                    scales_chunks.append(flat_scales)
                    scale_offset = cursor_scales_bytes
                    cursor_scales_bytes += flat_scales.nbytes
                    scale = float(flat_scales[0])
                    group_size = group_size_default
                    num_groups = expected_groups
            elif meta["dtype"] in ("F32", "F16", "BF16"):
                # Real HF/BitNet checkpoints usually store weights as plain
                # float, not pre-ternarized. Quantize on the fly with the
                # standard BitNet b1.58 absmean scheme instead of failing.
                float_arr = np.asarray(arr, dtype=np.float32)
                if float_arr.ndim != 2:
                    raise ValueError(f"{field} must be a 2D matrix, got shape {float_arr.shape}")
                out_features, in_features = float_arr.shape
                codes, group_scales, num_groups = quantize_absmean_ternary(
                    float_arr, group_size=group_size_default
                )
                packed, n = pack_ternary_2bit(codes)
                weights_chunks.append(packed)

                flat_scales = group_scales.reshape(-1)
                scales_chunks.append(flat_scales)
                scale_offset = cursor_scales_bytes
                cursor_scales_bytes += flat_scales.nbytes
                scale = float(flat_scales[0])
                group_size = group_size_default
            else:
                raise ValueError(
                    f"{field} has unsupported dtype {meta['dtype']} "
                    "(expected I8/U8 pre-quantized ternary, or F32/F16/BF16 to "
                    "quantize on the fly)"
                )

            byte_offset = cursor_weights_bytes
            layer_manifest[field] = {
                "file": "weights.zrn",
                "byte_offset": byte_offset,
                "out_features": int(out_features),
                "in_features": int(in_features),
                "scale": scale,
                "scale_offset": scale_offset,
                "group_size": group_size,
                "num_groups": num_groups,
                "quant_method": "bitnet_i2_s" if num_groups else "bitnet_ternary",
            }

            cursor_weights_bytes += packed.nbytes

        if li == 0:
            q_name = find_layer_tensor_name(header, 0, TERNARY_FIELD_SUFFIXES["q_proj"])
            k_name = find_layer_tensor_name(header, 0, TERNARY_FIELD_SUFFIXES["k_proj"])
            gate_name = find_layer_tensor_name(header, 0, TERNARY_FIELD_SUFFIXES["gate_proj"])
            if q_name and k_name:
                q_dim = header[q_name]["shape"][0]
                kv_dim = header[k_name]["shape"][0]
                manifest["config"]["num_attn_heads"] = q_dim // head_dim
                manifest["config"]["num_kv_heads"] = kv_dim // head_dim
                manifest["config"]["head_dim"] = head_dim
            if gate_name:
                manifest["config"]["mlp_intermediate"] = header[gate_name]["shape"][0]
            manifest["config"]["rope_theta"] = rope_theta
            manifest["config"]["rmsnorm_eps"] = 1e-6
            manifest["config"]["max_context_length"] = max_ctx

        manifest["layers"].append(layer_manifest)

    if weights_chunks:
        with open(os.path.join(out_dir, "weights.zrn"), "wb") as f:
            f.write(np.concatenate(weights_chunks).tobytes())

    if scales_chunks:
        with open(os.path.join(out_dir, "scales.zrn"), "wb") as f:
            f.write(np.concatenate(scales_chunks).astype(np.float32, copy=False).tobytes())

    with open(os.path.join(out_dir, "zorn_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    src_dir = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    tok_src = os.path.join(src_dir, "tokenizer.json")
    if os.path.isfile(tok_src):
        shutil.copy2(tok_src, os.path.join(out_dir, "tokenizer.json"))
    else:
        raise FileNotFoundError(
            f"tokenizer.json not found next to '{src_dir}'. "
            "A converted ZORN model cannot be loaded without the tokenizer."
        )

    print(f"Converted {layer_count} layers to {out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Convert BitNet/Falcon3 safetensors checkpoints to ZORN."
    )
    ap.add_argument(
        "model_path",
        help="A .safetensors file or a directory containing a safetensors index/checkpoint",
    )
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--hidden-size", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--rope-theta", type=float, default=1000042.0)
    ap.add_argument("--max-context", type=int, default=32768)
    args = ap.parse_args()

    if args.discover:
        discover(args.model_path)
        return
    if not args.out_dir:
        ap.error("out_dir is required unless --discover is used")

    convert(
        args.model_path,
        args.out_dir,
        args.hidden_size,
        args.head_dim,
        args.rope_theta,
        args.max_context,
    )


if __name__ == "__main__":
    main()
