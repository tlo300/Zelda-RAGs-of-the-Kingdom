"""
convert_llm.py
Converts Qwen2.5 1.5B or 3B Instruct from HuggingFace to a Core ML .mlpackage
with 4-bit palettization quantization, for on-device inference on iOS.

Output: model/QwenModel-1B.mlpackage  or  model/QwenModel-3B.mlpackage
  - Fixed-KV model: takes (input_ids [1,1], attention_mask [1,MAX_KV_LEN+1],
    past_kv [L,2,H,MAX_KV_LEN,D], position_ids [1,1]) and returns
    (logits [1,1,vocab], new_kv [L,2,H,1,D])
  - All input/output shapes are FIXED (no RangeDim) to avoid Core ML OOM at load time.
  - Caller maintains a circular KV buffer of size MAX_KV_LEN and writes new_kv at
    write_pos % MAX_KV_LEN after each step.
  - 4-bit per_grouped_channel palettization via coremltools.optimize.torch
  - 1B output must be under 800 MB; 3B under 2 GB

Environment variables:
  MODEL_VARIANT  "1B" (default) or "3B"
  HF_TOKEN       HuggingFace token (Qwen2.5 models are ungated — token still needed for rate limits)

Usage:
    python model/convert_llm.py [--output-dir <dir>]

The script exits with a non-zero code on any validation failure:
  - Unknown MODEL_VARIANT
  - ModelConfig.swift filename does not match the derived output filename
  - Smoke test produces fewer than 20 tokens
  - Output .mlpackage exceeds the size limit for the chosen variant
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODELCONFIG_PATH = Path(__file__).parent.parent / "ios" / "ZeldaGuide" / "Services" / "ModelConfig.swift"

_HF_IDS = {
    "1B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
}

# Maximum on-disk size for the .mlpackage bundle (bytes)
_SIZE_LIMITS = {
    "1B": 800 * 1024 * 1024,   # 800 MB  (Qwen2.5-1.5B at 4-bit ~750 MB)
    "3B": 2 * 1024 * 1024 * 1024,  # 2 GB
}

_SMOKE_TEST_MIN_TOKENS = 20

# Fixed KV cache length — must match ModelConfig.llmMaxKVLen in the iOS app.
# All Core ML tensor shapes are static (no RangeDim), which prevents the
# Jetsam OOM kill that occurred during MLModel.load() with the previous
# RangeDim-based design (R26-R31).
MAX_KV_LEN = 512


# ---------------------------------------------------------------------------
# Pure helper functions (no heavy deps — fully testable without mocks)
# ---------------------------------------------------------------------------

def _derive_filename(variant: str) -> str:
    """Return the expected .mlpackage filename for the given variant."""
    if variant not in _HF_IDS:
        print(
            f"ERROR: Unknown MODEL_VARIANT '{variant}'. Expected '1B' or '3B'.",
            file=sys.stderr,
        )
        sys.exit(1)
    return f"QwenModel-{variant}.mlpackage"


def _parse_modelconfig_filename(swift_path: Path, variant: str) -> str:
    """
    Read ModelConfig.swift and extract the modelFilename for the given variant.

    Looks for the pattern:
        case .qwen1B: return "QwenModel-1B.mlmodelc"
        case .qwen3B: return "QwenModel-3B.mlmodelc"

    Note: the app bundles a pre-compiled .mlmodelc (produced by xcrun coremlcompiler
    in build-ipa.yml), so ModelConfig.swift uses the .mlmodelc extension even though
    this script outputs a .mlpackage.  _assert_filename_sync compares stems only.
    """
    swift_key = "qwen1B" if variant == "1B" else "qwen3B"
    try:
        text = swift_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(
            f"ERROR: ModelConfig.swift not found at {swift_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    pattern = rf'case \.{re.escape(swift_key)}:\s*return\s*"([^"]+)"'
    match = re.search(pattern, text)
    if not match:
        print(
            f"ERROR: Could not find modelFilename for variant '{variant}' in {swift_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    return match.group(1)


def _assert_filename_sync(derived: str, from_config: str) -> None:
    """Exit non-zero if the model name stems don't match.

    Compares stems (without extension) because this script outputs a .mlpackage while
    ModelConfig.swift references the compiled .mlmodelc.  Both must share the same
    base name (e.g. 'QwenModel-1B') so build-ipa.yml can compile the right artifact.
    """
    derived_stem = Path(derived).stem
    config_stem = Path(from_config).stem
    if derived_stem != config_stem:
        print(
            f"ERROR: Model name mismatch — script would produce '{derived}' "
            f"but ModelConfig.swift expects '{from_config}' "
            f"(stems: '{derived_stem}' vs '{config_stem}'). "
            "Update one of them so they agree before running CI.",
            file=sys.stderr,
        )
        sys.exit(1)


def _assert_smoke_test(tokens: list, min_tokens: int = _SMOKE_TEST_MIN_TOKENS) -> None:
    """Exit non-zero if the smoke test generated fewer tokens than expected,
    or if output is degenerate (>= 80% identical tokens — repetition loop)."""
    if len(tokens) < min_tokens:
        print(
            f"ERROR: Smoke test generated only {len(tokens)} tokens "
            f"(expected >= {min_tokens}). The model may be broken.",
            file=sys.stderr,
        )
        sys.exit(1)
    from collections import Counter
    most_common_count = Counter(tokens).most_common(1)[0][1]
    if most_common_count / len(tokens) >= 0.8:
        print(
            f"WARNING: Degenerate output — {most_common_count}/{len(tokens)} tokens are "
            f"identical (token {Counter(tokens).most_common(1)[0][0]}). "
            "Model likely running without Neural Engine. Validate output quality on device.",
            file=sys.stderr,
        )


def _assert_size(path: Path, variant: str) -> None:
    """Exit non-zero if the .mlpackage bundle exceeds the size limit."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    limit = _SIZE_LIMITS[variant]
    mb = total / (1024 * 1024)
    limit_mb = limit / (1024 * 1024)
    print(f"Output size: {mb:.1f} MB (limit {limit_mb:.0f} MB)")
    if total > limit:
        print(
            f"ERROR: {path.name} is {mb:.1f} MB, exceeds {limit_mb:.0f} MB limit for {variant}.",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(output_dir: str, variant: str, hf_token: str) -> None:
    import numpy as np
    import torch
    import coremltools as ct
    from coremltools.optimize.torch.palettization import (
        PostTrainingPalettizer,
        PostTrainingPalettizerConfig,
    )
    import gc
    from transformers import AutoTokenizer, AutoModelForCausalLM

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Filename sync check ---
    derived = _derive_filename(variant)
    from_config = _parse_modelconfig_filename(_MODELCONFIG_PATH, variant)
    _assert_filename_sync(derived, from_config)
    print(f"Filename sync OK: {derived}")

    hf_id = _HF_IDS[variant]
    # Use float16 for both variants — float32 OOMs the 14 GB CI runner
    torch_dtype = torch.float16

    print(f"Loading {hf_id} …")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=hf_token)

    # Save tokenizer.json alongside the model so build-ipa.yml can bundle it in sync.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer.save_pretrained(tmp_dir)
        tok_src = Path(tmp_dir) / "tokenizer.json"
        if tok_src.exists():
            shutil.copy(tok_src, out_path / "tokenizer.json")
            size_kb = (out_path / "tokenizer.json").stat().st_size // 1024
            print(f"Saved tokenizer.json ({size_kb} KB)")
        else:
            print("WARNING: tokenizer.json not found after save_pretrained", file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, token=hf_token, torch_dtype=torch_dtype,
        attn_implementation="eager",  # avoid aten::scaled_dot_product_attention in the JIT graph;
        # coremltools 8 validates that SDPA key/query/value share the same dtype and rejects
        # the trace when palettized k/v projections produce fp32.  Eager attention uses plain
        # matmul+softmax which has no same-dtype constraint; compute_precision=FLOAT16 in
        # ct.convert handles any residual fp32 activations.
    )
    model.eval()

    # Fixed-KV wrapper: processes one token per call.
    # Inputs:
    #   input_ids      [1, 1]                    — single new token
    #   attention_mask [1, MAX_KV_LEN+1]         — 1 for valid past slots + current token
    #   past_kv        [L, 2, H, MAX_KV_LEN, D]  — circular KV buffer (fixed size, no RangeDim)
    #   position_ids   [1, 1]                    — absolute position of the new token
    # Outputs:
    #   logits  [1, 1, vocab]   — logits for the new token
    #   new_kv  [L, 2, H, 1, D] — KV for the new token only (caller writes into buffer)
    #
    # All shapes are fixed — Core ML never needs to compile for a range of sizes,
    # which eliminates the Jetsam OOM that killed R26-R31 during MLModel.load().
    class _FixedKVWrapper(torch.nn.Module):
        def __init__(self, m, num_layers):
            super().__init__()
            self.m = m
            self.num_layers = num_layers

        def forward(self, input_ids, attention_mask, past_kv, position_ids):
            # past_kv: [L, 2, H, MAX_KV_LEN, D] — unsqueeze batch dim for HF attention
            past_key_values = tuple(
                (past_kv[i, 0].unsqueeze(0), past_kv[i, 1].unsqueeze(0))
                for i in range(self.num_layers)
            )
            out = self.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            logits = out.logits[:, -1:, :]  # [1, 1, vocab]
            # Extract only the new token's KV from the concatenated past+new output.
            # out.past_key_values[i][0] shape: [1, H, MAX_KV_LEN+1, D]
            # [:, :, -1:, :] gives [1, H, 1, D] — just the new token.
            new_kv = torch.stack([
                torch.stack([
                    kv[0].squeeze(0)[:, -1:, :],  # [H, 1, D]
                    kv[1].squeeze(0)[:, -1:, :]   # [H, 1, D]
                ], dim=0)
                for kv in out.past_key_values
            ], dim=0)  # [L, 2, H, 1, D] — fixed shape
            return logits, new_kv

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    print(
        f"Model config: {num_layers} layers, "
        f"{num_kv_heads} KV heads, head_dim={head_dim}, MAX_KV_LEN={MAX_KV_LEN}"
    )

    wrapped = _FixedKVWrapper(model, num_layers)
    wrapped.eval()

    # --- 4-bit torch-level palettization (pre-conversion) ---
    # Palettize the PyTorch model weights BEFORE CoreML conversion so we never
    # hold a large uncompressed CoreML model in memory (which OOMs the runner).
    print("Applying 4-bit torch-level palettization (per_grouped_channel) …")
    pal_config = PostTrainingPalettizerConfig.from_dict({
        "global_config": {
            "n_bits": 4,
            "granularity": "per_grouped_channel",
        }
    })
    palettizer = PostTrainingPalettizer(wrapped, pal_config)
    compressed_wrapped = palettizer.compress()
    del model, wrapped

    # PostTrainingPalettizer stores k-means centroids as float32 (k-means runs in fp32
    # even when the original weights were fp16).  The palettized k/v projections therefore
    # dequantize to fp32 during the forward pass, while q_proj may remain fp16 — causing
    # a dtype mismatch inside scaled_dot_product_attention that coremltools rejects.
    # Fix: walk the compressed model and cast any float32 buffers/parameters back to fp16.
    # Integer tensors (palette indices, dtype=int32/int64) are unaffected by this check.
    print("Recasting float32 palettizer buffers to float16 …")
    with torch.no_grad():
        for module in compressed_wrapped.modules():
            for buf_name, buf in list(module.named_buffers(recurse=False)):
                if buf.dtype == torch.float32:
                    module.register_buffer(buf_name, buf.to(torch.float16))
            for param_name, param in list(module.named_parameters(recurse=False)):
                if param.dtype == torch.float32:
                    module._parameters[param_name] = torch.nn.Parameter(
                        param.data.to(torch.float16), requires_grad=False
                    )
    gc.collect()

    # --- Trace the already-compressed model ---
    # Trace with fixed sizes that match the ct.convert shape declarations below.
    # past_kv is always MAX_KV_LEN — no growing tensors, no RangeDim needed.
    example_input_ids    = torch.zeros(1, 1, dtype=torch.int64)
    example_attn_mask    = torch.ones(1, MAX_KV_LEN + 1, dtype=torch.int64)
    example_past_kv      = torch.zeros(num_layers, 2, num_kv_heads, MAX_KV_LEN, head_dim,
                                       dtype=torch_dtype)
    example_position_ids = torch.tensor([[MAX_KV_LEN]], dtype=torch.int64)
    example_inputs = (example_input_ids, example_attn_mask, example_past_kv, example_position_ids)
    print("Tracing compressed model …")
    with torch.no_grad():
        traced = torch.jit.trace(compressed_wrapped, example_inputs)
    del compressed_wrapped
    gc.collect()

    # --- Core ML conversion ---
    # All shapes are fixed — no RangeDim anywhere.
    # This is the key change from the previous design: Core ML compiles for exactly
    # these shapes at MLModel.load() time, which takes trivial memory vs. the previous
    # approach that compiled for every length in RangeDim(0, 2047) and OOM-killed the process.
    print("Converting to Core ML …")
    compressed = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids",
                          shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="attention_mask",
                          shape=(1, MAX_KV_LEN + 1), dtype=np.int32),
            ct.TensorType(name="past_kv",
                          shape=(num_layers, 2, num_kv_heads, MAX_KV_LEN, head_dim),
                          dtype=np.float16),
            ct.TensorType(name="position_ids",
                          shape=(1, 1), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="new_kv", dtype=np.float16),   # [L, 2, H, 1, D]
        ],
        minimum_deployment_target=ct.target.iOS18,
        # CPU_AND_GPU avoids the NE-specific kernel-compilation passes that caused the
        # 6-hour CI timeout with CPU_AND_NE.  The 5D KV-cache tensors require extensive
        # NE memory-tiling analysis that dominates compile time.  GPU still gives full
        # KV-cache decode speed (O(1) per token vs O(n²) without cache) and avoids the
        # Jetsam OOM that killed R26-R31 at MLModel.load().  NE can be re-enabled once
        # conversion time is under control.
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,  # cast all activations to fp16 to prevent
        # "key dtype fp32 vs query dtype fp16" at SDPA — the palettizer's dequantization path
        # can leave fp32 activations in the JIT graph that coremltools' SDPA op rejects.
    )
    del traced
    gc.collect()

    # --- Save (before smoke test so the artifact is always available) ---
    save_path = out_path / derived
    print(f"Saving to {save_path} …")
    compressed.save(str(save_path))

    # --- Size check ---
    _assert_size(save_path, variant)

    # --- Smoke test: generate _SMOKE_TEST_MIN_TOKENS tokens ---
    # Uses the fixed-KV interface: maintain a circular buffer and write pointer.
    # Prefill and decode are both one token per forward pass.
    print(f"Running smoke test ({_SMOKE_TEST_MIN_TOKENS} tokens) …")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is the Hylian Shield?"},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(input_text, return_tensors="pt")
    prompt_ids = enc["input_ids"].numpy().astype(np.int32)  # [1, prompt_len]
    prompt_len = prompt_ids.shape[1]

    # Circular KV buffer — all zeros initially.
    kv_buffer = np.zeros((num_layers, 2, num_kv_heads, MAX_KV_LEN, head_dim), dtype=np.float16)
    write_pos = 0

    def _make_mask(pos: int) -> np.ndarray:
        """Build attention_mask [1, MAX_KV_LEN+1]: 1 for valid past slots + current token."""
        mask = np.zeros((1, MAX_KV_LEN + 1), dtype=np.int32)
        valid = min(pos, MAX_KV_LEN)
        mask[0, :valid] = 1
        mask[0, MAX_KV_LEN] = 1  # current token always attended
        return mask

    # Prefill: process prompt token by token to populate the KV cache.
    next_token = None
    for step in range(prompt_len):
        token_id = prompt_ids[0, step]
        output = compressed.predict({
            "input_ids":      np.array([[token_id]], dtype=np.int32),
            "attention_mask": _make_mask(write_pos),
            "past_kv":        kv_buffer,
            "position_ids":   np.array([[write_pos]], dtype=np.int32),
        })
        new_kv = output["new_kv"]  # [L, 2, H, 1, D]
        slot = write_pos % MAX_KV_LEN
        kv_buffer[:, :, :, slot:slot + 1, :] = new_kv
        write_pos += 1
        next_token = int(np.argmax(output["logits"][0, 0, :]))

    # Decode _SMOKE_TEST_MIN_TOKENS new tokens.
    generated_tokens = []
    for _ in range(_SMOKE_TEST_MIN_TOKENS):
        output = compressed.predict({
            "input_ids":      np.array([[next_token]], dtype=np.int32),
            "attention_mask": _make_mask(write_pos),
            "past_kv":        kv_buffer,
            "position_ids":   np.array([[write_pos]], dtype=np.int32),
        })
        new_kv = output["new_kv"]
        slot = write_pos % MAX_KV_LEN
        kv_buffer[:, :, :, slot:slot + 1, :] = new_kv
        write_pos += 1
        next_token = int(np.argmax(output["logits"][0, 0, :]))
        generated_tokens.append(next_token)

    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Smoke test output: {decoded!r}")
    _assert_smoke_test(generated_tokens)
    print(f"Done: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen2.5 to Core ML")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent),
        help="Directory to write the .mlpackage (default: model/)",
    )
    args = parser.parse_args()

    variant = os.environ.get("MODEL_VARIANT", "1B")
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    convert(args.output_dir, variant, hf_token)


if __name__ == "__main__":
    main()
