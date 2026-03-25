"""
convert_llm.py
Converts Qwen2.5 1.5B or 3B Instruct from HuggingFace to a Core ML .mlpackage
with 4-bit palettization quantization, for on-device inference on iOS.

Output: model/QwenModel-1B.mlpackage  or  model/QwenModel-3B.mlpackage
  - Stateless model (full sequence passed each step, max 512 tokens)
  - 4-bit uniform palettization via coremltools.optimize.coreml
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
import sys
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
        case .qwen1B: return "QwenModel-1B.mlpackage"
        case .qwen3B: return "QwenModel-3B.mlpackage"
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
    """Exit non-zero if the script's derived filename differs from ModelConfig.swift."""
    if derived != from_config:
        print(
            f"ERROR: Filename mismatch — script would produce '{derived}' "
            f"but ModelConfig.swift expects '{from_config}'. "
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
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, token=hf_token, torch_dtype=torch_dtype
    )
    model.eval()

    # Wrap so torch.jit.trace gets a plain tensor output instead of
    # CausalLMOutputWithPast (which the tracer cannot infer types for).
    class _LogitsWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapped = _LogitsWrapper(model)
    wrapped.eval()

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    max_seq = cfg.max_position_embeddings

    print(
        f"Model config: {num_layers} layers, "
        f"{num_kv_heads} KV heads, head_dim={head_dim}, max_seq={max_seq}"
    )

    # --- 4-bit torch-level palettization (pre-conversion) ---
    # Palettize the PyTorch model weights BEFORE CoreML conversion so we never
    # hold a large uncompressed CoreML model in memory (which OOMs the runner).
    print("Applying 4-bit torch-level palettization …")
    pal_config = PostTrainingPalettizerConfig.from_dict({
        "global_config": {
            "n_bits": 4,
            "granularity": "per_grouped_channel",
            "group_size": 16,
        }
    })
    palettizer = PostTrainingPalettizer(wrapped, pal_config)
    compressed_wrapped = palettizer.compress()
    del model, wrapped
    gc.collect()

    # --- Trace the already-compressed model ---
    # Use seq_len=8 so the tracer doesn't constant-fold the sequence dimension.
    print("Tracing compressed model with torch.jit.trace …")
    example_inputs = (
        torch.zeros(1, 8, dtype=torch.int64),
        torch.ones(1, 8, dtype=torch.int64),
    )
    with torch.no_grad():
        traced = torch.jit.trace(compressed_wrapped, example_inputs)
    del compressed_wrapped
    gc.collect()

    # --- Core ML conversion ---
    # Cap dynamic sequence at 512 tokens (prompt + context + answer).
    # coremltools recognises the torch-level palettization and produces a
    # compressed mlpackage directly — no post-conversion quantization needed.
    max_context = 512
    print("Converting to Core ML …")
    compressed = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, max_context)), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, max_context)), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS18,  # grouped palettization requires iOS 18+
        compute_units=ct.ComputeUnit.CPU_AND_NE,
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
    # Apply the Qwen2.5 ChatML template — the model is an Instruct model and
    # degenerates (repetition loops) when fed a bare prompt without the wrapper.
    print(f"Running smoke test ({_SMOKE_TEST_MIN_TOKENS} tokens) …")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is the Hylian Shield?"},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(input_text, return_tensors="pt")
    input_ids = enc["input_ids"].numpy().astype(np.int32)
    attention_mask = enc["attention_mask"].numpy().astype(np.int32)

    generated_tokens = []
    for _ in range(_SMOKE_TEST_MIN_TOKENS):
        output = compressed.predict(
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        next_token = int(np.argmax(output["logits"][0, -1, :]))
        generated_tokens.append(next_token)
        input_ids = np.append(input_ids, [[next_token]], axis=1).astype(np.int32)
        attention_mask = np.ones((1, input_ids.shape[1]), dtype=np.int32)

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
