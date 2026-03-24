"""
convert_llm.py
Converts Llama 3.2 1B or 3B Instruct from HuggingFace to a Core ML .mlpackage
with 4-bit palettization quantization, for on-device inference on iOS.

Output: model/LlamaModel-1B.mlpackage  or  model/LlamaModel-3B.mlpackage
  - Stateful model with KV-cache state (requires iOS 18+)
  - 4-bit palettized weights via coremltools.optimize.coreml
  - 1B output must be under 700 MB; 3B under 2 GB

Environment variables:
  MODEL_VARIANT  "1B" (default) or "3B"
  HF_TOKEN       HuggingFace token — must have accepted Meta's Llama 3.2 licence

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
    "1B": "meta-llama/Llama-3.2-1B-Instruct",
    "3B": "meta-llama/Llama-3.2-3B-Instruct",
}

# Maximum on-disk size for the .mlpackage bundle (bytes)
_SIZE_LIMITS = {
    "1B": 700 * 1024 * 1024,   # 700 MB
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
    return f"LlamaModel-{variant}.mlpackage"


def _parse_modelconfig_filename(swift_path: Path, variant: str) -> str:
    """
    Read ModelConfig.swift and extract the modelFilename for the given variant.

    Looks for the pattern:
        case .llama1B: return "LlamaModel-1B.mlpackage"
        case .llama3B: return "LlamaModel-3B.mlpackage"
    """
    swift_key = "llama1B" if variant == "1B" else "llama3B"
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
    """Exit non-zero if the smoke test generated fewer tokens than expected."""
    if len(tokens) < min_tokens:
        print(
            f"ERROR: Smoke test generated only {len(tokens)} tokens "
            f"(expected >= {min_tokens}). The model may be broken.",
            file=sys.stderr,
        )
        sys.exit(1)


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
    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpPalettizerConfig,
        palettize_weights,
    )
    from transformers import AutoTokenizer, AutoModelForCausalLM

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Filename sync check ---
    derived = _derive_filename(variant)
    from_config = _parse_modelconfig_filename(_MODELCONFIG_PATH, variant)
    _assert_filename_sync(derived, from_config)
    print(f"Filename sync OK: {derived}")

    hf_id = _HF_IDS[variant]
    # 3B is too large in float32 on the 14 GB CI runner — use float16
    torch_dtype = torch.float16 if variant == "3B" else torch.float32

    print(f"Loading {hf_id} …")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, token=hf_token, torch_dtype=torch_dtype
    )
    model.eval()

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    max_seq = cfg.max_position_embeddings

    print(
        f"Model config: {num_layers} layers, "
        f"{num_kv_heads} KV heads, head_dim={head_dim}, max_seq={max_seq}"
    )

    # --- Export with torch.export (required for stateful KV-cache conversion) ---
    print("Exporting with torch.export …")
    example_inputs = (
        torch.zeros(1, 1, dtype=torch.int64),
        torch.zeros(1, 1, dtype=torch.int64),
    )
    seq_dim = torch.export.Dim("seq", min=1, max=max_seq)
    dynamic_shapes = {
        "input_ids": {1: seq_dim},
        "attention_mask": {1: seq_dim},
    }
    with torch.no_grad():
        exported = torch.export.export(
            model,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
        )

    # --- Core ML stateful conversion ---
    # Stateful models (KV cache via StateType) require iOS 18+.
    # iPhone 12 supports iOS 18, so this does not raise the minimum device.
    print("Converting to Core ML (stateful) …")
    kv_cache_shape = (num_layers, 2, 1, num_kv_heads, max_seq, head_dim)
    mlmodel = ct.convert(
        exported,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, max_seq)), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, max_seq)), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
        ],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=kv_cache_shape,
                    dtype=np.float16,
                ),
                name="kv_cache",
            )
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    # --- 4-bit palettization ---
    print("Applying 4-bit palettization …")
    config = OptimizationConfig(
        global_config=OpPalettizerConfig(
            mode="kmeans",
            nbits=4,
            weight_threshold=512,
        )
    )
    compressed = palettize_weights(mlmodel, config)

    # --- Smoke test: generate _SMOKE_TEST_MIN_TOKENS tokens ---
    print(f"Running smoke test ({_SMOKE_TEST_MIN_TOKENS} tokens) …")
    state = compressed.make_state()
    enc = tokenizer("What is the Hylian Shield?", return_tensors="pt")
    input_ids = enc["input_ids"].numpy().astype(np.int32)
    attention_mask = enc["attention_mask"].numpy().astype(np.int32)

    generated_tokens = []
    for _ in range(_SMOKE_TEST_MIN_TOKENS):
        output = compressed.predict(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            state=state,
        )
        next_token = int(np.argmax(output["logits"][0, -1, :]))
        generated_tokens.append(next_token)
        input_ids = np.array([[next_token]], dtype=np.int32)
        attention_mask = np.ones((1, attention_mask.shape[1] + 1), dtype=np.int32)

    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Smoke test output: {decoded!r}")
    _assert_smoke_test(generated_tokens)
    print("Smoke test passed.")

    # --- Save ---
    save_path = out_path / derived
    print(f"Saving to {save_path} …")
    compressed.save(str(save_path))

    # --- Size check ---
    _assert_size(save_path, variant)
    print(f"Done: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Llama 3.2 to Core ML")
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
