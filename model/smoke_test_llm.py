"""
smoke_test_llm.py
Loads an existing LlamaModel .mlpackage and runs a structural smoke test:
verifies the model loads, runs predict() without crashing, and produces the
expected number of output tokens.

NOTE: Output quality (non-degenerate text) is NOT checked here. The 4-bit
palettized model targets the iOS Neural Engine and produces garbage logits on
macOS CI runners where the NE is unavailable. Quality validation must happen
on device. The degenerate-output check lives in convert_llm.py, which runs
immediately after conversion in the same process where the model is still
in memory and behaves correctly.

Usage:
    python model/smoke_test_llm.py --model-path model/LlamaModel-1B.mlpackage

Environment variables:
    MODEL_VARIANT   "1B" (default) or "3B" — selects the tokenizer to load
    HF_TOKEN        HuggingFace token (optional for Qwen2.5 ungated models)
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

from model.convert_llm import (
    _HF_IDS,
    _SMOKE_TEST_MIN_TOKENS,
    _assert_size,
)


def _assert_run(tokens: list, min_tokens: int = _SMOKE_TEST_MIN_TOKENS) -> None:
    """Exit non-zero if fewer than min_tokens were produced.
    Does NOT check for degenerate output — see module docstring."""
    if len(tokens) < min_tokens:
        print(
            f"ERROR: Smoke test produced only {len(tokens)} tokens "
            f"(expected >= {min_tokens}). Model may have crashed or returned empty output.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Structural smoke test for an existing LlamaModel .mlpackage")
    parser.add_argument("--model-path", required=True, help="Path to the .mlpackage directory")
    args = parser.parse_args()

    variant  = os.environ.get("MODEL_VARIANT", "1B")
    hf_token = os.environ.get("HF_TOKEN", "")

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: {model_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    hf_id = _HF_IDS.get(variant)
    if hf_id is None:
        print(f"ERROR: Unknown MODEL_VARIANT '{variant}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer from {hf_id} …")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=hf_token or None)

    print(f"Loading Core ML model from {model_path} …")
    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)

    # Size check
    _assert_size(model_path, variant)

    print(f"Running smoke test ({_SMOKE_TEST_MIN_TOKENS} tokens) …")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is the Hylian Shield?"},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(input_text, return_tensors="pt")
    input_ids    = enc["input_ids"].numpy().astype(np.int32)
    attn_mask    = enc["attention_mask"].numpy().astype(np.int32)

    generated_tokens = []
    for step in range(_SMOKE_TEST_MIN_TOKENS):
        output     = model.predict({"input_ids": input_ids, "attention_mask": attn_mask})
        logits     = output["logits"][0, -1, :]
        next_token = int(np.argmax(logits))
        generated_tokens.append(next_token)
        input_ids = np.append(input_ids, [[next_token]], axis=1).astype(np.int32)
        attn_mask = np.ones((1, input_ids.shape[1]), dtype=np.int32)

        if step == 0:
            # Log logit diagnostics on the first step to aid future debugging.
            print(f"  Logit stats — min: {logits.min():.3f}, max: {logits.max():.3f}, "
                  f"mean: {logits.mean():.3f}, argmax: {next_token}")

    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    unique  = len(set(generated_tokens))
    most_common_token, most_common_count = Counter(generated_tokens).most_common(1)[0]
    print(f"Smoke test output: {decoded!r}")
    print(f"  Unique tokens: {unique}/{len(generated_tokens)}, "
          f"most common: token {most_common_token} × {most_common_count}")
    if unique == 1:
        print("  WARNING: all tokens identical — model likely running without Neural Engine. "
              "Output quality must be validated on device.")

    _assert_run(generated_tokens)
    print("Smoke test passed (structural check only — validate output quality on device).")


if __name__ == "__main__":
    main()
