"""
smoke_test_llm.py
Loads an existing LlamaModel .mlpackage and runs the same 20-token smoke test
used by convert_llm.py. Use this to re-validate a converted model without
re-running the full 1h+ conversion.

Usage:
    python model/smoke_test_llm.py --model-path model/LlamaModel-1B.mlpackage

Environment variables:
    MODEL_VARIANT   "1B" (default) or "3B" — selects the tokenizer to load
    HF_TOKEN        HuggingFace token (optional for Qwen2.5 ungated models)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

from model.convert_llm import (
    _HF_IDS,
    _SMOKE_TEST_MIN_TOKENS,
    _assert_smoke_test,
    _assert_size,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test an existing LlamaModel .mlpackage")
    parser.add_argument("--model-path", required=True, help="Path to the .mlpackage directory")
    args = parser.parse_args()

    variant = os.environ.get("MODEL_VARIANT", "1B")
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
    model = ct.models.MLModel(str(model_path))

    # Size check
    _assert_size(model_path, variant)

    print(f"Running smoke test ({_SMOKE_TEST_MIN_TOKENS} tokens) …")
    enc = tokenizer("What is the Hylian Shield?", return_tensors="pt")
    input_ids = enc["input_ids"].numpy().astype(np.int32)
    attention_mask = enc["attention_mask"].numpy().astype(np.int32)

    generated_tokens = []
    for _ in range(_SMOKE_TEST_MIN_TOKENS):
        output = model.predict({"input_ids": input_ids, "attention_mask": attention_mask})
        next_token = int(np.argmax(output["logits"][0, -1, :]))
        generated_tokens.append(next_token)
        input_ids = np.append(input_ids, [[next_token]], axis=1).astype(np.int32)
        attention_mask = np.ones((1, input_ids.shape[1]), dtype=np.int32)

    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Smoke test output: {decoded!r}")
    _assert_smoke_test(generated_tokens)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
