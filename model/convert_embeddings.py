"""
convert_embeddings.py
Converts all-MiniLM-L6-v2 from sentence-transformers to a Core ML .mlpackage
so the iOS app can generate query embeddings on-device.

Output: model/MiniLMEmbedder.mlpackage
  - Input : tokenized text (via the tokenizer, passed as token IDs + attention mask)
  - Output: float32 array of shape (384,) — must match the pipeline DB dimension

Usage:
    python model/convert_embeddings.py [--output-dir <dir>]

The script exits with a non-zero code if the output shape is not exactly (384,),
preventing a silent dimension mismatch between the pipeline DB and the iOS embedder.
"""

import argparse
import sys
import os
import numpy as np

# Silence tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Average token embeddings weighted by the attention mask."""
    mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts  # (batch, 384)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, 1e-12, None)


def convert(output_dir: str) -> None:
    import torch
    import coremltools as ct
    from transformers import AutoTokenizer, AutoModel

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModel.from_pretrained(model_name)
    hf_model.eval()

    # Trace with a short representative input so TorchScript captures the right shapes.
    probe_text = "Hylian Shield"
    enc = tokenizer(probe_text, return_tensors="pt", padding="max_length",
                    max_length=128, truncation=True)
    input_ids = enc["input_ids"]          # (1, 128)
    attention_mask = enc["attention_mask"]  # (1, 128)

    class EmbeddingWrapper(torch.nn.Module):
        """Wraps HF model: returns mean-pooled, L2-normalised embedding (1, 384)."""
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            token_emb = out.last_hidden_state          # (1, seq, 384)
            mask_exp = attention_mask.unsqueeze(-1).float()
            summed = (token_emb * mask_exp).sum(dim=1)
            counts = mask_exp.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts                   # (1, 384)
            norm = pooled.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            return pooled / norm                       # (1, 384)

    wrapper = EmbeddingWrapper(hf_model)
    wrapper.eval()

    print("Tracing model …")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (input_ids, attention_mask))

    # Verify Python output shape before converting
    with torch.no_grad():
        py_out = traced(input_ids, attention_mask)
    assert py_out.shape == (1, 384), f"Unexpected traced output shape: {py_out.shape}"

    print("Converting to Core ML …")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids",      shape=input_ids.shape,      dtype=np.int32),
            ct.TensorType(name="attention_mask",  shape=attention_mask.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="embedding", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    # --- Shape assertion (exits non-zero if wrong) ---
    print("Verifying Core ML output shape …")
    spec = mlmodel.get_spec()
    output_shape = list(spec.description.output[0].type.multiArrayType.shape)
    # Core ML spec stores the full output shape; for a (1,384) trace it will be [1, 384]
    if output_shape[-1] != 384:
        print(
            f"ERROR: output shape {output_shape} does not end with 384. "
            "Pipeline DB and iOS embedder dimensions would mismatch.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Output shape OK: {output_shape}")

    # --- Save ---
    out_path = os.path.join(output_dir, "MiniLMEmbedder.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved: {out_path}")

    # --- Cross-check: run the saved model and compare with Python ---
    print("Cross-checking Core ML output against Python output …")
    loaded = ct.models.MLModel(out_path)
    cml_out = loaded.predict({
        "input_ids":     input_ids.numpy().astype(np.int32),
        "attention_mask": attention_mask.numpy().astype(np.int32),
    })
    cml_emb = list(cml_out.values())[0].flatten()   # (384,)
    py_emb = py_out.detach().numpy().flatten()       # (384,)
    max_diff = float(np.abs(cml_emb - py_emb).max())
    print(f"Max abs diff Core ML vs Python: {max_diff:.6e}")
    if max_diff > 1e-3:
        print(
            f"ERROR: Core ML and Python embeddings differ by {max_diff:.6e} (tolerance 1e-3).",
            file=sys.stderr,
        )
        sys.exit(1)
    print("Cross-check passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert all-MiniLM-L6-v2 to Core ML")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__)),
        help="Directory to write MiniLMEmbedder.mlpackage (default: model/)",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    convert(args.output_dir)


if __name__ == "__main__":
    main()
