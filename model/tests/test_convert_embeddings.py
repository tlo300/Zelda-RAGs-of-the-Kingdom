# model/tests/test_convert_embeddings.py
# Tests for convert_embeddings.py assertion helpers and the full convert() path.
# All heavy dependencies (torch, coremltools, transformers) are mocked —
# no live model downloads in tests.

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest

from model.convert_embeddings import (
    EMBEDDING_DIM,
    _assert_output_shape,
    _assert_embedding_match,
)


# ---------------------------------------------------------------------------
# _assert_output_shape
# ---------------------------------------------------------------------------


def test_assert_output_shape_passes_for_correct_dim():
    _assert_output_shape([1, EMBEDDING_DIM])  # should not raise or exit


def test_assert_output_shape_passes_for_flat_shape():
    _assert_output_shape([EMBEDDING_DIM])


def test_assert_output_shape_exits_for_wrong_dim():
    with pytest.raises(SystemExit) as exc:
        _assert_output_shape([1, 512])
    assert exc.value.code == 1


def test_assert_output_shape_exits_for_truncated_dim():
    with pytest.raises(SystemExit) as exc:
        _assert_output_shape([1, 256])
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# _assert_embedding_match
# ---------------------------------------------------------------------------


def test_assert_embedding_match_passes_for_identical():
    v = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    _assert_embedding_match(v, v.copy())  # should not raise or exit


def test_assert_embedding_match_passes_within_tolerance():
    v = np.ones(EMBEDDING_DIM, dtype=np.float32)
    w = v + 5e-4  # max diff 5e-4 < 1e-3
    _assert_embedding_match(v, w)


def test_assert_embedding_match_exits_when_diff_exceeds_tolerance():
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    w = v.copy()
    w[0] = 2e-3  # max diff 2e-3 > 1e-3
    with pytest.raises(SystemExit) as exc:
        _assert_embedding_match(v, w)
    assert exc.value.code == 1


def test_assert_embedding_match_exits_exactly_at_boundary():
    """Diff == tolerance should fail (strict >)."""
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    w = v.copy()
    w[0] = 1e-3 + 1e-7  # just over tolerance
    with pytest.raises(SystemExit) as exc:
        _assert_embedding_match(v, w)
    assert exc.value.code == 1


def test_assert_embedding_match_custom_tolerance():
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    w = v.copy()
    w[0] = 5e-2  # within tol=0.1
    _assert_embedding_match(v, w, tol=0.1)


# ---------------------------------------------------------------------------
# convert() — full path with all heavy deps mocked
# ---------------------------------------------------------------------------


class _FakeModule:
    """Minimal stand-in for torch.nn.Module so EmbeddingWrapper can inherit from it."""
    def __init__(self):
        pass

    def eval(self):
        return self


def _make_torch_mock(emb_dim: int = EMBEDDING_DIM):
    """Return a minimal torch mock that produces (1, emb_dim) tensors."""
    torch_mock = MagicMock()

    # tensor values used as the "Python side" embedding
    py_tensor = MagicMock()
    py_tensor.shape = (1, emb_dim)
    py_tensor.detach.return_value.numpy.return_value = np.full(
        (1, emb_dim), 0.5, dtype=np.float32
    )

    # tokenizer encoding mock
    enc_mock = {
        "input_ids": MagicMock(shape=(1, 128)),
        "attention_mask": MagicMock(shape=(1, 128)),
    }
    enc_mock["input_ids"].numpy.return_value = np.zeros((1, 128), dtype=np.int32)
    enc_mock["attention_mask"].numpy.return_value = np.ones((1, 128), dtype=np.int32)

    traced = MagicMock()
    traced.return_value = py_tensor

    torch_mock.jit.trace.return_value = traced
    torch_mock.no_grad.return_value.__enter__ = lambda s: None
    torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    torch_mock.nn.Module = _FakeModule  # EmbeddingWrapper inherits eval() from this

    return torch_mock, enc_mock, py_tensor


def _make_ct_mock(output_shape: list, cml_embedding: np.ndarray | None = None):
    """Return a coremltools mock with a configurable output spec shape."""
    ct_mock = MagicMock()

    # spec shape
    spec = MagicMock()
    spec.description.output[0].type.multiArrayType.shape = output_shape
    mlmodel = MagicMock()
    mlmodel.get_spec.return_value = spec
    ct_mock.convert.return_value = mlmodel

    # loaded model predict
    if cml_embedding is None:
        cml_embedding = np.full(EMBEDDING_DIM, 0.5, dtype=np.float32)
    loaded = MagicMock()
    loaded.predict.return_value = {"embedding": cml_embedding.reshape(1, -1)}
    ct_mock.models.MLModel.return_value = loaded

    ct_mock.TensorType = MagicMock
    ct_mock.target.iOS17 = "iOS17"
    ct_mock.ComputeUnit.CPU_AND_NE = "CPU_AND_NE"

    return ct_mock, mlmodel


def _make_transformers_mock(enc_mock: dict):
    transformers_mock = MagicMock()
    tokenizer = MagicMock()
    tokenizer.return_value = enc_mock
    transformers_mock.AutoTokenizer.from_pretrained.return_value = tokenizer

    hf_model = MagicMock()
    hf_model.eval.return_value = hf_model
    transformers_mock.AutoModel.from_pretrained.return_value = hf_model

    return transformers_mock


@pytest.fixture
def _patch_heavy(tmp_path):
    """Yield (tmp_path, ct_mock, mlmodel_mock) with correct shapes and matching embeddings."""
    torch_mock, enc_mock, py_tensor = _make_torch_mock()
    ct_mock, mlmodel = _make_ct_mock(output_shape=[1, EMBEDDING_DIM])
    transformers_mock = _make_transformers_mock(enc_mock)

    with patch.dict(sys.modules, {
        "torch": torch_mock,
        "torch.jit": torch_mock.jit,
        "torch.nn": torch_mock.nn,
        "coremltools": ct_mock,
        "coremltools.models": ct_mock.models,
        "transformers": transformers_mock,
    }):
        yield tmp_path, ct_mock, mlmodel


def test_convert_saves_mlpackage(_patch_heavy):
    tmp_path, ct_mock, mlmodel = _patch_heavy
    from model.convert_embeddings import convert
    convert(str(tmp_path))
    mlmodel.save.assert_called_once_with(str(tmp_path / "MiniLMEmbedder.mlpackage"))


def test_convert_calls_ct_convert(_patch_heavy):
    tmp_path, ct_mock, _ = _patch_heavy
    from model.convert_embeddings import convert
    convert(str(tmp_path))
    ct_mock.convert.assert_called_once()


def test_convert_exits_on_wrong_output_shape(tmp_path):
    torch_mock, enc_mock, _ = _make_torch_mock()
    ct_mock, _ = _make_ct_mock(output_shape=[1, 512])  # wrong dim
    transformers_mock = _make_transformers_mock(enc_mock)

    with patch.dict(sys.modules, {
        "torch": torch_mock,
        "torch.jit": torch_mock.jit,
        "torch.nn": torch_mock.nn,
        "coremltools": ct_mock,
        "coremltools.models": ct_mock.models,
        "transformers": transformers_mock,
    }):
        from model.convert_embeddings import convert
        with pytest.raises(SystemExit) as exc:
            convert(str(tmp_path))
        assert exc.value.code == 1


def test_convert_exits_on_embedding_mismatch(tmp_path):
    torch_mock, enc_mock, _ = _make_torch_mock()
    # py_emb is all 0.5; cml_emb differs by 0.1 at index 0 — exceeds 1e-3
    bad_cml = np.full(EMBEDDING_DIM, 0.5, dtype=np.float32)
    bad_cml[0] = 0.6
    ct_mock, _ = _make_ct_mock(output_shape=[1, EMBEDDING_DIM], cml_embedding=bad_cml)
    transformers_mock = _make_transformers_mock(enc_mock)

    with patch.dict(sys.modules, {
        "torch": torch_mock,
        "torch.jit": torch_mock.jit,
        "torch.nn": torch_mock.nn,
        "coremltools": ct_mock,
        "coremltools.models": ct_mock.models,
        "transformers": transformers_mock,
    }):
        from model.convert_embeddings import convert
        with pytest.raises(SystemExit) as exc:
            convert(str(tmp_path))
        assert exc.value.code == 1
