# model/tests/test_convert_llm.py
# Tests for convert_llm.py helper functions and the full convert() path.
# All heavy dependencies (torch, coremltools, transformers) are mocked —
# no live model downloads or HuggingFace requests in tests.

import sys
import os
import re
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pytest

from model.convert_llm import (
    _derive_filename,
    _parse_modelconfig_filename,
    _assert_filename_sync,
    _assert_smoke_test,
    _assert_size,
    _SMOKE_TEST_MIN_TOKENS,
    _MODELCONFIG_PATH,
)


# ---------------------------------------------------------------------------
# _derive_filename
# ---------------------------------------------------------------------------

def test_derive_filename_1b():
    assert _derive_filename("1B") == "QwenModel-1B.mlpackage"


def test_derive_filename_3b():
    assert _derive_filename("3B") == "QwenModel-3B.mlpackage"


def test_derive_filename_invalid():
    with pytest.raises(SystemExit) as exc:
        _derive_filename("5B")
    assert exc.value.code == 1


def test_derive_filename_empty_string():
    with pytest.raises(SystemExit) as exc:
        _derive_filename("")
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# _parse_modelconfig_filename
# ---------------------------------------------------------------------------

_FAKE_MODELCONFIG = """\
static var modelFilename: String {
    switch activeVariant {
    case .qwen1B: return "QwenModel-1B.mlpackage"
    case .qwen3B: return "QwenModel-3B.mlpackage"
    }
}
"""


def test_parse_modelconfig_filename_1b(tmp_path):
    swift_file = tmp_path / "ModelConfig.swift"
    swift_file.write_text(_FAKE_MODELCONFIG, encoding="utf-8")
    assert _parse_modelconfig_filename(swift_file, "1B") == "QwenModel-1B.mlpackage"


def test_parse_modelconfig_filename_3b(tmp_path):
    swift_file = tmp_path / "ModelConfig.swift"
    swift_file.write_text(_FAKE_MODELCONFIG, encoding="utf-8")
    assert _parse_modelconfig_filename(swift_file, "3B") == "QwenModel-3B.mlpackage"


def test_parse_modelconfig_filename_missing_file(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _parse_modelconfig_filename(tmp_path / "nonexistent.swift", "1B")
    assert exc.value.code == 1


def test_parse_modelconfig_filename_pattern_absent(tmp_path):
    swift_file = tmp_path / "ModelConfig.swift"
    swift_file.write_text("// no model filename here", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _parse_modelconfig_filename(swift_file, "1B")
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# _assert_filename_sync
# ---------------------------------------------------------------------------

def test_assert_filename_sync_passes():
    _assert_filename_sync("QwenModel-1B.mlpackage", "QwenModel-1B.mlpackage")


def test_assert_filename_sync_passes_cross_extension():
    # build-ipa.yml compiles .mlpackage → .mlmodelc; ModelConfig.swift uses .mlmodelc
    # while this script outputs .mlpackage.  Stem comparison must accept this pair.
    _assert_filename_sync("QwenModel-1B.mlpackage", "QwenModel-1B.mlmodelc")


def test_assert_filename_sync_exits_on_mismatch():
    with pytest.raises(SystemExit) as exc:
        _assert_filename_sync("QwenModel-1B.mlpackage", "QwenModel-WRONG.mlpackage")
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# _assert_smoke_test
# ---------------------------------------------------------------------------

def test_assert_smoke_test_passes():
    tokens = list(range(_SMOKE_TEST_MIN_TOKENS))
    _assert_smoke_test(tokens)  # should not raise or exit


def test_assert_smoke_test_passes_exactly_min():
    _assert_smoke_test(list(range(_SMOKE_TEST_MIN_TOKENS)))


def test_assert_smoke_test_exits_too_short():
    with pytest.raises(SystemExit) as exc:
        _assert_smoke_test(list(range(_SMOKE_TEST_MIN_TOKENS - 1)))
    assert exc.value.code == 1


def test_assert_smoke_test_exits_empty():
    with pytest.raises(SystemExit) as exc:
        _assert_smoke_test([])
    assert exc.value.code == 1


def test_assert_smoke_test_warns_degenerate(capsys):
    # All tokens identical — degenerate output prints a WARNING but does not exit.
    # CI macOS runners lack the Neural Engine so palettized models always produce
    # identical tokens; making this fatal would block every conversion run.
    tokens = [42] * _SMOKE_TEST_MIN_TOKENS
    _assert_smoke_test(tokens)  # must not raise
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "Degenerate" in captured.err


def test_assert_smoke_test_warns_mostly_degenerate(capsys):
    # 80 % identical tokens still triggers the warning (but not a fatal exit).
    tokens = [42] * 16 + [1, 2, 3, 4]   # 80 % token 42
    _assert_smoke_test(tokens)  # must not raise
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


def test_assert_smoke_test_passes_varied():
    # 75 % identical is below the 80 % threshold — should pass silently.
    tokens = [42] * 15 + [1, 2, 3, 4, 5]   # 75 % token 42
    _assert_smoke_test(tokens)  # must not raise


# ---------------------------------------------------------------------------
# _assert_size
# ---------------------------------------------------------------------------

def test_assert_size_passes_small_file(tmp_path):
    pkg = tmp_path / "QwenModel-1B.mlpackage"
    pkg.mkdir()
    (pkg / "weights.bin").write_bytes(b"x" * 1024)  # 1 KB
    _assert_size(pkg, "1B")  # well under 700 MB


def test_assert_size_exits_1b_over_limit(tmp_path):
    pkg = tmp_path / "QwenModel-1B.mlpackage"
    pkg.mkdir()
    limit_1b = 800 * 1024 * 1024
    # Patch stat to report oversized file without writing gigabytes
    fake_file = MagicMock()
    fake_stat = MagicMock()
    fake_stat.st_size = limit_1b + 1
    fake_file.stat.return_value = fake_stat
    fake_file.is_file.return_value = True
    with patch.object(Path, "rglob", return_value=[fake_file]):
        with pytest.raises(SystemExit) as exc:
            _assert_size(pkg, "1B")
    assert exc.value.code == 1


def test_assert_size_exits_3b_over_limit(tmp_path):
    pkg = tmp_path / "QwenModel-3B.mlpackage"
    pkg.mkdir()
    limit_3b = 2 * 1024 * 1024 * 1024
    fake_file = MagicMock()
    fake_stat = MagicMock()
    fake_stat.st_size = limit_3b + 1
    fake_file.stat.return_value = fake_stat
    fake_file.is_file.return_value = True
    with patch.object(Path, "rglob", return_value=[fake_file]):
        with pytest.raises(SystemExit) as exc:
            _assert_size(pkg, "3B")
    assert exc.value.code == 1


def test_assert_size_passes_3b_under_limit(tmp_path):
    pkg = tmp_path / "QwenModel-3B.mlpackage"
    pkg.mkdir()
    (pkg / "weights.bin").write_bytes(b"x" * 1024)
    _assert_size(pkg, "3B")


# ---------------------------------------------------------------------------
# convert() — full path with all heavy deps mocked
# ---------------------------------------------------------------------------

_FAKE_SWIFT = _FAKE_MODELCONFIG  # reuse the same fake ModelConfig content


def _make_torch_mock():
    torch_mock = MagicMock()

    # Fake model config
    cfg = MagicMock()
    cfg.num_hidden_layers = 16
    cfg.num_key_value_heads = 8
    cfg.num_attention_heads = 32
    cfg.hidden_size = 2048
    cfg.max_position_embeddings = 4096

    model_mock = MagicMock()
    model_mock.config = cfg
    model_mock.eval.return_value = model_mock

    # Fake tokenizer
    tokenizer_mock = MagicMock()
    enc = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    enc["input_ids"].numpy.return_value = np.zeros((1, 5), dtype=np.int32)
    enc["attention_mask"].numpy.return_value = np.ones((1, 5), dtype=np.int32)
    tokenizer_mock.return_value = enc
    tokenizer_mock.decode.return_value = "A shield found in Hyrule Castle."
    tokenizer_mock.apply_chat_template.return_value = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nWhat is the Hylian Shield?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    torch_mock.float32 = "float32"
    torch_mock.float16 = "float16"
    torch_mock.int64 = "int64"
    torch_mock.zeros.return_value = MagicMock()
    torch_mock.no_grad.return_value.__enter__ = lambda s: None
    torch_mock.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    # torch.export.export returns an exportedprogram mock
    exported_mock = MagicMock()
    torch_mock.export.export.return_value = exported_mock
    torch_mock.export.Dim.return_value = MagicMock()

    return torch_mock, model_mock, tokenizer_mock


def _make_ct_mock():
    ct_mock = MagicMock()
    ct_mock.target.iOS18 = "iOS18"
    ct_mock.ComputeUnit.CPU_AND_NE = "CPU_AND_NE"
    ct_mock.TensorType = MagicMock
    ct_mock.StateType = MagicMock
    ct_mock.RangeDim = MagicMock

    mlmodel_mock = MagicMock()
    ct_mock.convert.return_value = mlmodel_mock
    return ct_mock, mlmodel_mock


def _make_optimize_mock():
    # Build a predict side_effect that cycles through _SMOKE_TEST_MIN_TOKENS
    # distinct token IDs so the new degenerate-output check doesn't fire.
    # Also returns "present_kv" so the KV-cache smoke-test loop can proceed.
    vocab_size = 32000
    # Fake model config dimensions (matches _make_torch_mock cfg: 16 layers, 8 kv_heads, 64 head_dim)
    _NUM_LAYERS, _NUM_KV_HEADS, _HEAD_DIM = 16, 8, 64
    call_counter = {"n": 0}

    def _cycling_predict(inputs):
        token_id = call_counter["n"] % _SMOKE_TEST_MIN_TOKENS
        call_counter["n"] += 1
        logits = np.zeros((1, 1, vocab_size), dtype=np.float32)
        logits[0, 0, token_id] = 1.0
        # Fixed-KV design: return new_kv [L, 2, H, 1, D] for this single token only.
        # The caller (convert_llm.py smoke test) writes it into its own circular buffer.
        new_kv = np.zeros((_NUM_LAYERS, 2, _NUM_KV_HEADS, 1, _HEAD_DIM), dtype=np.float16)
        return {"logits": logits, "new_kv": new_kv}

    # compressed_mock is returned by ct.convert() (torch-level palettization
    # means ct.convert() produces the final compressed model directly)
    compressed_mock = MagicMock()
    compressed_mock.predict.side_effect = _cycling_predict
    compressed_mock.save = MagicMock()

    # torch_optimize_mock represents coremltools.optimize.torch.palettization
    torch_optimize_mock = MagicMock()
    palettizer_mock = MagicMock()
    palettizer_mock.compress.return_value = MagicMock()
    torch_optimize_mock.PostTrainingPalettizer.return_value = palettizer_mock
    torch_optimize_mock.PostTrainingPalettizerConfig = MagicMock()
    torch_optimize_mock.PostTrainingPalettizerConfig.from_dict.return_value = MagicMock()

    return torch_optimize_mock, compressed_mock


@pytest.fixture
def _patch_heavy(tmp_path, monkeypatch):
    torch_mock, model_mock, tokenizer_mock = _make_torch_mock()
    ct_mock, mlmodel_mock = _make_ct_mock()
    optimize_mock, compressed_mock = _make_optimize_mock()

    # Write a fake ModelConfig.swift so _parse_modelconfig_filename works
    swift_dir = tmp_path / "ios" / "ZeldaGuide" / "Services"
    swift_dir.mkdir(parents=True)
    (swift_dir / "ModelConfig.swift").write_text(_FAKE_SWIFT, encoding="utf-8")

    # Create a fake .mlpackage directory after save is called
    def fake_save(path):
        pkg = Path(path)
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "weights.bin").write_bytes(b"x" * 1024)

    compressed_mock.save.side_effect = fake_save

    # save_pretrained must create tokenizer.json in the given directory so the
    # shutil.copy in convert_llm.py succeeds.
    def fake_save_pretrained(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text(
            '{"model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}',
            encoding="utf-8",
        )
    tokenizer_mock.save_pretrained.side_effect = fake_save_pretrained

    transformers_mock = MagicMock()
    transformers_mock.AutoTokenizer.from_pretrained.return_value = tokenizer_mock
    transformers_mock.AutoModelForCausalLM.from_pretrained.return_value = model_mock

    # ct.convert returns the compressed_mock directly (torch-level palettization)
    ct_mock.convert.return_value = compressed_mock

    with patch.dict(sys.modules, {
        "torch": torch_mock,
        "torch.export": torch_mock.export,
        "coremltools": ct_mock,
        "coremltools.optimize": MagicMock(),
        "coremltools.optimize.torch": MagicMock(),
        "coremltools.optimize.torch.palettization": optimize_mock,
        "transformers": transformers_mock,
    }):
        monkeypatch.setenv("MODEL_VARIANT", "1B")
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        # Point _MODELCONFIG_PATH at our fake swift file
        with patch("model.convert_llm._MODELCONFIG_PATH", swift_dir / "ModelConfig.swift"):
            yield tmp_path, ct_mock, optimize_mock, compressed_mock


def test_convert_calls_ct_convert(_patch_heavy):
    tmp_path, ct_mock, _, _ = _patch_heavy
    from model.convert_llm import convert
    convert(str(tmp_path), "1B", "fake-token")
    ct_mock.convert.assert_called_once()


def test_convert_calls_palettizer(_patch_heavy):
    tmp_path, _, optimize_mock, _ = _patch_heavy
    from model.convert_llm import convert
    convert(str(tmp_path), "1B", "fake-token")
    optimize_mock.PostTrainingPalettizer.assert_called_once()


def test_convert_saves_correct_filename(_patch_heavy):
    tmp_path, _, _, compressed_mock = _patch_heavy
    from model.convert_llm import convert
    convert(str(tmp_path), "1B", "fake-token")
    compressed_mock.save.assert_called_once_with(str(tmp_path / "QwenModel-1B.mlpackage"))


def test_convert_saves_tokenizer_json(_patch_heavy):
    tmp_path, _, _, _ = _patch_heavy
    from model.convert_llm import convert
    convert(str(tmp_path), "1B", "fake-token")
    assert (tmp_path / "tokenizer.json").exists(), \
        "convert() must save tokenizer.json alongside the .mlpackage"


def test_convert_exits_on_filename_mismatch(_patch_heavy):
    tmp_path, _, _, _ = _patch_heavy
    from model.convert_llm import convert
    with patch("model.convert_llm._parse_modelconfig_filename",
               return_value="QwenModel-WRONG.mlpackage"):
        with pytest.raises(SystemExit) as exc:
            convert(str(tmp_path), "1B", "fake-token")
    assert exc.value.code == 1


def test_convert_exits_on_short_smoke_test(_patch_heavy):
    tmp_path, _, _, compressed_mock = _patch_heavy
    vocab_size = 32000
    # Fake model config dimensions (must match _make_torch_mock cfg: 16 layers, 8 kv_heads, 64 head_dim)
    _NUM_LAYERS, _NUM_KV_HEADS, _HEAD_DIM = 16, 8, 64

    def kv_predict(inputs):
        """Return logits + new_kv [L, 2, H, 1, D] for the fixed-KV interface."""
        logits = np.zeros((1, 1, vocab_size), dtype=np.float32)
        new_kv = np.zeros((_NUM_LAYERS, 2, _NUM_KV_HEADS, 1, _HEAD_DIM), dtype=np.float16)
        return {"logits": logits, "new_kv": new_kv}

    # Force _assert_smoke_test to raise SystemExit(1) immediately so we
    # confirm that convert() propagates the failure.
    compressed_mock.predict.side_effect = kv_predict
    from model.convert_llm import convert
    with patch("model.convert_llm._assert_smoke_test") as mock_assert:
        mock_assert.side_effect = SystemExit(1)
        with pytest.raises(SystemExit) as exc:
            convert(str(tmp_path), "1B", "fake-token")
        assert exc.value.code == 1
