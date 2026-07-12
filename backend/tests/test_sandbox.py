"""
Tests for the sandbox diff function.
"""
import pytest

from app.sandbox.compiler import SandboxClient


class TestSandboxDiff:

    def test_identical_outputs_pass(self):
        sandbox = SandboxClient()
        result = sandbox.diff(
            actual={"output": [1.0, 2.0, 3.0]},
            baseline={"output": [1.0, 2.0, 3.0]},
            threshold=1e-5,
        )
        assert result.success is True
        assert result.max_abs_error == 0.0

    def test_large_diff_fails(self):
        sandbox = SandboxClient()
        result = sandbox.diff(
            actual={"output": [1.0, 2.0, 3.0]},
            baseline={"output": [10.0, 20.0, 30.0]},
            threshold=1e-5,
        )
        assert result.success is False
        assert result.max_abs_error == 27.0  # 30 - 3

    def test_small_diff_passes(self):
        sandbox = SandboxClient()
        result = sandbox.diff(
            actual={"output": [1.0, 2.0, 3.0]},
            baseline={"output": [1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7]},
            threshold=1e-5,
        )
        assert result.success is True

    def test_missing_key_fails(self):
        sandbox = SandboxClient()
        result = sandbox.diff(
            actual={"output": [1.0]},
            baseline={"output": [1.0], "extra_key": [2.0]},
            threshold=1e-5,
        )
        assert result.success is False
        assert "extra_key" in result.mismatched_keys

    def test_empty_baseline_fails(self):
        sandbox = SandboxClient()
        result = sandbox.diff(
            actual={"output": [1.0]},
            baseline={},
            threshold=1e-5,
        )
        assert result.success is False

    def test_scalar_values(self):
        sandbox = SandboxClient()
        result = sandbox.diff(
            actual={"max_error": 1e-7},
            baseline={"max_error": 1e-7},
            threshold=1e-5,
        )
        assert result.success is True

    def test_shape_mismatch_fails(self):
        sandbox = SandboxClient()
        result = sandbox.diff(
            actual={"output": [1.0, 2.0]},
            baseline={"output": [1.0, 2.0, 3.0]},
            threshold=1e-5,
        )
        assert result.success is False
        assert "output" in result.mismatched_keys

    def test_mse_computed(self):
        sandbox = SandboxClient()
        result = sandbox.diff(
            actual={"output": [1.0, 2.0]},
            baseline={"output": [2.0, 3.0]},
            threshold=1e-5,
        )
        # diff = [1, 1], mse = (1^2 + 1^2) / 2 = 1.0
        assert result.mse == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
