"""
Tests for the fixed sandbox behavior (Fix #4 and #11).

Verifies that:
- Sandbox does NOT fabricate success=True when unavailable
- Output parsing correctly extracts JSON from between markers
- Stub mode (when allowed) returns explicit failure, not fake success
"""
import os
import pytest

from app.sandbox.compiler import SandboxClient
from app.schemas import CompileResult, RunResult


class TestSandboxStubBehavior:
    """Verify the stub-fabricates-success pattern is FIXED (Fix #4)."""

    def test_compile_returns_failure_when_sandbox_unavailable(self, monkeypatch):
        """When sandbox is unavailable and ALLOW_STUB_SANDBOX is false, compile must fail."""
        monkeypatch.delenv("ALLOW_STUB_SANDBOX", raising=False)
        client = SandboxClient(container="nonexistent-container-xyz")

        import asyncio
        result = asyncio.run(client.compile("int main() { return 0; }"))

        assert result.success is False, "Compile must NOT fabricate success when sandbox unavailable"
        assert result.errors is not None
        assert "not running" in result.errors or "STUB" in result.errors

    def test_run_returns_failure_when_sandbox_unavailable(self, monkeypatch):
        """When sandbox is unavailable and ALLOW_STUB_SANDBOX is false, run must fail."""
        monkeypatch.delenv("ALLOW_STUB_SANDBOX", raising=False)
        client = SandboxClient(container="nonexistent-container-xyz")

        import asyncio
        result = asyncio.run(client.run("/fake/path", {"N": 100}))

        assert result.success is False, "Run must NOT fabricate success when sandbox unavailable"
        assert result.stderr is not None
        assert "not running" in result.stderr or "STUB" in result.stderr

    def test_stub_mode_returns_failure_with_clear_message(self, monkeypatch):
        """Even with ALLOW_STUB_SANDBOX=true, stub mode returns failure (not fake success)."""
        monkeypatch.setenv("ALLOW_STUB_SANDBOX", "true")
        client = SandboxClient(container="nonexistent-container-xyz")

        import asyncio
        result = asyncio.run(client.compile("int main() { return 0; }"))

        assert result.success is False, "Stub mode must return failure, not fabricated success"
        assert "STUB" in result.errors or "not running" in result.errors


class TestOutputParsing:
    """Verify output parsing correctly extracts JSON from markers (Fix #11)."""

    def test_parse_output_with_markers(self):
        """Should extract JSON between ===OUTPUT_BEGIN=== and ===OUTPUT_END=== markers."""
        client = SandboxClient()
        stdout = """Max error: 0.000000e+00
Result: PASS
===OUTPUT_BEGIN===
{"max_error": 0.0, "status": "pass"}
===OUTPUT_END===
"""
        result = client._parse_output(stdout)
        assert result == {"max_error": 0.0, "status": "pass"}

    def test_parse_output_with_complex_json(self):
        """Should extract complex JSON (lists, nested objects) from markers."""
        client = SandboxClient()
        stdout = """Some debug output
===OUTPUT_BEGIN===
{"block_sums": [256.0, 256.0, 256.0, 256.0], "max_error": 0.0, "status": "pass"}
===OUTPUT_END===
trailing text
"""
        result = client._parse_output(stdout)
        assert result["block_sums"] == [256.0, 256.0, 256.0, 256.0]
        assert result["status"] == "pass"

    def test_parse_output_without_markers_fallback_to_json(self):
        """If no markers, try to parse whole stdout as JSON."""
        client = SandboxClient()
        stdout = '{"max_error": 0.0, "status": "pass"}'
        result = client._parse_output(stdout)
        assert result == {"max_error": 0.0, "status": "pass"}

    def test_parse_output_invalid_fallback_to_raw(self):
        """If no markers and not valid JSON, return raw_stdout dict."""
        client = SandboxClient()
        stdout = "Just some plain text output, no JSON here"
        result = client._parse_output(stdout)
        assert "raw_stdout" in result
        assert "plain text" in result["raw_stdout"]

    def test_parse_output_handles_malformed_json_between_markers(self):
        """If JSON between markers is malformed, fall back to raw_stdout."""
        client = SandboxClient()
        stdout = """===OUTPUT_BEGIN===
{not valid json}
===OUTPUT_END==="""
        result = client._parse_output(stdout)
        # Should fall back (either to whole-stdout JSON parse, which fails, then to raw)
        assert "raw_stdout" in result or "not valid json" in str(result)


class TestBaselineFiles:
    """Verify baseline files were generated for all 20 samples (Fix #10)."""

    def test_all_baseline_files_exist(self):
        """All 20 samples should have _outputs.json and _inputs.json files."""
        from pathlib import Path
        baselines_dir = Path(__file__).parent.parent.parent / "samples" / "baselines"

        # Get list of sample stems
        samples_dir = Path(__file__).parent.parent.parent / "samples" / "cuda"
        sample_stems = [f.stem for f in samples_dir.glob("*.cu")]

        assert len(sample_stems) == 20, f"Expected 20 samples, found {len(sample_stems)}"

        missing_outputs = []
        missing_inputs = []
        for stem in sample_stems:
            outputs_file = baselines_dir / f"{stem}_outputs.json"
            inputs_file = baselines_dir / f"{stem}_inputs.json"
            if not outputs_file.exists():
                missing_outputs.append(stem)
            if not inputs_file.exists():
                missing_inputs.append(stem)

        assert not missing_outputs, f"Missing baseline outputs: {missing_outputs}"
        assert not missing_inputs, f"Missing baseline inputs: {missing_inputs}"

    def test_baseline_outputs_are_valid_json(self):
        """All baseline output files should be valid JSON."""
        import json
        from pathlib import Path
        baselines_dir = Path(__file__).parent.parent.parent / "samples" / "baselines"

        for outputs_file in baselines_dir.glob("*_outputs.json"):
            with open(outputs_file) as f:
                try:
                    data = json.load(f)
                    assert isinstance(data, dict), f"{outputs_file.name} should be a JSON object"
                except json.JSONDecodeError as e:
                    pytest.fail(f"{outputs_file.name} is not valid JSON: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
