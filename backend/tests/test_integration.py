"""
Integration tests — exercise the full agent loop and API routes.

These tests use stub mode (ALLOW_STUB_FALLBACK=true, ALLOW_STUB_SANDBOX=true)
so they run without real vLLM, Fireworks API, or Docker sandbox.
"""
import asyncio
import os
import pytest
from pathlib import Path

# Set stub mode for all integration tests
os.environ.setdefault("ALLOW_STUB_FALLBACK", "true")
os.environ.setdefault("ALLOW_STUB_SANDBOX", "true")
os.environ.setdefault("FIREWORKS_API_KEY", "test_key")

from fastapi.testclient import TestClient
from app.main import app
from app.agents.graph import run_agent
from app.schemas import JobStatus


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_cuda_source():
    """Load a real CUDA sample for testing."""
    samples_dir = Path(__file__).parent.parent.parent / "samples" / "cuda"
    return (samples_dir / "01_vector_add.cu").read_text()


# ============================================================
# API Route Tests
# ============================================================

class TestAPIRoutes:
    """Test FastAPI routes via TestClient."""

    def test_health_endpoint(self, client):
        """GET /health should return 200 with status info."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "amd_gpu_available" in data

    def test_root_endpoint(self, client):
        """GET / should return project info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "Crossfire"

    def test_list_samples(self, client):
        """GET /api/samples should return list of CUDA samples."""
        response = client.get("/api/samples")
        assert response.status_code == 200
        data = response.json()
        assert "samples" in data
        assert data["count"] == 20
        assert len(data["samples"]) == 20

    def test_get_sample(self, client):
        """GET /api/samples/{filename} should return sample source."""
        response = client.get("/api/samples/01_vector_add.cu")
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "01_vector_add.cu"
        assert "__global__" in data["source"]

    def test_get_sample_path_traversal_blocked(self, client):
        """GET /api/samples/../etc/passwd should return 400 (path traversal)."""
        response = client.get("/api/samples/..%2Fetc%2Fpasswd")
        assert response.status_code in (400, 404)

    def test_get_nonexistent_sample(self, client):
        """GET /api/samples/nonexistent.cu should return 404."""
        response = client.get("/api/samples/nonexistent.cu")
        assert response.status_code == 404

    def test_list_jobs_empty(self, client):
        """GET /api/jobs should return empty list initially."""
        response = client.get("/api/jobs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_translate_endpoint_validation(self, client):
        """POST /api/translate with empty source should return 422."""
        response = client.post("/api/translate", json={"cuda_source": ""})
        assert response.status_code == 422  # Pydantic validation error


# ============================================================
# Agent Loop Integration Tests
# ============================================================

class TestAgentLoop:
    """Test the full agent loop end-to-end (in stub mode)."""

    @pytest.mark.asyncio
    async def test_agent_returns_result(self, sample_cuda_source):
        """Agent should return a TranslationResult for any input."""
        result = await asyncio.wait_for(
            run_agent(
                job_id="integration-test-1",
                cuda_source=sample_cuda_source,
                filename="01_vector_add.cu",
                max_iterations=1,  # keep fast
            ),
            timeout=120,
        )
        assert result is not None
        assert result.job_id == "integration-test-1"
        assert result.status in (JobStatus.DONE, JobStatus.FAILED)
        assert result.cuda_source == sample_cuda_source

    @pytest.mark.asyncio
    async def test_agent_records_analysis(self, sample_cuda_source):
        """Agent should populate the analysis field."""
        result = await asyncio.wait_for(
            run_agent(
                job_id="integration-test-2",
                cuda_source=sample_cuda_source,
                filename="01_vector_add.cu",
                max_iterations=1,
            ),
            timeout=120,
        )
        assert result.analysis is not None
        assert result.analysis.kernel_count >= 1
        assert 0.0 <= result.analysis.difficulty_score <= 1.0

    @pytest.mark.asyncio
    async def test_agent_with_empty_source(self):
        """Agent should handle empty source gracefully."""
        result = await asyncio.wait_for(
            run_agent(
                job_id="integration-test-3",
                cuda_source="// empty file\n",
                filename="empty.cu",
                max_iterations=1,
            ),
            timeout=120,
        )
        # Should not crash; may fail (no kernel to translate)
        assert result is not None
        assert result.status in (JobStatus.DONE, JobStatus.FAILED)

    @pytest.mark.asyncio
    async def test_agent_iteration_count_respected(self, sample_cuda_source):
        """Agent should not exceed max_iterations."""
        max_iter = 2
        result = await asyncio.wait_for(
            run_agent(
                job_id="integration-test-4",
                cuda_source=sample_cuda_source,
                filename="01_vector_add.cu",
                max_iterations=max_iter,
            ),
            timeout=180,
        )
        assert len(result.iterations) <= max_iter


# ============================================================
# Baseline Loading Tests
# ============================================================

class TestBaselineLoading:
    """Test that baselines load correctly for all 20 samples."""

    def test_all_baselines_load(self):
        """All 20 samples should have loadable baselines."""
        from app.agents.nodes import _load_baseline_outputs, _load_baseline_inputs

        samples_dir = Path(__file__).parent.parent.parent / "samples" / "cuda"
        for cu_file in sorted(samples_dir.glob("*.cu")):
            outputs = _load_baseline_outputs(cu_file.name)
            inputs = _load_baseline_inputs(cu_file.name)
            assert outputs, f"No baseline outputs for {cu_file.name}"
            assert inputs, f"No baseline inputs for {cu_file.name}"

    def test_baseline_outputs_have_status(self):
        """Each baseline output should have a 'status' field."""
        from app.agents.nodes import _load_baseline_outputs

        samples_dir = Path(__file__).parent.parent.parent / "samples" / "cuda"
        for cu_file in sorted(samples_dir.glob("*.cu")):
            outputs = _load_baseline_outputs(cu_file.name)
            assert "status" in outputs, f"{cu_file.name} baseline missing 'status' field"


# ============================================================
# Middleware Tests
# ============================================================

class TestMiddleware:
    """Test rate limiting and basic auth middleware."""

    def test_health_bypasses_auth(self, client, monkeypatch):
        """Health endpoint should work without auth."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_rate_limiting_kicks_in(self, client):
        """After burst limit, should return 429."""
        # Make many requests rapidly (burst is 20)
        statuses = []
        for _ in range(25):
            response = client.get("/api/samples")
            statuses.append(response.status_code)
        # At least one should be rate limited (429)
        assert 429 in statuses, f"Expected rate limiting, got statuses: {set(statuses)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
