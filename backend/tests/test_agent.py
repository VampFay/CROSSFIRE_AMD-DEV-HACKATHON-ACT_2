"""
Tests for the agent routing logic.
"""
import pytest

from app.agents.routing import ModelRouter
from app.schemas import AnalysisResult, ModelChoice, TranslationPattern


def make_analysis(
    patterns=None,
    difficulty=0.2,
    has_shared=False,
    has_warp=False,
    kernel_count=0,
    library_calls=None,
):
    """Helper to create an AnalysisResult."""
    return AnalysisResult(
        patterns=patterns or [],
        difficulty_score=difficulty,
        kernel_count=kernel_count,
        has_shared_memory=has_shared,
        has_warp_primitives=has_warp,
        library_calls=library_calls or [],
        file_dependencies=[],
        notes="test",
    )


class TestModelRouter:

    def test_easy_routes_to_local(self):
        """Low difficulty → LOCAL."""
        analysis = make_analysis(difficulty=0.1)
        router = ModelRouter()
        choice = router.route(analysis)
        assert choice == ModelChoice.LOCAL

    def test_hard_routes_to_remote(self):
        """High difficulty → REMOTE."""
        analysis = make_analysis(difficulty=0.8)
        router = ModelRouter()
        choice = router.route(analysis)
        assert choice == ModelChoice.REMOTE

    def test_force_remote(self):
        """force_remote should always pick REMOTE."""
        analysis = make_analysis(difficulty=0.1)
        router = ModelRouter()
        choice = router.route(analysis, force_remote=True)
        assert choice == ModelChoice.REMOTE

    def test_cudnn_routes_to_remote(self):
        """cuDNN pattern always goes REMOTE."""
        analysis = make_analysis(
            patterns=[TranslationPattern.CUDNN],
            difficulty=0.1,  # low difficulty but cuDNN forces remote
        )
        router = ModelRouter()
        choice = router.route(analysis)
        assert choice == ModelChoice.REMOTE

    def test_triton_routes_to_remote(self):
        """Triton pattern always goes REMOTE."""
        analysis = make_analysis(
            patterns=[TranslationPattern.TRITON],
            difficulty=0.1,
        )
        router = ModelRouter()
        choice = router.route(analysis)
        assert choice == ModelChoice.REMOTE

    def test_warp_shuffle_routes_to_remote(self):
        """Warp shuffle always goes REMOTE."""
        analysis = make_analysis(
            patterns=[TranslationPattern.WARP_SHUFFLE],
            difficulty=0.1,
        )
        router = ModelRouter()
        choice = router.route(analysis)
        assert choice == ModelChoice.REMOTE

    def test_threshold_boundary(self):
        """At exactly threshold, goes REMOTE."""
        from app.config import settings
        analysis = make_analysis(difficulty=settings.agent_routing_threshold)
        router = ModelRouter()
        choice = router.route(analysis)
        assert choice == ModelChoice.REMOTE

    def test_just_below_threshold(self):
        """Just below threshold, goes LOCAL."""
        from app.config import settings
        analysis = make_analysis(difficulty=settings.agent_routing_threshold - 0.01)
        router = ModelRouter()
        choice = router.route(analysis)
        assert choice == ModelChoice.LOCAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
