"""Tests for the SmartRouter module."""

from __future__ import annotations

import pytest

from src.models import Complexity, QualityTier, RoutingRequest
from src.router import SmartRouter


@pytest.fixture
def router() -> SmartRouter:
    """Create a SmartRouter instance with default config."""
    return SmartRouter()


class TestClassifyComplexity:
    """Tests for complexity classification."""

    def test_empty_text_is_simple(self, router: SmartRouter) -> None:
        assert router.classify_complexity("") == Complexity.SIMPLE

    def test_short_greeting_is_simple(self, router: SmartRouter) -> None:
        assert router.classify_complexity("hello") == Complexity.SIMPLE

    def test_short_question_is_simple(self, router: SmartRouter) -> None:
        assert router.classify_complexity("What is Python?") == Complexity.SIMPLE

    def test_moderate_question(self, router: SmartRouter) -> None:
        text = (
            "Can you explain in detail how to set up a REST API with authentication "
            "and database integration using FastAPI? I need to understand the step by step "
            "process for configuring OAuth2 with JWT tokens and connecting to a PostgreSQL "
            "database using SQLAlchemy async sessions."
        )
        result = router.classify_complexity(text)
        assert result in (Complexity.MODERATE, Complexity.COMPLEX)

    def test_complex_multi_part_request(self, router: SmartRouter) -> None:
        text = (
            "I need you to analyze the trade-offs between microservices and monolithic "
            "architectures. Please provide:\n"
            "1. A comprehensive comparison of scalability\n"
            "2. Cost analysis for both approaches\n"
            "3. Evaluate the deployment complexity\n"
            "4. Provide a step by step migration guide\n"
            "5. Compare the pros and cons of each approach"
        )
        result = router.classify_complexity(text)
        assert result == Complexity.COMPLEX

    def test_code_related_increases_complexity(self, router: SmartRouter) -> None:
        text = "Can you help me refactor this code?\n```python\ndef process(data):\n    pass\n```"
        result = router.classify_complexity(text)
        assert result in (Complexity.MODERATE, Complexity.COMPLEX)

    def test_simple_keywords_reduce_complexity(self, router: SmartRouter) -> None:
        assert router.classify_complexity("yes") == Complexity.SIMPLE
        assert router.classify_complexity("ok") == Complexity.SIMPLE
        assert router.classify_complexity("thanks") == Complexity.SIMPLE


class TestRouting:
    """Tests for request routing decisions."""

    def test_simple_routes_to_economy(self, router: SmartRouter) -> None:
        decision = router.route_text("hello")
        assert decision.quality_tier == QualityTier.ECONOMY
        assert decision.complexity == Complexity.SIMPLE

    def test_complex_routes_to_premium(self, router: SmartRouter) -> None:
        text = (
            "Analyze the trade-offs between different database architectures. "
            "Compare SQL vs NoSQL, evaluate the pros and cons, provide a "
            "comprehensive step by step guide for choosing the right database "
            "for a large-scale distributed system. Explain in detail the "
            "design patterns used in each approach."
        )
        decision = router.route_text(text)
        assert decision.quality_tier == QualityTier.PREMIUM

    def test_routing_with_required_quality(self, router: SmartRouter) -> None:
        decision = router.route_text(
            "hello",
            required_quality=QualityTier.PREMIUM,
        )
        assert decision.quality_tier == QualityTier.PREMIUM

    def test_routing_includes_estimated_cost(self, router: SmartRouter) -> None:
        decision = router.route_text("What is Python?")
        assert decision.estimated_cost >= 0

    def test_routing_includes_reason(self, router: SmartRouter) -> None:
        decision = router.route_text("hello")
        assert len(decision.reason) > 0

    def test_routing_with_prespecified_complexity(self, router: SmartRouter) -> None:
        request = RoutingRequest(
            content="hello",
            complexity=Complexity.COMPLEX,
        )
        decision = router.route(request)
        assert decision.complexity == Complexity.COMPLEX
        assert decision.quality_tier == QualityTier.PREMIUM

    def test_routing_with_max_cost_constraint(self, router: SmartRouter) -> None:
        """Route with a very tight cost constraint should still find a model."""
        decision = router.route_text("hello", max_cost=10.0)
        assert decision.estimated_cost <= 10.0

    def test_routing_no_model_found_raises(self, router: SmartRouter) -> None:
        """Extremely low max_cost should raise ValueError."""
        with pytest.raises(ValueError, match="No suitable model"):
            router.route_text("hello", max_cost=0.0)

    def test_route_text_convenience(self, router: SmartRouter) -> None:
        decision = router.route_text("Summarize briefly this paragraph.")
        assert decision.selected_model is not None


class TestGetModels:
    """Tests for model retrieval."""

    def test_get_all_models(self, router: SmartRouter) -> None:
        models = router.get_available_models()
        assert len(models) == 5

    def test_get_economy_models(self, router: SmartRouter) -> None:
        models = router.get_available_models(QualityTier.ECONOMY)
        assert len(models) >= 1
        assert all(m.quality_tier == QualityTier.ECONOMY for m in models)

    def test_get_premium_models(self, router: SmartRouter) -> None:
        models = router.get_available_models(QualityTier.PREMIUM)
        assert len(models) >= 1
        assert all(m.quality_tier == QualityTier.PREMIUM for m in models)
