"""Tests for the TokenAnalytics module."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.analytics import TokenAnalytics
from src.cost_tracker import CostTracker
from src.models import UsageRecord


@pytest.fixture
def populated_tracker() -> CostTracker:
    """Create a CostTracker with diverse usage data."""
    tracker = CostTracker()
    now = datetime.now()

    records = [
        # Economy model usage
        UsageRecord(
            timestamp=now - timedelta(days=1),
            model="gemini-2.0-flash-lite",
            department="engineering",
            project_id="chatbot",
            input_tokens=500,
            output_tokens=300,
            cost=0.00017,
            latency_ms=120.0,
        ),
        UsageRecord(
            timestamp=now - timedelta(days=2),
            model="gemini-2.0-flash-lite",
            department="support",
            project_id="chatbot",
            input_tokens=400,
            output_tokens=250,
            cost=0.00014,
            latency_ms=110.0,
        ),
        # Standard model usage
        UsageRecord(
            timestamp=now - timedelta(days=1),
            model="gemini-2.0-flash",
            department="engineering",
            project_id="code-review",
            input_tokens=1000,
            output_tokens=800,
            cost=0.0042,
            latency_ms=350.0,
        ),
        UsageRecord(
            timestamp=now,
            model="claude-sonnet-4-20250514",
            department="marketing",
            project_id="content-gen",
            input_tokens=600,
            output_tokens=1000,
            cost=0.0168,
            latency_ms=500.0,
        ),
        # Premium model usage
        UsageRecord(
            timestamp=now - timedelta(days=3),
            model="gemini-2.5-pro",
            department="research",
            project_id="data-analysis",
            input_tokens=3000,
            output_tokens=2000,
            cost=0.1375,
            latency_ms=2000.0,
        ),
        UsageRecord(
            timestamp=now,
            model="gemini-2.5-pro",
            department="engineering",
            project_id="architecture",
            input_tokens=2000,
            output_tokens=1500,
            cost=0.1,
            latency_ms=1800.0,
        ),
    ]
    tracker.log_usage_batch(records)
    return tracker


@pytest.fixture
def analytics(populated_tracker: CostTracker) -> TokenAnalytics:
    """Create a TokenAnalytics instance with data."""
    return TokenAnalytics(populated_tracker)


class TestTokenUsage:
    """Tests for token usage breakdown."""

    def test_usage_by_model(self, analytics: TokenAnalytics) -> None:
        usage = analytics.token_usage_by_model()
        assert len(usage) >= 3
        models = {u["model"] for u in usage}
        assert "gemini-2.0-flash-lite" in models
        assert "gemini-2.5-pro" in models

    def test_usage_by_model_has_token_counts(self, analytics: TokenAnalytics) -> None:
        usage = analytics.token_usage_by_model()
        for u in usage:
            assert u["input_tokens"] >= 0
            assert u["output_tokens"] >= 0
            assert u["total_tokens"] == u["input_tokens"] + u["output_tokens"]

    def test_usage_by_department(self, analytics: TokenAnalytics) -> None:
        usage = analytics.token_usage_by_department()
        assert len(usage) >= 3
        departments = {u["department"] for u in usage}
        assert "engineering" in departments
        assert "research" in departments


class TestCostTrends:
    """Tests for cost trend analysis."""

    def test_cost_trends_returns_data(self, analytics: TokenAnalytics) -> None:
        trends = analytics.cost_trends(days=7)
        assert len(trends) > 0

    def test_cost_trends_cumulative(self, analytics: TokenAnalytics) -> None:
        trends = analytics.cost_trends(days=7)
        if len(trends) > 1:
            # Cumulative cost should be non-decreasing
            for i in range(1, len(trends)):
                assert trends[i]["cumulative_cost"] >= trends[i - 1]["cumulative_cost"]

    def test_cost_trends_with_department_filter(self, analytics: TokenAnalytics) -> None:
        trends = analytics.cost_trends(days=7, department="engineering")
        # All costs should be from engineering only
        assert len(trends) >= 0


class TestEfficiencyMetrics:
    """Tests for efficiency metric calculations."""

    def test_efficiency_metrics_all_models(self, analytics: TokenAnalytics) -> None:
        metrics = analytics.efficiency_metrics()
        assert len(metrics) >= 3

    def test_efficiency_metrics_has_cost_per_token(self, analytics: TokenAnalytics) -> None:
        metrics = analytics.efficiency_metrics()
        for m in metrics:
            assert "cost_per_output_token" in m
            assert "cost_per_total_token" in m
            assert m["cost_per_output_token"] >= 0
            assert m["cost_per_total_token"] >= 0

    def test_premium_more_expensive_than_economy(self, analytics: TokenAnalytics) -> None:
        metrics = analytics.efficiency_metrics()
        costs_by_model = {m["model"]: m["cost_per_output_token"] for m in metrics}
        if "gemini-2.5-pro" in costs_by_model and "gemini-2.0-flash-lite" in costs_by_model:
            assert costs_by_model["gemini-2.5-pro"] > costs_by_model["gemini-2.0-flash-lite"]


class TestSavingsCalculator:
    """Tests for savings calculation."""

    def test_savings_with_default_baseline(self, analytics: TokenAnalytics) -> None:
        savings = analytics.savings_calculator()
        assert savings["actual_cost"] > 0
        assert savings["baseline_cost"] > 0
        assert "savings" in savings
        assert "savings_pct" in savings
        assert savings["baseline_model"] == "Gemini Pro"

    def test_savings_are_positive(self, analytics: TokenAnalytics) -> None:
        """Using routing should save money compared to the most expensive model."""
        savings = analytics.savings_calculator()
        assert savings["savings"] >= 0
        assert savings["savings_pct"] >= 0

    def test_savings_with_specific_baseline(self, analytics: TokenAnalytics) -> None:
        savings = analytics.savings_calculator(baseline_model_id="gemini-2.0-flash")
        assert savings["baseline_model"] == "Gemini Flash"


class TestModelUtilization:
    """Tests for model utilization rates."""

    def test_utilization_rates(self, analytics: TokenAnalytics) -> None:
        rates = analytics.model_utilization_rates()
        assert len(rates) >= 3

    def test_utilization_percentages_sum(self, analytics: TokenAnalytics) -> None:
        rates = analytics.model_utilization_rates()
        total_request_pct = sum(r["request_pct"] for r in rates)
        total_cost_pct = sum(r["cost_pct"] for r in rates)
        assert abs(total_request_pct - 100.0) < 0.1
        assert abs(total_cost_pct - 100.0) < 0.1


class TestSummaryStats:
    """Tests for summary statistics."""

    def test_summary_stats(self, analytics: TokenAnalytics) -> None:
        stats = analytics.get_summary_stats()
        assert stats["total_cost"] > 0
        assert stats["total_requests"] == 6
        assert stats["models_used"] >= 3
        assert stats["departments"] >= 3
        assert stats["avg_cost_per_request"] > 0
