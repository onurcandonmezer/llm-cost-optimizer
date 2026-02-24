"""Tests for the CostTracker module."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.cost_tracker import CostTracker
from src.models import UsageRecord


@pytest.fixture
def tracker() -> CostTracker:
    """Create an in-memory CostTracker."""
    return CostTracker()


@pytest.fixture
def populated_tracker() -> CostTracker:
    """Create a CostTracker pre-populated with sample data."""
    tracker = CostTracker()
    now = datetime.now()

    records = [
        UsageRecord(
            timestamp=now - timedelta(days=1),
            model="gemini-2.0-flash-lite",
            department="engineering",
            project_id="chatbot",
            input_tokens=500,
            output_tokens=300,
            cost=0.00017,
            latency_ms=150.0,
        ),
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
            timestamp=now - timedelta(days=2),
            model="gemini-2.5-pro",
            department="research",
            project_id="data-analysis",
            input_tokens=2000,
            output_tokens=1500,
            cost=0.1,
            latency_ms=1200.0,
        ),
        UsageRecord(
            timestamp=now,
            model="gpt-4o-mini",
            department="marketing",
            project_id="content-gen",
            input_tokens=300,
            output_tokens=500,
            cost=0.000345,
            latency_ms=200.0,
        ),
        UsageRecord(
            timestamp=now,
            model="gemini-2.0-flash-lite",
            department="engineering",
            project_id="chatbot",
            input_tokens=200,
            output_tokens=100,
            cost=0.00006,
            latency_ms=100.0,
        ),
    ]
    tracker.log_usage_batch(records)
    return tracker


class TestLogUsage:
    """Tests for logging usage records."""

    def test_log_single_record(self, tracker: CostTracker) -> None:
        record = UsageRecord(
            model="gemini-2.0-flash",
            department="engineering",
            input_tokens=100,
            output_tokens=50,
            cost=0.003,
            latency_ms=100.0,
        )
        record_id = tracker.log_usage(record)
        assert record_id is not None
        assert record_id > 0

    def test_log_batch_records(self, tracker: CostTracker) -> None:
        records = [
            UsageRecord(
                model=f"model-{i}",
                department="test",
                input_tokens=100 * i,
                output_tokens=50 * i,
                cost=0.001 * i,
                latency_ms=100.0,
            )
            for i in range(1, 6)
        ]
        count = tracker.log_usage_batch(records)
        assert count == 5
        assert tracker.get_record_count() == 5


class TestQueryMethods:
    """Tests for querying cost data."""

    def test_total_cost(self, populated_tracker: CostTracker) -> None:
        total = populated_tracker.total_cost()
        assert total > 0
        # Sum of all records: 0.00017 + 0.0042 + 0.1 + 0.000345 + 0.00006
        expected = round(0.00017 + 0.0042 + 0.1 + 0.000345 + 0.00006, 6)
        assert abs(total - expected) < 0.0001

    def test_avg_cost_per_request(self, populated_tracker: CostTracker) -> None:
        avg = populated_tracker.avg_cost_per_request()
        assert avg > 0
        total = populated_tracker.total_cost()
        count = populated_tracker.get_record_count()
        assert abs(avg - total / count) < 0.0001

    def test_costs_by_department(self, populated_tracker: CostTracker) -> None:
        summaries = populated_tracker.get_costs_by_department()
        assert len(summaries) == 3  # engineering, research, marketing
        departments = {s.entity for s in summaries}
        assert departments == {"engineering", "research", "marketing"}

    def test_costs_by_project(self, populated_tracker: CostTracker) -> None:
        summaries = populated_tracker.get_costs_by_project()
        assert len(summaries) >= 3
        projects = {s.entity for s in summaries}
        assert "chatbot" in projects
        assert "data-analysis" in projects

    def test_costs_by_project_filtered(self, populated_tracker: CostTracker) -> None:
        summaries = populated_tracker.get_costs_by_project(department="engineering")
        projects = {s.entity for s in summaries}
        assert "chatbot" in projects
        assert "data-analysis" not in projects

    def test_costs_by_model(self, populated_tracker: CostTracker) -> None:
        summaries = populated_tracker.get_costs_by_model()
        assert len(summaries) >= 3
        models = {s.entity for s in summaries}
        assert "gemini-2.0-flash-lite" in models

    def test_daily_costs(self, populated_tracker: CostTracker) -> None:
        daily = populated_tracker.get_daily_costs(days=7)
        assert len(daily) > 0
        for entry in daily:
            assert "date" in entry
            assert "total_cost" in entry
            assert "request_count" in entry

    def test_department_spend(self, populated_tracker: CostTracker) -> None:
        spend = populated_tracker.get_department_spend("engineering")
        assert spend > 0

    def test_department_spend_nonexistent(self, populated_tracker: CostTracker) -> None:
        spend = populated_tracker.get_department_spend("nonexistent")
        assert spend == 0.0

    def test_top_spending_departments(self, populated_tracker: CostTracker) -> None:
        top = populated_tracker.top_spending_departments(limit=2)
        assert len(top) <= 2
        # The top spender should be research (0.1) or engineering
        assert top[0].total_cost >= top[-1].total_cost

    def test_record_count(self, populated_tracker: CostTracker) -> None:
        assert populated_tracker.get_record_count() == 5

    def test_empty_tracker(self, tracker: CostTracker) -> None:
        assert tracker.total_cost() == 0.0
        assert tracker.avg_cost_per_request() == 0.0
        assert tracker.get_record_count() == 0
        assert len(tracker.get_costs_by_department()) == 0
