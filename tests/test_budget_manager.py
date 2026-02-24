"""Tests for the BudgetManager module."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.budget_manager import BudgetManager
from src.cost_tracker import CostTracker
from src.models import AlertType, BudgetPeriod, UsageRecord


@pytest.fixture
def tracker_with_data() -> CostTracker:
    """Create a CostTracker with engineering spending data."""
    tracker = CostTracker()
    now = datetime.now()

    # Engineering has spent $45 this period
    records = [
        UsageRecord(
            timestamp=now - timedelta(hours=i),
            model="gemini-2.0-flash",
            department="engineering",
            project_id="chatbot",
            input_tokens=1000,
            output_tokens=500,
            cost=9.0,  # $9 each, 5 records = $45 total
            latency_ms=200.0,
        )
        for i in range(5)
    ]
    # Marketing has spent $5
    records.append(
        UsageRecord(
            timestamp=now,
            model="gpt-4o-mini",
            department="marketing",
            project_id="content-gen",
            input_tokens=300,
            output_tokens=200,
            cost=5.0,
            latency_ms=150.0,
        )
    )
    tracker.log_usage_batch(records)
    return tracker


@pytest.fixture
def budget_mgr(tracker_with_data: CostTracker) -> BudgetManager:
    """Create a BudgetManager with budgets configured."""
    mgr = BudgetManager(tracker_with_data)
    mgr.set_budget("engineering", 50.0, BudgetPeriod.MONTHLY)
    mgr.set_budget("marketing", 100.0, BudgetPeriod.MONTHLY)
    return mgr


class TestBudgetConfiguration:
    """Tests for budget setup and management."""

    def test_set_budget(self, budget_mgr: BudgetManager) -> None:
        config = budget_mgr.get_budget("engineering")
        assert config is not None
        assert config.budget_limit == 50.0
        assert config.period == BudgetPeriod.MONTHLY

    def test_set_weekly_budget(self, tracker_with_data: CostTracker) -> None:
        mgr = BudgetManager(tracker_with_data)
        config = mgr.set_budget("sales", 200.0, BudgetPeriod.WEEKLY)
        assert config.period == BudgetPeriod.WEEKLY

    def test_remove_budget(self, budget_mgr: BudgetManager) -> None:
        assert budget_mgr.remove_budget("engineering") is True
        assert budget_mgr.get_budget("engineering") is None

    def test_remove_nonexistent_budget(self, budget_mgr: BudgetManager) -> None:
        assert budget_mgr.remove_budget("nonexistent") is False

    def test_get_all_budgets(self, budget_mgr: BudgetManager) -> None:
        budgets = budget_mgr.get_all_budgets()
        assert len(budgets) == 2
        assert "engineering" in budgets
        assert "marketing" in budgets

    def test_custom_thresholds(self, tracker_with_data: CostTracker) -> None:
        mgr = BudgetManager(tracker_with_data)
        config = mgr.set_budget(
            "research",
            500.0,
            warning_threshold_pct=70.0,
            critical_threshold_pct=85.0,
        )
        assert config.warning_threshold_pct == 70.0
        assert config.critical_threshold_pct == 85.0


class TestBudgetChecking:
    """Tests for checking budget status."""

    def test_check_budget_engineering_critical(self, budget_mgr: BudgetManager) -> None:
        """Engineering spent $45 out of $50 budget = 90% = critical."""
        status = budget_mgr.check_budget("engineering")
        assert status["entity_id"] == "engineering"
        assert status["budget_limit"] == 50.0
        assert status["current_spend"] == 45.0
        assert status["status"] in ("critical", "exceeded")

    def test_check_budget_marketing_ok(self, budget_mgr: BudgetManager) -> None:
        """Marketing spent $5 out of $100 budget = 5% = ok."""
        status = budget_mgr.check_budget("marketing")
        assert status["entity_id"] == "marketing"
        assert status["status"] == "ok"
        assert status["usage_pct"] < 80.0

    def test_check_nonexistent_budget_raises(self, budget_mgr: BudgetManager) -> None:
        with pytest.raises(ValueError, match="No budget configured"):
            budget_mgr.check_budget("nonexistent")

    def test_check_all_budgets(self, budget_mgr: BudgetManager) -> None:
        all_status = budget_mgr.check_all_budgets()
        assert len(all_status) == 2
        entities = {s["entity_id"] for s in all_status}
        assert entities == {"engineering", "marketing"}


class TestAlerts:
    """Tests for alert generation."""

    def test_generate_alerts_critical(self, budget_mgr: BudgetManager) -> None:
        """Engineering at 90% should trigger critical alert."""
        alerts = budget_mgr.generate_alerts("engineering")
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.CRITICAL
        assert alerts[0].department == "engineering"

    def test_generate_alerts_ok(self, budget_mgr: BudgetManager) -> None:
        """Marketing at 5% should not trigger any alert."""
        alerts = budget_mgr.generate_alerts("marketing")
        assert len(alerts) == 0

    def test_generate_all_alerts(self, budget_mgr: BudgetManager) -> None:
        alerts = budget_mgr.generate_alerts()
        # At least engineering should have an alert
        assert len(alerts) >= 1
        alert_depts = {a.department for a in alerts}
        assert "engineering" in alert_depts

    def test_alert_message_format(self, budget_mgr: BudgetManager) -> None:
        alerts = budget_mgr.generate_alerts("engineering")
        assert len(alerts) > 0
        alert = alerts[0]
        assert "engineering" in alert.message
        assert "$" in alert.message


class TestForecasting:
    """Tests for spend forecasting."""

    def test_forecast_spend(self, budget_mgr: BudgetManager) -> None:
        forecast = budget_mgr.forecast_spend("engineering", days_ahead=30)
        assert forecast["entity_id"] == "engineering"
        assert forecast["current_spend"] > 0
        assert forecast["daily_rate"] >= 0
        assert forecast["projected_spend"] >= forecast["current_spend"]

    def test_forecast_includes_budget_limit(self, budget_mgr: BudgetManager) -> None:
        forecast = budget_mgr.forecast_spend("engineering")
        assert forecast["budget_limit"] == 50.0
        assert "will_exceed" in forecast

    def test_forecast_without_budget(self, tracker_with_data: CostTracker) -> None:
        """Forecast for entity without a budget should still work."""
        mgr = BudgetManager(tracker_with_data)
        forecast = mgr.forecast_spend("engineering", days_ahead=7)
        assert forecast["budget_limit"] is None
        assert forecast["will_exceed"] is None
