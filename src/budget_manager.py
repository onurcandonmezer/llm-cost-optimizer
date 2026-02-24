"""Budget management module with alerts and forecasting.

Provides budget tracking, threshold-based alerting, and
linear cost projection for departments and projects.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from src.cost_tracker import CostTracker
from src.models import (
    AlertType,
    BudgetAlert,
    BudgetConfig,
    BudgetPeriod,
)


class BudgetManager:
    """Manages budgets, generates alerts, and forecasts spending.

    Works with CostTracker to monitor spending against configured
    budgets and generate alerts when thresholds are crossed.
    """

    def __init__(self, cost_tracker: CostTracker) -> None:
        """Initialize the budget manager.

        Args:
            cost_tracker: CostTracker instance for querying spend data.
        """
        self.cost_tracker = cost_tracker
        self._budgets: dict[str, BudgetConfig] = {}

    def set_budget(
        self,
        entity_id: str,
        budget_limit: float,
        period: BudgetPeriod = BudgetPeriod.MONTHLY,
        warning_threshold_pct: float = 80.0,
        critical_threshold_pct: float = 90.0,
    ) -> BudgetConfig:
        """Set a budget for a department or project.

        Args:
            entity_id: Department or project identifier.
            budget_limit: Budget limit in USD.
            period: Budget period (weekly or monthly).
            warning_threshold_pct: Warning alert threshold percentage.
            critical_threshold_pct: Critical alert threshold percentage.

        Returns:
            The created budget configuration.
        """
        config = BudgetConfig(
            entity_id=entity_id,
            budget_limit=budget_limit,
            period=period,
            warning_threshold_pct=warning_threshold_pct,
            critical_threshold_pct=critical_threshold_pct,
        )
        self._budgets[entity_id] = config
        return config

    def remove_budget(self, entity_id: str) -> bool:
        """Remove a budget configuration.

        Args:
            entity_id: Department or project identifier.

        Returns:
            True if budget was removed, False if it did not exist.
        """
        if entity_id in self._budgets:
            del self._budgets[entity_id]
            return True
        return False

    def get_budget(self, entity_id: str) -> BudgetConfig | None:
        """Get the budget configuration for an entity.

        Args:
            entity_id: Department or project identifier.

        Returns:
            Budget configuration or None if not set.
        """
        return self._budgets.get(entity_id)

    def get_all_budgets(self) -> dict[str, BudgetConfig]:
        """Get all configured budgets.

        Returns:
            Dictionary mapping entity IDs to budget configurations.
        """
        return dict(self._budgets)

    def _get_period_start(self, period: BudgetPeriod) -> datetime:
        """Calculate the start date for the current budget period.

        Args:
            period: The budget period type.

        Returns:
            Start datetime for the current period.
        """
        now = datetime.now()
        if period == BudgetPeriod.MONTHLY:
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:  # weekly
            start_of_week = now - timedelta(days=now.weekday())
            return start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)

    def check_budget(self, entity_id: str) -> dict:
        """Check the budget status for an entity.

        Args:
            entity_id: Department or project identifier.

        Returns:
            Dictionary with budget status information including:
            - entity_id, budget_limit, current_spend, remaining,
              usage_pct, period, status ('ok', 'warning', 'critical', 'exceeded')

        Raises:
            ValueError: If no budget is set for the entity.
        """
        config = self._budgets.get(entity_id)
        if config is None:
            raise ValueError(f"No budget configured for '{entity_id}'")

        period_start = self._get_period_start(config.period)
        current_spend = self.cost_tracker.get_department_spend(entity_id, start_date=period_start)

        remaining = max(0, config.budget_limit - current_spend)
        usage_pct = (current_spend / config.budget_limit * 100) if config.budget_limit > 0 else 0

        if usage_pct >= 100:
            status = "exceeded"
        elif usage_pct >= config.critical_threshold_pct:
            status = "critical"
        elif usage_pct >= config.warning_threshold_pct:
            status = "warning"
        else:
            status = "ok"

        return {
            "entity_id": entity_id,
            "budget_limit": config.budget_limit,
            "current_spend": round(current_spend, 6),
            "remaining": round(remaining, 6),
            "usage_pct": round(usage_pct, 2),
            "period": config.period.value,
            "status": status,
        }

    def check_all_budgets(self) -> list[dict]:
        """Check budget status for all configured entities.

        Returns:
            List of budget status dictionaries.
        """
        return [self.check_budget(entity_id) for entity_id in self._budgets]

    def generate_alerts(self, entity_id: str | None = None) -> list[BudgetAlert]:
        """Generate alerts for entities that have crossed thresholds.

        Args:
            entity_id: Specific entity to check, or None for all.

        Returns:
            List of budget alerts for entities exceeding thresholds.
        """
        alerts: list[BudgetAlert] = []
        entities = [entity_id] if entity_id else list(self._budgets.keys())

        for eid in entities:
            config = self._budgets.get(eid)
            if config is None:
                continue

            period_start = self._get_period_start(config.period)
            current_spend = self.cost_tracker.get_department_spend(eid, start_date=period_start)

            usage_pct = (
                (current_spend / config.budget_limit * 100) if config.budget_limit > 0 else 0
            )

            if usage_pct >= config.critical_threshold_pct:
                alerts.append(
                    BudgetAlert(
                        department=eid,
                        budget_limit=config.budget_limit,
                        current_spend=round(current_spend, 6),
                        threshold_pct=config.critical_threshold_pct,
                        alert_type=AlertType.CRITICAL,
                    )
                )
            elif usage_pct >= config.warning_threshold_pct:
                alerts.append(
                    BudgetAlert(
                        department=eid,
                        budget_limit=config.budget_limit,
                        current_spend=round(current_spend, 6),
                        threshold_pct=config.warning_threshold_pct,
                        alert_type=AlertType.WARNING,
                    )
                )

        return alerts

    def forecast_spend(
        self,
        entity_id: str,
        days_ahead: int = 30,
    ) -> dict:
        """Forecast future spending using linear projection.

        Uses the daily cost trend from the current period to project
        future spending.

        Args:
            entity_id: Department or project identifier.
            days_ahead: Number of days to project into the future.

        Returns:
            Dictionary with forecast information:
            - entity_id, current_spend, daily_rate, projected_spend,
              days_ahead, projected_end_of_period
        """
        config = self._budgets.get(entity_id)
        period_start = self._get_period_start(config.period if config else BudgetPeriod.MONTHLY)

        now = datetime.now()
        days_elapsed = max((now - period_start).days, 1)

        current_spend = self.cost_tracker.get_department_spend(entity_id, start_date=period_start)

        daily_rate = current_spend / days_elapsed
        projected_spend = current_spend + (daily_rate * days_ahead)

        # Project to end of period
        if config:
            days_in_period = 30 if config.period == BudgetPeriod.MONTHLY else 7
            remaining_days = max(days_in_period - days_elapsed, 0)
            projected_end_of_period = current_spend + (daily_rate * remaining_days)
        else:
            projected_end_of_period = projected_spend

        return {
            "entity_id": entity_id,
            "current_spend": round(current_spend, 6),
            "daily_rate": round(daily_rate, 6),
            "projected_spend": round(projected_spend, 6),
            "days_ahead": days_ahead,
            "projected_end_of_period": round(projected_end_of_period, 6),
            "budget_limit": config.budget_limit if config else None,
            "will_exceed": (projected_end_of_period > config.budget_limit if config else None),
        }
