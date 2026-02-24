"""Token usage analytics module.

Provides analytical capabilities for understanding LLM usage patterns,
cost trends, efficiency metrics, and potential savings from smart routing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml

from src.cost_tracker import CostTracker
from src.models import ModelConfig

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "routing_config.yaml"


class TokenAnalytics:
    """Analytics engine for LLM token usage and cost optimization.

    Provides insights into usage patterns, cost trends, efficiency
    metrics, and potential savings from intelligent routing.
    """

    def __init__(
        self,
        cost_tracker: CostTracker,
        config_path: Optional[str | Path] = None,
    ) -> None:
        """Initialize the analytics engine.

        Args:
            cost_tracker: CostTracker instance for querying usage data.
            config_path: Path to routing configuration for model pricing.
        """
        self.cost_tracker = cost_tracker
        config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._load_models(config_path)

    def _load_models(self, config_path: Path) -> None:
        """Load model configurations for pricing data."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.models: dict[str, ModelConfig] = {
            m["model_id"]: ModelConfig(**m) for m in config["models"]
        }
        self._most_expensive_model: Optional[ModelConfig] = None
        if self.models:
            self._most_expensive_model = max(
                self.models.values(), key=lambda m: m.cost_per_1k_output
            )

    def token_usage_by_model(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[dict]:
        """Get token usage breakdown by model.

        Args:
            start_date: Optional start of date range.
            end_date: Optional end of date range.

        Returns:
            List of dicts with model, input_tokens, output_tokens,
            total_tokens, and cost for each model.
        """
        summaries = self.cost_tracker.get_costs_by_model(start_date, end_date)
        return [
            {
                "model": s.entity,
                "input_tokens": s.total_input_tokens,
                "output_tokens": s.total_output_tokens,
                "total_tokens": s.total_input_tokens + s.total_output_tokens,
                "cost": s.total_cost,
                "request_count": s.request_count,
            }
            for s in summaries
        ]

    def token_usage_by_department(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[dict]:
        """Get token usage breakdown by department.

        Args:
            start_date: Optional start of date range.
            end_date: Optional end of date range.

        Returns:
            List of dicts with department, tokens, and cost data.
        """
        summaries = self.cost_tracker.get_costs_by_department(start_date, end_date)
        return [
            {
                "department": s.entity,
                "input_tokens": s.total_input_tokens,
                "output_tokens": s.total_output_tokens,
                "total_tokens": s.total_input_tokens + s.total_output_tokens,
                "cost": s.total_cost,
                "request_count": s.request_count,
                "avg_cost_per_request": s.avg_cost_per_request,
            }
            for s in summaries
        ]

    def cost_trends(
        self,
        days: int = 30,
        department: Optional[str] = None,
    ) -> list[dict]:
        """Get cost trends over time.

        Args:
            days: Number of past days to analyze.
            department: Optional department filter.

        Returns:
            List of dicts with date, total_cost, request_count,
            and cumulative_cost.
        """
        daily = self.cost_tracker.get_daily_costs(days, department)
        cumulative = 0.0
        result = []
        for entry in daily:
            cumulative += entry["total_cost"]
            result.append(
                {
                    "date": entry["date"],
                    "daily_cost": entry["total_cost"],
                    "request_count": entry["request_count"],
                    "cumulative_cost": round(cumulative, 6),
                }
            )
        return result

    def efficiency_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[dict]:
        """Calculate efficiency metrics per model.

        Computes cost per output token and cost per total token
        for each model used.

        Args:
            start_date: Optional start of date range.
            end_date: Optional end of date range.

        Returns:
            List of dicts with model, cost_per_output_token,
            cost_per_total_token, and avg_latency_ms.
        """
        summaries = self.cost_tracker.get_costs_by_model(start_date, end_date)
        metrics = []
        for s in summaries:
            cost_per_output = (
                s.total_cost / s.total_output_tokens
                if s.total_output_tokens > 0
                else 0
            )
            total_tokens = s.total_input_tokens + s.total_output_tokens
            cost_per_total = s.total_cost / total_tokens if total_tokens > 0 else 0

            metrics.append(
                {
                    "model": s.entity,
                    "cost_per_output_token": round(cost_per_output, 8),
                    "cost_per_total_token": round(cost_per_total, 8),
                    "avg_latency_ms": s.avg_latency_ms,
                    "total_cost": s.total_cost,
                    "request_count": s.request_count,
                }
            )
        return metrics

    def savings_calculator(
        self,
        baseline_model_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """Calculate savings from smart routing vs using a single model.

        Compares actual cost (with routing) against the hypothetical
        cost of sending all requests to a single expensive model.

        Args:
            baseline_model_id: Model ID to use as baseline. Uses the
                              most expensive model if None.
            start_date: Optional start of date range.
            end_date: Optional end of date range.

        Returns:
            Dictionary with actual_cost, baseline_cost, savings,
            savings_pct, and baseline_model.
        """
        if baseline_model_id and baseline_model_id in self.models:
            baseline_model = self.models[baseline_model_id]
        elif self._most_expensive_model:
            baseline_model = self._most_expensive_model
        else:
            return {
                "actual_cost": 0,
                "baseline_cost": 0,
                "savings": 0,
                "savings_pct": 0,
                "baseline_model": "none",
            }

        # Get actual costs
        actual_cost = self.cost_tracker.total_cost(start_date, end_date)

        # Calculate what it would have cost with the baseline model
        summaries = self.cost_tracker.get_costs_by_model(start_date, end_date)
        baseline_cost = 0.0
        for s in summaries:
            baseline_cost += baseline_model.estimate_cost(
                s.total_input_tokens, s.total_output_tokens
            )

        savings = baseline_cost - actual_cost
        savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

        return {
            "actual_cost": round(actual_cost, 6),
            "baseline_cost": round(baseline_cost, 6),
            "savings": round(savings, 6),
            "savings_pct": round(savings_pct, 2),
            "baseline_model": baseline_model.name,
        }

    def model_utilization_rates(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[dict]:
        """Calculate model utilization rates.

        Shows the percentage of total requests and cost attributed
        to each model.

        Args:
            start_date: Optional start of date range.
            end_date: Optional end of date range.

        Returns:
            List of dicts with model, request_pct, cost_pct,
            request_count, and cost.
        """
        summaries = self.cost_tracker.get_costs_by_model(start_date, end_date)

        total_requests = sum(s.request_count for s in summaries)
        total_cost = sum(s.total_cost for s in summaries)

        rates = []
        for s in summaries:
            request_pct = (
                s.request_count / total_requests * 100 if total_requests > 0 else 0
            )
            cost_pct = s.total_cost / total_cost * 100 if total_cost > 0 else 0

            rates.append(
                {
                    "model": s.entity,
                    "request_count": s.request_count,
                    "request_pct": round(request_pct, 2),
                    "cost": s.total_cost,
                    "cost_pct": round(cost_pct, 2),
                }
            )
        return rates

    def get_summary_stats(self) -> dict:
        """Get high-level summary statistics.

        Returns:
            Dictionary with total_cost, total_requests, avg_cost_per_request,
            models_used, and departments count.
        """
        total_cost = self.cost_tracker.total_cost()
        avg_cost = self.cost_tracker.avg_cost_per_request()
        models = self.cost_tracker.get_costs_by_model()
        departments = self.cost_tracker.get_costs_by_department()
        record_count = self.cost_tracker.get_record_count()

        return {
            "total_cost": total_cost,
            "total_requests": record_count,
            "avg_cost_per_request": avg_cost,
            "models_used": len(models),
            "departments": len(departments),
        }
