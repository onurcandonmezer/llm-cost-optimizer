"""Pydantic data models for the LLM Cost Optimizer."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class QualityTier(StrEnum):
    """Quality tier for LLM models."""

    PREMIUM = "premium"
    STANDARD = "standard"
    ECONOMY = "economy"


class Complexity(StrEnum):
    """Task complexity classification."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class AlertType(StrEnum):
    """Budget alert severity levels."""

    WARNING = "warning"
    CRITICAL = "critical"


class BudgetPeriod(StrEnum):
    """Budget period options."""

    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ModelConfig(BaseModel):
    """Configuration for an LLM model including pricing and capabilities."""

    name: str = Field(description="Human-readable model name")
    provider: str = Field(description="Model provider (e.g., google, openai, anthropic)")
    model_id: str = Field(description="API model identifier")
    cost_per_1k_input: float = Field(ge=0, description="Cost per 1,000 input tokens in USD")
    cost_per_1k_output: float = Field(ge=0, description="Cost per 1,000 output tokens in USD")
    max_tokens: int = Field(gt=0, description="Maximum context window size")
    quality_tier: QualityTier = Field(description="Quality tier classification")

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost for a given number of tokens."""
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return round(input_cost + output_cost, 6)


class RoutingRequest(BaseModel):
    """A request to be routed to an appropriate LLM model."""

    content: str = Field(description="The text content to process")
    complexity: Optional[Complexity] = Field(
        default=None, description="Pre-classified complexity (auto-detected if None)"
    )
    department: Optional[str] = Field(default=None, description="Department making the request")
    project_id: Optional[str] = Field(default=None, description="Project identifier")
    max_cost: Optional[float] = Field(
        default=None, ge=0, description="Maximum acceptable cost in USD"
    )
    required_quality: Optional[QualityTier] = Field(
        default=None, description="Minimum required quality tier"
    )


class RoutingDecision(BaseModel):
    """The routing decision made by the SmartRouter."""

    selected_model: ModelConfig = Field(description="The chosen model configuration")
    reason: str = Field(description="Explanation for the routing decision")
    estimated_cost: float = Field(ge=0, description="Estimated cost in USD")
    quality_tier: QualityTier = Field(description="Quality tier of the selected model")
    complexity: Complexity = Field(description="Detected or specified complexity")


class UsageRecord(BaseModel):
    """Record of a single LLM API usage."""

    timestamp: datetime = Field(default_factory=datetime.now)
    model: str = Field(description="Model identifier used")
    department: str = Field(default="default", description="Department that made the request")
    project_id: str = Field(default="default", description="Project identifier")
    input_tokens: int = Field(ge=0, description="Number of input tokens consumed")
    output_tokens: int = Field(ge=0, description="Number of output tokens generated")
    cost: float = Field(ge=0, description="Total cost in USD")
    latency_ms: float = Field(ge=0, description="Request latency in milliseconds")


class BudgetAlert(BaseModel):
    """Alert generated when budget thresholds are reached."""

    department: str = Field(description="Department that triggered the alert")
    budget_limit: float = Field(ge=0, description="Budget limit in USD")
    current_spend: float = Field(ge=0, description="Current spending in USD")
    threshold_pct: float = Field(ge=0, le=100, description="Threshold percentage that was crossed")
    alert_type: AlertType = Field(description="Alert severity level")
    message: str = Field(default="", description="Human-readable alert message")
    timestamp: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context: object) -> None:
        """Generate default message if not provided."""
        if not self.message:
            pct_used = (self.current_spend / self.budget_limit * 100) if self.budget_limit > 0 else 0
            self.message = (
                f"{self.alert_type.value.upper()}: Department '{self.department}' has used "
                f"{pct_used:.1f}% of its ${self.budget_limit:.2f} budget "
                f"(${self.current_spend:.2f} spent)"
            )


class BudgetConfig(BaseModel):
    """Budget configuration for a department or project."""

    entity_id: str = Field(description="Department or project identifier")
    budget_limit: float = Field(ge=0, description="Budget limit in USD")
    period: BudgetPeriod = Field(default=BudgetPeriod.MONTHLY)
    warning_threshold_pct: float = Field(default=80.0, ge=0, le=100)
    critical_threshold_pct: float = Field(default=90.0, ge=0, le=100)


class CostSummary(BaseModel):
    """Summary of costs for a given period or entity."""

    entity: str = Field(description="Department, project, or model name")
    total_cost: float = Field(ge=0)
    request_count: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    avg_cost_per_request: float = Field(ge=0)
    avg_latency_ms: float = Field(ge=0)
