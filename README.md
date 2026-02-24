# LLM Cost Optimizer

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/Gemini-8E75B2?logo=google&logoColor=white)](https://ai.google.dev/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)](https://sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/onurcandnmz/llm-cost-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/onurcandnmz/llm-cost-optimizer/actions)

Smart request routing and cost management system for LLM APIs. Automatically classifies task complexity and routes requests to the most cost-effective model while maintaining quality.

## Overview

LLM Cost Optimizer solves the problem of runaway API costs by intelligently routing each request to the cheapest model that can handle its complexity level. Simple questions go to economy models, while complex analysis tasks are sent to premium models -- saving up to 70% on LLM API costs without sacrificing output quality.

## Key Features

- **Smart Routing Engine** -- Classifies request complexity using text heuristics (length, keywords, structure) and maps to optimal model tiers
- **Multi-Model Support** -- Routes across Gemini, GPT-4o, and Claude models with configurable cost/quality tiers
- **Cost Tracking** -- SQLite-backed usage logging with per-department, per-project, and per-model breakdowns
- **Budget Management** -- Set spending limits with warning (80%) and critical (90%) threshold alerts
- **Spend Forecasting** -- Linear projection of costs based on current trends to predict budget overruns
- **Savings Analytics** -- Calculates actual savings compared to single-model usage
- **Interactive Dashboard** -- Streamlit UI for monitoring costs, testing routing, and viewing analytics

## Architecture

```
                    +------------------+
                    |  Routing Request  |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Complexity        |
                    | Classifier        |
                    | (text heuristics) |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
        +-----v----+  +-----v----+  +------v-----+
        | Economy   |  | Standard |  | Premium    |
        | Tier      |  | Tier     |  | Tier       |
        | Flash Lite|  | Flash    |  | Gemini Pro |
        | GPT-4o    |  | Claude   |  |            |
        | Mini      |  | Sonnet   |  |            |
        +-----+----+  +-----+----+  +------+-----+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v---------+
                    |   Cost Tracker    |
                    |   (SQLite)        |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
        +-----v----+  +-----v----+  +------v-----+
        | Budget    |  | Analytics|  | Dashboard  |
        | Manager   |  | Engine   |  | (Streamlit)|
        +-----------+  +----------+  +------------+
```

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/onurcandnmz/llm-cost-optimizer.git
cd llm-cost-optimizer

# Install dependencies
uv pip install -e ".[dev]"

# Run tests
uv run python -m pytest tests/ -v --tb=short

# Launch the dashboard
uv run streamlit run src/app.py
```

## Usage

### Smart Routing

```python
from src.router import SmartRouter

router = SmartRouter()

# Simple request -> routes to economy tier
decision = router.route_text("What is Python?")
print(decision.selected_model.name)    # "GPT-4o Mini"
print(decision.estimated_cost)         # ~$0.00003
print(decision.quality_tier)           # "economy"

# Complex request -> routes to premium tier
decision = router.route_text(
    "Analyze the trade-offs between microservices and monolithic "
    "architectures. Compare scalability, deployment complexity, "
    "and provide a step by step migration guide."
)
print(decision.selected_model.name)    # "Gemini Pro"
print(decision.quality_tier)           # "premium"
```

### Cost Tracking

```python
from src.cost_tracker import CostTracker
from src.models import UsageRecord

tracker = CostTracker()  # in-memory, or pass db_path for persistence

# Log usage
tracker.log_usage(UsageRecord(
    model="gemini-2.0-flash",
    department="engineering",
    project_id="chatbot",
    input_tokens=1000,
    output_tokens=500,
    cost=0.003,
    latency_ms=250.0,
))

# Query costs
print(tracker.total_cost())
print(tracker.get_costs_by_department())
print(tracker.top_spending_departments(limit=3))
```

### Budget Management

```python
from src.budget_manager import BudgetManager
from src.models import BudgetPeriod

budget_mgr = BudgetManager(tracker)

# Set department budgets
budget_mgr.set_budget("engineering", 500.0, BudgetPeriod.MONTHLY)

# Check budget status
status = budget_mgr.check_budget("engineering")
print(status["usage_pct"])   # e.g., 45.2%
print(status["status"])      # "ok" | "warning" | "critical" | "exceeded"

# Generate alerts
alerts = budget_mgr.generate_alerts()
for alert in alerts:
    print(alert.message)

# Forecast spending
forecast = budget_mgr.forecast_spend("engineering", days_ahead=30)
print(forecast["will_exceed"])  # True/False
```

### Savings Analytics

```python
from src.analytics import TokenAnalytics

analytics = TokenAnalytics(tracker)

# Calculate savings from routing
savings = analytics.savings_calculator()
print(f"Saved ${savings['savings']:.2f} ({savings['savings_pct']:.1f}%)")

# Efficiency metrics
for metric in analytics.efficiency_metrics():
    print(f"{metric['model']}: ${metric['cost_per_output_token']:.8f}/token")

# Model utilization
for rate in analytics.model_utilization_rates():
    print(f"{rate['model']}: {rate['request_pct']:.1f}% of requests")
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Data Validation | Pydantic v2 |
| Configuration | PyYAML |
| Database | SQLite |
| API Framework | FastAPI |
| Dashboard | Streamlit + Plotly |
| Package Manager | uv |
| Linting | Ruff |
| Testing | pytest + pytest-cov |

## Project Structure

```
llm-cost-optimizer/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── models.py            # Pydantic data models
│   ├── router.py            # Smart request routing engine
│   ├── cost_tracker.py      # SQLite-backed cost tracking
│   ├── budget_manager.py    # Budget alerts and forecasting
│   ├── analytics.py         # Token usage analytics
│   └── app.py               # Streamlit cost dashboard
├── tests/
│   ├── test_router.py       # Router tests (14 tests)
│   ├── test_cost_tracker.py # Cost tracker tests (14 tests)
│   ├── test_budget_manager.py # Budget manager tests (14 tests)
│   └── test_analytics.py    # Analytics tests (15 tests)
├── configs/
│   └── routing_config.yaml  # Model routing configuration
├── pyproject.toml            # Project configuration
├── Makefile                  # Development commands
└── LICENSE                   # MIT License
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 Onurcan Donmezer
