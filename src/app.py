"""Streamlit cost dashboard for LLM Cost Optimizer.

Provides an interactive web interface for monitoring costs,
testing the router, viewing department breakdowns, budget status,
and savings reports.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import TokenAnalytics
from src.budget_manager import BudgetManager
from src.cost_tracker import CostTracker
from src.models import BudgetPeriod, UsageRecord
from src.router import SmartRouter


def generate_sample_data(tracker: CostTracker, days: int = 30) -> None:
    """Generate sample usage data for demonstration."""
    models = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "gpt-4o-mini",
        "claude-sonnet-4-20250514",
    ]
    departments = ["engineering", "marketing", "research", "support", "sales"]
    projects = ["chatbot", "content-gen", "data-analysis", "code-review", "translation"]

    cost_map = {
        "gemini-2.0-flash-lite": (0.0001, 0.0004),
        "gemini-2.0-flash": (0.0010, 0.0040),
        "gemini-2.5-pro": (0.0125, 0.0500),
        "gpt-4o-mini": (0.00015, 0.0006),
        "claude-sonnet-4-20250514": (0.003, 0.015),
    }

    records = []
    now = datetime.now()

    for day_offset in range(days):
        date = now - timedelta(days=day_offset)
        num_requests = random.randint(5, 25)

        for _ in range(num_requests):
            model = random.choice(models)
            dept = random.choice(departments)
            project = random.choice(projects)
            input_tokens = random.randint(100, 5000)
            output_tokens = random.randint(50, 3000)

            input_rate, output_rate = cost_map[model]
            cost = (input_tokens / 1000) * input_rate + (output_tokens / 1000) * output_rate

            records.append(
                UsageRecord(
                    timestamp=date.replace(
                        hour=random.randint(8, 20),
                        minute=random.randint(0, 59),
                    ),
                    model=model,
                    department=dept,
                    project_id=project,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=round(cost, 6),
                    latency_ms=round(random.uniform(100, 3000), 2),
                )
            )

    tracker.log_usage_batch(records)


@st.cache_resource
def get_services():
    """Initialize and cache service instances."""
    tracker = CostTracker()
    generate_sample_data(tracker)
    router = SmartRouter()
    budget_mgr = BudgetManager(tracker)
    analytics = TokenAnalytics(tracker)

    # Set up sample budgets
    for dept in ["engineering", "marketing", "research", "support", "sales"]:
        budget_mgr.set_budget(dept, random.uniform(50, 200), BudgetPeriod.MONTHLY)

    return tracker, router, budget_mgr, analytics


def render_cost_overview(analytics: TokenAnalytics) -> None:
    """Render the Cost Overview page."""
    st.header("Cost Overview")

    stats = analytics.get_summary_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spend", f"${stats['total_cost']:.4f}")
    col2.metric("Total Requests", f"{stats['total_requests']:,}")
    col3.metric("Avg Cost/Request", f"${stats['avg_cost_per_request']:.6f}")
    col4.metric("Models Used", stats["models_used"])

    st.subheader("Cost Trend (Last 30 Days)")
    trends = analytics.cost_trends(days=30)
    if trends:
        df = pd.DataFrame(trends)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["daily_cost"],
                mode="lines+markers",
                name="Daily Cost",
                line={"color": "#636EFA"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["cumulative_cost"],
                mode="lines",
                name="Cumulative Cost",
                yaxis="y2",
                line={"color": "#EF553B", "dash": "dash"},
            )
        )
        fig.update_layout(
            yaxis={"title": "Daily Cost ($)"},
            yaxis2={
                "title": "Cumulative Cost ($)",
                "overlaying": "y",
                "side": "right",
            },
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_router_demo(router: SmartRouter) -> None:
    """Render the Router Demo page."""
    st.header("Router Demo")
    st.write("Enter text to see how the smart router selects a model.")

    input_text = st.text_area(
        "Input Text",
        value="Explain the trade-offs between microservices and monolithic architectures.",
        height=120,
    )

    col1, col2 = st.columns(2)
    max_cost = col1.number_input("Max Cost ($)", min_value=0.0, value=0.0, step=0.001)
    quality = col2.selectbox(
        "Required Quality",
        options=["Auto", "economy", "standard", "premium"],
    )

    if st.button("Route Request", type="primary"):
        kwargs = {}
        if max_cost > 0:
            kwargs["max_cost"] = max_cost
        if quality != "Auto":
            kwargs["required_quality"] = quality

        try:
            decision = router.route_text(input_text, **kwargs)

            st.success(f"Routed to: **{decision.selected_model.name}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Complexity", decision.complexity.value.title())
            col2.metric("Quality Tier", decision.quality_tier.value.title())
            col3.metric("Est. Cost", f"${decision.estimated_cost:.6f}")

            st.info(f"**Reason:** {decision.reason}")

            with st.expander("Model Details"):
                st.json(decision.selected_model.model_dump())

        except ValueError as e:
            st.error(f"Routing failed: {e}")


def render_department_breakdown(analytics: TokenAnalytics) -> None:
    """Render the Department Breakdown page."""
    st.header("Department Breakdown")

    dept_data = analytics.token_usage_by_department()
    if dept_data:
        df = pd.DataFrame(dept_data)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                df,
                values="cost",
                names="department",
                title="Cost by Department",
                hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                df,
                x="department",
                y="total_tokens",
                title="Token Usage by Department",
                color="department",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Department Details")
        st.dataframe(
            df[["department", "request_count", "cost", "total_tokens", "avg_cost_per_request"]],
            use_container_width=True,
        )

    # Model utilization
    st.subheader("Model Utilization")
    util = analytics.model_utilization_rates()
    if util:
        df_util = pd.DataFrame(util)
        fig = px.bar(
            df_util,
            x="model",
            y=["request_pct", "cost_pct"],
            barmode="group",
            title="Model Utilization (Request % vs Cost %)",
            labels={"value": "Percentage (%)", "variable": "Metric"},
        )
        st.plotly_chart(fig, use_container_width=True)


def render_budget_status(budget_mgr: BudgetManager) -> None:
    """Render the Budget Status page."""
    st.header("Budget Status")

    budgets = budget_mgr.check_all_budgets()
    if not budgets:
        st.info("No budgets configured.")
        return

    for budget in budgets:
        status = budget["status"]
        icon = {"ok": "green", "warning": "orange", "critical": "red", "exceeded": "red"}
        color = icon.get(status, "gray")

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"**{budget['entity_id'].title()}**")
        col2.metric("Budget", f"${budget['budget_limit']:.2f}")
        col3.metric("Spent", f"${budget['current_spend']:.4f}")
        col4.markdown(
            f":{color}[{status.upper()}] ({budget['usage_pct']:.1f}%)"
        )

        st.progress(min(budget["usage_pct"] / 100, 1.0))
        st.divider()

    # Alerts
    alerts = budget_mgr.generate_alerts()
    if alerts:
        st.subheader("Active Alerts")
        for alert in alerts:
            if alert.alert_type.value == "critical":
                st.error(alert.message)
            else:
                st.warning(alert.message)


def render_savings_report(analytics: TokenAnalytics) -> None:
    """Render the Savings Report page."""
    st.header("Savings Report")

    savings = analytics.savings_calculator()

    col1, col2, col3 = st.columns(3)
    col1.metric("Actual Cost", f"${savings['actual_cost']:.4f}")
    col2.metric("Without Routing", f"${savings['baseline_cost']:.4f}")
    col3.metric(
        "Savings",
        f"${savings['savings']:.4f}",
        delta=f"{savings['savings_pct']:.1f}%",
    )

    st.info(
        f"Compared against using **{savings['baseline_model']}** for all requests."
    )

    # Efficiency metrics
    st.subheader("Model Efficiency")
    metrics = analytics.efficiency_metrics()
    if metrics:
        df = pd.DataFrame(metrics)
        fig = px.bar(
            df,
            x="model",
            y="cost_per_output_token",
            title="Cost per Output Token by Model",
            color="model",
        )
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Main entry point for the Streamlit dashboard."""
    st.set_page_config(
        page_title="LLM Cost Optimizer",
        page_icon="$",
        layout="wide",
    )

    st.title("LLM Cost Optimizer Dashboard")
    st.caption("Smart request routing and cost management for LLM APIs")

    tracker, router, budget_mgr, analytics = get_services()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Cost Overview",
            "Router Demo",
            "Department Breakdown",
            "Budget Status",
            "Savings Report",
        ]
    )

    with tab1:
        render_cost_overview(analytics)

    with tab2:
        render_router_demo(router)

    with tab3:
        render_department_breakdown(analytics)

    with tab4:
        render_budget_status(budget_mgr)

    with tab5:
        render_savings_report(analytics)


if __name__ == "__main__":
    main()
