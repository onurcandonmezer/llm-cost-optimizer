"""Cost tracking module with SQLite backend.

Provides persistent storage and querying for LLM usage records,
with aggregation capabilities for departments, projects, and models.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from src.models import CostSummary, UsageRecord

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "costs.db"


class CostTracker:
    """Tracks and queries LLM API costs using a SQLite database.

    Provides methods to log usage records and query cost data
    by department, project, model, and time period.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the cost tracker.

        Args:
            db_path: Path to SQLite database file. Uses ':memory:' for
                     in-memory database if None.
        """
        if db_path is None:
            self.db_path = ":memory:"
        else:
            self.db_path = str(db_path)
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the usage_records table if it does not exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                department TEXT NOT NULL DEFAULT 'default',
                project_id TEXT NOT NULL DEFAULT 'default',
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cost REAL NOT NULL DEFAULT 0.0,
                latency_ms REAL NOT NULL DEFAULT 0.0
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_department
            ON usage_records(department)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_timestamp
            ON usage_records(timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_model
            ON usage_records(model)
        """)
        self._conn.commit()

    def log_usage(self, record: UsageRecord) -> int:
        """Log a usage record to the database.

        Args:
            record: The usage record to store.

        Returns:
            The ID of the inserted record.
        """
        cursor = self._conn.execute(
            """
            INSERT INTO usage_records
                (timestamp, model, department, project_id, input_tokens,
                 output_tokens, cost, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.timestamp.isoformat(),
                record.model,
                record.department,
                record.project_id,
                record.input_tokens,
                record.output_tokens,
                record.cost,
                record.latency_ms,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def log_usage_batch(self, records: list[UsageRecord]) -> int:
        """Log multiple usage records in a single transaction.

        Args:
            records: List of usage records to store.

        Returns:
            Number of records inserted.
        """
        data = [
            (
                r.timestamp.isoformat(),
                r.model,
                r.department,
                r.project_id,
                r.input_tokens,
                r.output_tokens,
                r.cost,
                r.latency_ms,
            )
            for r in records
        ]
        self._conn.executemany(
            """
            INSERT INTO usage_records
                (timestamp, model, department, project_id, input_tokens,
                 output_tokens, cost, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            data,
        )
        self._conn.commit()
        return len(data)

    def _build_summary(self, rows: list[sqlite3.Row], entity_col: str) -> list[CostSummary]:
        """Build CostSummary objects from query results."""
        summaries = []
        for row in rows:
            avg_cost = row["total_cost"] / row["request_count"] if row["request_count"] > 0 else 0
            avg_latency = (
                row["total_latency"] / row["request_count"] if row["request_count"] > 0 else 0
            )
            summaries.append(
                CostSummary(
                    entity=row[entity_col],
                    total_cost=round(row["total_cost"], 6),
                    request_count=row["request_count"],
                    total_input_tokens=row["total_input"],
                    total_output_tokens=row["total_output"],
                    avg_cost_per_request=round(avg_cost, 6),
                    avg_latency_ms=round(avg_latency, 2),
                )
            )
        return summaries

    def get_costs_by_department(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[CostSummary]:
        """Get cost summaries grouped by department.

        Args:
            start_date: Optional start of date range.
            end_date: Optional end of date range.

        Returns:
            List of cost summaries per department.
        """
        query = """
            SELECT
                department,
                SUM(cost) as total_cost,
                COUNT(*) as request_count,
                SUM(input_tokens) as total_input,
                SUM(output_tokens) as total_output,
                SUM(latency_ms) as total_latency
            FROM usage_records
            WHERE 1=1
        """
        params: list = []
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        query += " GROUP BY department ORDER BY total_cost DESC"

        rows = self._conn.execute(query, params).fetchall()
        return self._build_summary(rows, "department")

    def get_costs_by_project(
        self,
        department: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[CostSummary]:
        """Get cost summaries grouped by project.

        Args:
            department: Optional department filter.
            start_date: Optional start of date range.
            end_date: Optional end of date range.

        Returns:
            List of cost summaries per project.
        """
        query = """
            SELECT
                project_id,
                SUM(cost) as total_cost,
                COUNT(*) as request_count,
                SUM(input_tokens) as total_input,
                SUM(output_tokens) as total_output,
                SUM(latency_ms) as total_latency
            FROM usage_records
            WHERE 1=1
        """
        params: list = []
        if department:
            query += " AND department = ?"
            params.append(department)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        query += " GROUP BY project_id ORDER BY total_cost DESC"

        rows = self._conn.execute(query, params).fetchall()
        return self._build_summary(rows, "project_id")

    def get_costs_by_model(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[CostSummary]:
        """Get cost summaries grouped by model.

        Args:
            start_date: Optional start of date range.
            end_date: Optional end of date range.

        Returns:
            List of cost summaries per model.
        """
        query = """
            SELECT
                model,
                SUM(cost) as total_cost,
                COUNT(*) as request_count,
                SUM(input_tokens) as total_input,
                SUM(output_tokens) as total_output,
                SUM(latency_ms) as total_latency
            FROM usage_records
            WHERE 1=1
        """
        params: list = []
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        query += " GROUP BY model ORDER BY total_cost DESC"

        rows = self._conn.execute(query, params).fetchall()
        return self._build_summary(rows, "model")

    def get_daily_costs(
        self,
        days: int = 30,
        department: str | None = None,
    ) -> list[dict]:
        """Get daily cost totals for the specified number of days.

        Args:
            days: Number of past days to include.
            department: Optional department filter.

        Returns:
            List of dicts with 'date', 'total_cost', 'request_count' keys.
        """
        start_date = datetime.now() - timedelta(days=days)
        query = """
            SELECT
                DATE(timestamp) as date,
                SUM(cost) as total_cost,
                COUNT(*) as request_count
            FROM usage_records
            WHERE timestamp >= ?
        """
        params: list = [start_date.isoformat()]
        if department:
            query += " AND department = ?"
            params.append(department)
        query += " GROUP BY DATE(timestamp) ORDER BY date"

        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "date": row["date"],
                "total_cost": round(row["total_cost"], 6),
                "request_count": row["request_count"],
            }
            for row in rows
        ]

    def total_cost(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> float:
        """Get the total cost across all records.

        Args:
            start_date: Optional start of date range.
            end_date: Optional end of date range.

        Returns:
            Total cost in USD.
        """
        query = "SELECT COALESCE(SUM(cost), 0) as total FROM usage_records WHERE 1=1"
        params: list = []
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        row = self._conn.execute(query, params).fetchone()
        return round(row["total"], 6) if row else 0.0

    def avg_cost_per_request(self) -> float:
        """Get the average cost per request across all records."""
        row = self._conn.execute(
            "SELECT COALESCE(AVG(cost), 0) as avg_cost FROM usage_records"
        ).fetchone()
        return round(row["avg_cost"], 6) if row else 0.0

    def top_spending_departments(self, limit: int = 5) -> list[CostSummary]:
        """Get the top spending departments.

        Args:
            limit: Maximum number of departments to return.

        Returns:
            List of cost summaries for top spending departments.
        """
        summaries = self.get_costs_by_department()
        return summaries[:limit]

    def get_department_spend(
        self,
        department: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> float:
        """Get total spend for a specific department.

        Args:
            department: Department name.
            start_date: Optional start of date range.
            end_date: Optional end of date range.

        Returns:
            Total spend in USD.
        """
        query = """
            SELECT COALESCE(SUM(cost), 0) as total
            FROM usage_records
            WHERE department = ?
        """
        params: list = [department]
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        row = self._conn.execute(query, params).fetchone()
        return round(row["total"], 6) if row else 0.0

    def get_record_count(self) -> int:
        """Get the total number of usage records."""
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM usage_records").fetchone()
        return row["cnt"] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        """Ensure the database connection is closed on cleanup."""
        import contextlib

        with contextlib.suppress(Exception):
            self._conn.close()
