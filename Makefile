.PHONY: install dev test lint format clean run dashboard help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv pip install -e .

dev: ## Install development dependencies
	uv pip install -e ".[dev]"

test: ## Run tests with coverage
	uv run python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-quick: ## Run tests without coverage
	uv run python -m pytest tests/ -v --tb=short

lint: ## Run linter
	uv run ruff check src/ tests/

format: ## Format code
	uv run ruff format src/ tests/

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

dashboard: ## Launch Streamlit dashboard
	uv run streamlit run src/app.py

run: ## Run the demo
	uv run python -c "from src.router import SmartRouter; r = SmartRouter(); print(r.route_text('Hello world'))"
