.PHONY: help install dev lint format typecheck test docker-build docker-up docker-down clean

# ── Variables ────────────────────────────────────────────────
IMAGE_NAME  := vision-inference-microservice
IMAGE_TAG   := latest

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Local Development ───────────────────────────────────────

install: ## Install dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

dev: ## Run development server with hot-reload
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

lint: ## Run linter (ruff)
	ruff check app/ tests/

format: ## Auto-format code (ruff)
	ruff format app/ tests/
	ruff check --fix app/ tests/

typecheck: ## Run static type checker (mypy)
	mypy app/

test: ## Run test suite
	pytest -v --tb=short

# ── Docker ───────────────────────────────────────────────────

docker-build: ## Build Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-up: ## Start service via Docker Compose
	docker compose up --build -d

docker-down: ## Stop service via Docker Compose
	docker compose down

docker-logs: ## Tail service logs
	docker compose logs -f api

# ── Housekeeping ─────────────────────────────────────────────

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/
