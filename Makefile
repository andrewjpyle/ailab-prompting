.DEFAULT_GOAL := help
.PHONY: help install lint format test eval demo red compare record hooks scan \
        lint-docs check-learning-numbers docker-build docker-run clean

IMAGE  ?= ailab-prompting:local
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
RUN_A  ?= results/run_zero_shot.json
RUN_B  ?= results/run_few_shot.json

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-24s %s\n", $$1, $$2}'

install: ## Create .venv from uv.lock (with dev tools)
	uv sync --locked

lint: ## Ruff lint + format check + mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format: ## Auto-fix lint issues and format
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run the test suite with coverage (fails under 90%)
	uv run pytest --cov --cov-report=term-missing

eval: ## Run the eval and enforce the regression gate (replays the REAL cassette)
	uv run ailab-eval

demo: ## Show each strategy's prompt on the stub, then run the eval (stub)
	uv run ailab-demo

red: ## Proof-of-gate: each sabotage must exit non-zero and name the failing metric
	@bash scripts/prove_gate.sh

compare: ## Diff two saved runs that change ONE variable (RUN_A vs RUN_B) with a CI
	uv run python -m ailab_prompting.compare $(RUN_A) $(RUN_B)

record: ## Record the extraction cassette from a live model (needs OLLAMA_HOST; opt-in)
	uv run python scripts/record_cassette.py

hooks: ## Enable the repo's git hooks (required once per clone)
	git config core.hooksPath .githooks
	@echo "hooks enabled: .githooks/pre-push will run gitleaks + denylist scan"

scan: ## Secret scan: gitleaks over full history + denylist scan
	gitleaks git --no-banner --redact --config .gitleaks.toml .
	scripts/denylist_scan.sh

lint-docs: ## Fail if any em dash (U+2014) appears in docs/ or *.md
	bash scripts/no_emdash.sh

check-learning-numbers: ## Assert tagged docs/LEARNING.md figures match the committed results
	uv run python scripts/check_learning_numbers.py

docker-build: ## Build the runtime image
	docker build --build-arg GIT_COMMIT=$(COMMIT) -t $(IMAGE) .

docker-run: ## Run the image (demo + eval)
	docker run --rm $(IMAGE)

clean: ## Remove caches and generated results
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
