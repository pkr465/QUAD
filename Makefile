.PHONY: serve test test-unit test-integration lint format clean install

install:
	pip install -e ".[dev]"
	pre-commit install

serve:
	python -m quad.server.main

test:
	pytest tests/ -v --cov=quad --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
