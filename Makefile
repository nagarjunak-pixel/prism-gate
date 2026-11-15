.PHONY: install lint typecheck test drift benchmark demo ui-build serve check

UV ?= uv

install:
	$(UV) sync --python 3.12 --all-extras

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

typecheck:
	$(UV) run mypy src

test:
	$(UV) run pytest --cov=prism_gate --cov-fail-under=95 --cov-report=term-missing

drift:
	$(UV) run prism drift-check

benchmark:
	$(UV) run prism benchmark --requests 300 --concurrency 32

demo:
	$(UV) run prism demo

ui-build:
	cd ui && npm ci && npm run build

serve:
	$(UV) run prism serve --host 0.0.0.0 --port 8000

check: lint typecheck test drift benchmark ui-build
