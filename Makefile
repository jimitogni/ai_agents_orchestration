PYTHON ?= python3

.PHONY: install lint format type-check test run docker-up ingest chat

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .
	ruff check . --fix

type-check:
	mypy app

test:
	pytest

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build

ingest:
	curl -X POST http://localhost:8000/ingest

chat:
	curl -X POST http://localhost:8000/chat \
		-H "Content-Type: application/json" \
		-d '{"question": "How can I identify overfitting in a machine learning model?"}'
