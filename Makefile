.PHONY: test lint

test:
	pytest -q -m "not slow" --strict-markers --tb=short

lint:
	ruff check src tests --select E9,F63,F7,F82
