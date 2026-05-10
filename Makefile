.PHONY: install test lint typecheck app docker

install:
	pip install -e ".[dev,app]"

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy src

app:
	streamlit run app/streamlit_app.py

docker:
	docker compose up --build

