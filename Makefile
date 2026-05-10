.PHONY: install test lint typecheck api app local docker eval

install:
	pip install -e ".[dev,backend,app,mlops]"

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy src

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn amfd.backend.api:app --host 0.0.0.0 --port 8000 --reload

local:
	PYTHONPATH=src python scripts/local_server.py

eval:
	PYTHONPATH=src python eval.py

docker:
	docker compose up --build
