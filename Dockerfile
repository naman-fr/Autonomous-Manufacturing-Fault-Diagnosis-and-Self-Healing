FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY examples ./examples

RUN pip install --no-cache-dir -e ".[backend,app,mlops]"

EXPOSE 8000 8501

CMD ["uvicorn", "amfd.backend.api:app", "--host=0.0.0.0", "--port=8000"]
