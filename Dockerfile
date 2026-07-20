# CPU-only image for the FastAPI service; no GPU is required. The exact
# 6,606-by-6,606 similarity engine currently uses about 0.9 GiB at runtime.
FROM python:3.12-slim

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser db ./db
COPY --chown=appuser:appuser data ./data

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
