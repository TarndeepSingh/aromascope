# One container serves the API and the web client (single-service deploy).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libxrender1 libxext6 libsm6 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src ./src
COPY services ./services
COPY apps/web ./apps/web
COPY artifacts ./artifacts
ENV PYTHONPATH=/app/src:/app \
    AROMASCOPE_SERVE_WEB=/app/apps/web \
    AROMASCOPE_ARTIFACTS=/app/artifacts/releases \
    AROMASCOPE_TRUST_PROXY=1 \
    AROMASCOPE_ORIGINS="*" \
    PORT=7860
RUN useradd -m -u 1000 app && chown -R app /app
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/health/ready')"
CMD ["sh", "-c", "uvicorn services.api.app.main:app --host 0.0.0.0 --port ${PORT}"]
