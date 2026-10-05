FROM python:3.12.14-slim-bookworm
WORKDIR /app
COPY requirements-tested-cpu.txt requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements-tested-cpu.txt -r requirements-service.txt
COPY data ./data
COPY ga ./ga
COPY models ./models
COPY training ./training
COPY utils ./utils
COPY serving ./serving
COPY serve.py ./
COPY make_example.py ./
COPY scripts ./scripts
RUN useradd --create-home --uid 1000 appuser
USER appuser
ENV NAS_ARTIFACT=/models/selected NAS_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD ["python", "scripts/container_health.py"]
CMD ["python", "serve.py", "--device", "cpu", "--host", "0.0.0.0", "--port", "8000"]
