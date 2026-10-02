FROM python:3.12-slim
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
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["python", "serve.py", "--device", "cpu", "--host", "0.0.0.0", "--port", "8000"]
