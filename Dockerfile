# Imagen base liviana de Python
FROM python:3.12-slim

WORKDIR /app

# Evita archivos .pyc y muestra los logs al instante
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Primero las dependencias (se guardan en caché si no cambian)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Solo se copia lo necesario para producción: la API y el modelo INT8
COPY app.py .
COPY modelo/modelo_autos_int8.onnx modelo/metadata.json ./modelo/

EXPOSE 8000

# Docker revisa cada 30 s que la API siga respondiendo
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]