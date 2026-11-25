"""
Pruebas automáticas de la API (se ejecutan en GitHub Actions en cada push).
Ejecutar localmente:  pytest -v
"""
import io
import os
import json
import sys
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)
from app import app  # noqa: E402

MUESTRAS = os.path.join(RAIZ, "tests", "muestras")
LATENCIA_MAX_MS = 100        # umbral de latencia de inferencia (p95) por imagen
ACCURACY_MIN = 0.75          # umbral mínimo de accuracy sobre las muestras


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:          # "with" ejecuta la carga del modelo
        yield c


def imagenes_muestra():
    for clase in sorted(os.listdir(MUESTRAS)):
        for archivo in sorted(os.listdir(os.path.join(MUESTRAS, clase))):
            yield clase, os.path.join(MUESTRAS, clase, archivo)


def jpg_en_memoria(color=(120, 120, 120)):
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), color).save(buf, format="JPEG")
    return buf.getvalue()


# --- 1. Pruebas funcionales ---
def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"


def test_predict_formato_respuesta(cliente):
    r = cliente.post("/predict", files={"archivo": ("auto.jpg", jpg_en_memoria(), "image/jpeg")})
    assert r.status_code == 200
    datos = r.json()
    assert datos["etiqueta"] in ("Dañado", "Sin daño")
    assert 0 <= datos["confianza"] <= 1
    assert abs(sum(datos["probabilidades"].values()) - 1) < 0.01


def test_predict_lote(cliente):
    archivos = [("archivos", (f"auto{i}.jpg", jpg_en_memoria((40 * i, 80, 160)), "image/jpeg"))
                for i in range(4)]
    r = cliente.post("/predict/lote", files=archivos)
    assert r.status_code == 200
    assert len(r.json()) == 4


# --- 2. Manejo de errores ---
def test_rechaza_formato_no_soportado(cliente):
    r = cliente.post("/predict", files={"archivo": ("notas.txt", b"hola", "text/plain")})
    assert r.status_code == 415


def test_rechaza_imagen_corrupta(cliente):
    r = cliente.post("/predict", files={"archivo": ("rota.jpg", b"no soy una imagen", "image/jpeg")})
    assert r.status_code == 400


def test_rechaza_archivo_vacio(cliente):
    r = cliente.post("/predict", files={"archivo": ("vacia.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


# --- 3. Pruebas propias de Deep Learning ---
def test_accuracy_minima(cliente):
    """Evita desplegar un modelo que empeoró (regresión de métricas)."""
    aciertos, total = 0, 0
    for clase, ruta in imagenes_muestra():
        with open(ruta, "rb") as f:
            r = cliente.post("/predict", files={"archivo": (os.path.basename(ruta), f.read(), "image/jpeg")})
        aciertos += r.json()["clase"] == clase
        total += 1
    accuracy = aciertos / total
    print(f"\nAccuracy en muestras: {accuracy:.2%} ({aciertos}/{total})")
    assert accuracy >= ACCURACY_MIN


def test_latencia(cliente):
    """La inferencia debe mantenerse bajo el umbral definido (tiempo real)."""
    tiempos = []
    for _ in range(30):
        r = cliente.post("/predict", files={"archivo": ("auto.jpg", jpg_en_memoria(), "image/jpeg")})
        tiempos.append(r.json()["tiempo_inferencia_ms"])
    p95 = float(np.percentile(tiempos, 95))
    print(f"\nLatencia de inferencia p95: {p95:.2f} ms")
    assert p95 < LATENCIA_MAX_MS


def test_sin_data_drift():
    """Compara el color promedio de las muestras con el de entrenamiento.
    Si se aleja más de 3 errores estándar, los datos cambiaron (drift)."""
    with open(os.path.join(RAIZ, "modelo", "referencia_drift.json"), encoding="utf-8") as f:
        ref = json.load(f)
    medias = np.array([np.asarray(Image.open(r).convert("RGB").resize((224, 224)), dtype="float32")
                       .mean(axis=(0, 1)) for _, r in imagenes_muestra()])
    media_muestras = medias.mean(axis=0)
    error_estandar = np.array(ref["desv_rgb"]) / np.sqrt(len(medias))
    z = np.abs(media_muestras - np.array(ref["media_rgb"])) / error_estandar
    print(f"\nDesviación por canal (z): R={z[0]:.2f} G={z[1]:.2f} B={z[2]:.2f}")
    assert (z < 3).all(), "Posible data drift: el color de las imágenes cambió"