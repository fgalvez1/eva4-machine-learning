import os
import json
import random
import shutil
import numpy as np
from PIL import Image

# Prepara lo que necesitan las pruebas automáticas de CI/CD (GitHub Actions):
#  1. tests/muestras/: 10 imágenes de PRUEBA por clase (el dataset completo no
#     se sube a GitHub, por eso se usa una muestra pequeña y fija).
#  2. modelo/referencia_drift.json: estadísticas de color de las imágenes de
#     ENTRENAMIENTO, para detectar si los datos nuevos cambian (data drift).

MODEL_DIR = "modelo"
DATA_DIR = "dataset"
DESTINO = os.path.join("tests", "muestras")
N_POR_CLASE = 10
N_REFERENCIA = 400
SEED = 42

with open(os.path.join(MODEL_DIR, "particion.json"), encoding="utf-8") as f:
    p = json.load(f)
clases = p["clases"]
IMG_SIZE = tuple(p["img_size"])

# --- 1. Muestras de prueba para CI ---
random.seed(SEED)
shutil.rmtree(DESTINO, ignore_errors=True)
for idx, clase in enumerate(clases):
    os.makedirs(os.path.join(DESTINO, clase), exist_ok=True)
    rutas = [r for r, y in zip(p["test_rutas"], p["test_etiquetas"]) if y == idx]
    for ruta in random.sample(rutas, N_POR_CLASE):
        shutil.copy(ruta, os.path.join(DESTINO, clase, os.path.basename(ruta)))
    print(f"{clase}: {N_POR_CLASE} imágenes copiadas a {DESTINO}/{clase}")

# --- 2. Referencia para detección de drift ---
# Para cada imagen se calcula el brillo promedio de cada canal (R, G, B).
# Se guarda la media y desviación de esos valores en el set de entrenamiento.
rutas_test = set(p["test_rutas"])
rutas_train = [os.path.join(DATA_DIR, c, a) for c in clases
               for a in sorted(os.listdir(os.path.join(DATA_DIR, c)))
               if os.path.join(DATA_DIR, c, a) not in rutas_test]
rutas_ref = random.sample(rutas_train, min(N_REFERENCIA, len(rutas_train)))

medias = []
for ruta in rutas_ref:
    img = Image.open(ruta).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    medias.append(np.asarray(img, dtype="float32").mean(axis=(0, 1)))
medias = np.array(medias)

referencia = {"n_imagenes": len(rutas_ref),
              "media_rgb": medias.mean(axis=0).round(3).tolist(),
              "desv_rgb": medias.std(axis=0).round(3).tolist()}
with open(os.path.join(MODEL_DIR, "referencia_drift.json"), "w", encoding="utf-8") as f:
    json.dump(referencia, f, indent=2)
print("\nReferencia de drift guardada en modelo/referencia_drift.json")
print(f"  Media RGB: {referencia['media_rgb']}")
print(f"  Desv. RGB: {referencia['desv_rgb']}")