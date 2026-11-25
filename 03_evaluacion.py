import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from sklearn.metrics import classification_report, confusion_matrix

OUT_DIR = "evidencias"
MODEL_DIR = "modelo"

with open(os.path.join(MODEL_DIR, "particion.json"), encoding="utf-8") as f:
    p = json.load(f)
clases = p["clases"]
IMG_SIZE = tuple(p["img_size"])
rutas_test = p["test_rutas"]
y_true = np.array(p["test_etiquetas"])
nombres_es = {"danado": "Dañado", "sin_dano": "Sin daño"}
etiquetas = [nombres_es.get(c, c) for c in clases]

model = keras.models.load_model(os.path.join(MODEL_DIR, "modelo_autos.keras"))

def cargar_imagen(ruta, etiqueta):
    img = tf.io.read_file(ruta)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return img, etiqueta

ds_test = (tf.data.Dataset.from_tensor_slices((rutas_test, y_true))
           .map(cargar_imagen).batch(32))

# --- Métricas globales ---
loss, acc = model.evaluate(ds_test, verbose=0)
probs = model.predict(ds_test, verbose=0)
y_pred = probs.argmax(axis=1)
print(f"Imágenes de prueba: {len(y_true)}")
print(f"Loss en prueba: {loss:.4f}")
print(f"Accuracy en prueba: {acc:.2%}\n")

# --- Reporte por clase (precision, recall, f1-score) ---
reporte = classification_report(y_true, y_pred, target_names=etiquetas, digits=3)
print(reporte)
with open(f"{OUT_DIR}/reporte_clasificacion.txt", "w", encoding="utf-8") as f:
    f.write(f"Loss en prueba: {loss:.4f}\nAccuracy en prueba: {acc:.2%}\n\n{reporte}")

# --- Matriz de confusión ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=etiquetas, yticklabels=etiquetas,
            annot_kws={"size": 14})
plt.xlabel("Clase predicha")
plt.ylabel("Clase real")
plt.title(f"Matriz de confusión (conjunto de prueba) — Accuracy: {acc:.1%}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_matriz_confusion.png", dpi=150)
plt.show()

# --- Ejemplos de predicciones (verde = correcta, rojo = incorrecta) ---
indices = np.random.default_rng(42).choice(len(y_true), 12, replace=False)
plt.figure(figsize=(13, 10))
for k, i in enumerate(indices):
    img, _ = cargar_imagen(rutas_test[i], 0)
    correcta = y_pred[i] == y_true[i]
    plt.subplot(3, 4, k + 1)
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(f"Real: {etiquetas[y_true[i]]}\nPredicción: {etiquetas[y_pred[i]]} ({probs[i].max():.0%})",
              color="green" if correcta else "red", fontsize=10)
    plt.axis("off")
plt.suptitle("Ejemplos de predicciones en el conjunto de prueba")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_ejemplos_predicciones.png", dpi=150)
plt.show()