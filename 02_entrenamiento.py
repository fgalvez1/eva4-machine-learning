import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# 1. CONFIGURACIÓN E HIPERPARÁMETROS
SEED = 42
DATA_DIR = "dataset"
OUT_DIR = "evidencias"
MODEL_DIR = "modelo"
IMG_SIZE = (224, 224)     # tamaño nativo de MobileNetV2
BATCH_SIZE = 32
EPOCHS_FASE1 = 15         # feature extraction: solo se entrena la parte final
EPOCHS_FASE2 = 15         # fine-tuning: se ajustan las últimas capas de MobileNetV2
LR_FASE1 = 1e-3
LR_FASE2 = 1e-5           # tasa muy baja para no destruir lo aprendido en ImageNet
CAPAS_A_DESCONGELAR = 30

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
keras.utils.set_random_seed(SEED)

# 2. PARTICIÓN DEL DATASET (70% entrenamiento / 15% validación / 15% prueba)

clases = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))
rutas, etiquetas = [], []
for idx, c in enumerate(clases):
    carpeta = os.path.join(DATA_DIR, c)
    for archivo in sorted(os.listdir(carpeta)):
        rutas.append(os.path.join(carpeta, archivo))
        etiquetas.append(idx)

X_train, X_temp, y_train, y_temp = train_test_split(
    rutas, etiquetas, test_size=0.30, stratify=etiquetas, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

print(f"Clases: {clases}")
print(f"Entrenamiento: {len(X_train)} | Validación: {len(X_val)} | Prueba: {len(X_test)}")

# Se guarda la partición de prueba para evaluarla en 03_evaluacion.py
with open(os.path.join(MODEL_DIR, "particion.json"), "w", encoding="utf-8") as f:
    json.dump({"clases": clases, "img_size": list(IMG_SIZE),
               "test_rutas": X_test, "test_etiquetas": y_test}, f, indent=2)

def cargar_imagen(ruta, etiqueta):
    img = tf.io.read_file(ruta)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)  # acepta JPG y PNG
    img = tf.image.resize(img, IMG_SIZE)
    return img, etiqueta

# 3. DATA AUGMENTATION
# Solo volteo horizontal y rotación leve: un auto nunca aparece de cabeza.
# A diferencia de la Eva 3, se aplica en el pipeline de datos y NO dentro del
# modelo, para que el modelo guardado quede limpio para producción (Eva 4).
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),          # auto visto desde el lado izquierdo o derecho
    layers.RandomRotation(0.05),              # inclinación leve de la cámara (±18°)
    layers.RandomZoom(0.15),                  # foto más cerca o más lejos del auto
    layers.RandomTranslation(0.1, 0.1),       # auto no siempre centrado
    layers.RandomContrast(0.2),               # distintas condiciones de luz
    layers.RandomBrightness(0.2),             # día nublado vs. sol directo
], name="data_augmentation")

def crear_dataset(rutas, etiquetas, entrenamiento):
    ds = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))
    ds = ds.map(cargar_imagen, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if entrenamiento:
        ds = ds.shuffle(len(rutas), seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    if entrenamiento:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

ds_train = crear_dataset(X_train, y_train, entrenamiento=True)
ds_val = crear_dataset(X_val, y_val, entrenamiento=False)

# Evidencia: una imagen original y 7 versiones aumentadas
img_ejemplo, _ = cargar_imagen(X_train[0], y_train[0])
plt.figure(figsize=(12, 6))
for i in range(8):
    if i == 0:
        img = img_ejemplo
    else:
        img = data_augmentation(tf.expand_dims(img_ejemplo, 0), training=True)[0]
    plt.subplot(2, 4, i + 1)
    plt.imshow(np.clip(np.array(img), 0, 255).astype("uint8"))
    plt.title("Original" if i == 0 else f"Aumentada {i}")
    plt.axis("off")
plt.suptitle(f"Data Augmentation — clase: {clases[y_train[0]]}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_data_augmentation.png", dpi=150)
plt.show()

# 4. ARQUITECTURA: TRANSFER LEARNING CON MOBILENETV2
# Mejora propuesta en la conclusión de la Eva 3: partir de una red ya entrenada
# con millones de imágenes (ImageNet) en vez de una CNN desde cero.

base = keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                      include_top=False, weights="imagenet")
base.trainable = False    # fase 1: la base queda congelada

entradas = keras.Input(shape=IMG_SIZE + (3,), name="imagen")
x = layers.Rescaling(1.0 / 127.5, offset=-1, name="normalizacion")(entradas)  # 0-255 -> -1..1
x = base(x, training=False)               # BatchNorm de la base siempre en modo inferencia
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.3, name="dropout")(x)
salidas = layers.Dense(len(clases), activation="softmax", name="clasificacion")(x)

model = keras.Model(entradas, salidas, name="MobileNetV2_Autos")
model.summary()
with open(f"{OUT_DIR}/resumen_modelo.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda linea, **kw: f.write(linea + "\n"))

# 5. ENTRENAMIENTO
pesos = compute_class_weight("balanced", classes=np.unique(y_train), y=np.array(y_train))
class_weight = {i: float(p) for i, p in enumerate(pesos)}
print("Pesos por clase:", {clases[i]: round(p, 2) for i, p in class_weight.items()})

def crear_callbacks(append):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(f"{OUT_DIR}/historial_entrenamiento.csv", append=append),
    ]

inicio = time.time()

# --- Fase 1: Feature extraction ---
print("\n===== FASE 1: Feature extraction (MobileNetV2 congelada) =====")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_FASE1),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist1 = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_FASE1,
                  class_weight=class_weight, callbacks=crear_callbacks(append=False))
epocas_fase1 = len(hist1.history["loss"])

# --- Fase 2: Fine-tuning ---
print(f"\n===== FASE 2: Fine-tuning (últimas {CAPAS_A_DESCONGELAR} capas) =====")
base.trainable = True
for capa in base.layers[:-CAPAS_A_DESCONGELAR]:
    capa.trainable = False
for capa in base.layers:                  # BatchNormalization siempre congelada
    if isinstance(capa, layers.BatchNormalization):
        capa.trainable = False

model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_FASE2),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist2 = model.fit(ds_train, validation_data=ds_val,
                  epochs=epocas_fase1 + EPOCHS_FASE2, initial_epoch=epocas_fase1,
                  class_weight=class_weight, callbacks=crear_callbacks(append=True))

minutos = (time.time() - inicio) / 60
print(f"\nTiempo de entrenamiento: {minutos:.1f} minutos")

model.save(os.path.join(MODEL_DIR, "modelo_autos.keras"))
print("Modelo guardado en modelo/modelo_autos.keras")

# 6. CURVAS DE PÉRDIDA Y PRECISIÓN

h = {k: hist1.history[k] + hist2.history[k] for k in ["loss", "val_loss", "accuracy", "val_accuracy"]}
epocas = range(1, len(h["loss"]) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.plot(epocas, h["loss"], label="Entrenamiento")
ax1.plot(epocas, h["val_loss"], label="Validación")
ax1.axvline(epocas_fase1 + 0.5, color="gray", linestyle="--", label="Inicio fine-tuning")
ax1.set_title("Pérdida (loss)"); ax1.set_xlabel("Época"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(epocas, h["accuracy"], label="Entrenamiento")
ax2.plot(epocas, h["val_accuracy"], label="Validación")
ax2.axvline(epocas_fase1 + 0.5, color="gray", linestyle="--", label="Inicio fine-tuning")
ax2.set_title("Precisión (accuracy)"); ax2.set_xlabel("Época"); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_curvas_entrenamiento.png", dpi=150)
plt.show()

print(f"Épocas totales: {len(h['loss'])} (fase 1: {epocas_fase1})")
print(f"Mejor accuracy de validación: {max(h['val_accuracy']):.2%}")