import os
import json
import time
import random
import numpy as np
import tensorflow as tf
import keras
import tf2onnx
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantFormat, QuantType)
from onnxruntime.quantization.shape_inference import quant_pre_process

MODEL_DIR = "modelo"
DATA_DIR = "dataset"
SEED = 42
RUTA_KERAS = os.path.join(MODEL_DIR, "modelo_autos.keras")
RUTA_FP32 = os.path.join(MODEL_DIR, "modelo_autos_fp32.onnx")
RUTA_PRE = os.path.join(MODEL_DIR, "modelo_autos_pre.onnx")
RUTA_INT8 = os.path.join(MODEL_DIR, "modelo_autos_int8.onnx")
N_CALIBRACION = 200      # imágenes de entrenamiento usadas para calibrar INT8

with open(os.path.join(MODEL_DIR, "particion.json"), encoding="utf-8") as f:
    p = json.load(f)
clases = p["clases"]
IMG_SIZE = tuple(p["img_size"])
rutas_test = p["test_rutas"]
y_test = np.array(p["test_etiquetas"])

def cargar_imagen(ruta):
    img = tf.io.read_file(ruta)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    return tf.image.resize(img, IMG_SIZE).numpy()   # float32 en rango 0-255

model = keras.models.load_model(RUTA_KERAS)

# 1. EXPORTACIÓN A ONNX (FP32)
# ONNX es un formato estándar e independiente de TensorFlow: el modelo se
# ejecuta con ONNX Runtime, un motor de inferencia optimizado y más liviano.
print("===== 1. Exportando a ONNX FP32 =====")
spec = (tf.TensorSpec((None,) + IMG_SIZE + (3,), tf.float32, name="imagen"),)

@tf.function(input_signature=spec)
def inferencia(imagen):
    return model(imagen, training=False)

inicio = time.time()
tf2onnx.convert.from_function(inferencia, input_signature=spec, opset=17,
                              output_path=RUTA_FP32)
print(f"Modelo ONNX FP32 guardado en {RUTA_FP32} ({time.time() - inicio:.1f} s)")

sesion_fp32 = ort.InferenceSession(RUTA_FP32, providers=["CPUExecutionProvider"])
NOMBRE_ENTRADA = sesion_fp32.get_inputs()[0].name
NOMBRE_SALIDA = sesion_fp32.get_outputs()[0].name
print(f"Entrada: {NOMBRE_ENTRADA} | Salida: {NOMBRE_SALIDA}")

# 2. VALIDACIÓN DEL ARTEFACTO
# Se comprueba que el modelo ONNX entregue las mismas probabilidades que Keras
print("\n===== 2. Validando la conversión =====")
X_test = np.stack([cargar_imagen(r) for r in rutas_test]).astype("float32")
probs_keras = model.predict(X_test, batch_size=32, verbose=0)
probs_onnx = np.concatenate([sesion_fp32.run(None, {NOMBRE_ENTRADA: X_test[i:i + 32]})[0]
                             for i in range(0, len(X_test), 32)])
dif_max = np.abs(probs_keras - probs_onnx).max()
coinciden = (probs_keras.argmax(1) == probs_onnx.argmax(1)).mean()
print(f"Diferencia máxima entre probabilidades: {dif_max:.2e}")
print(f"Predicciones iguales Keras vs ONNX: {coinciden:.2%}")
assert dif_max < 1e-3, "La conversión a ONNX cambió los resultados"
print("Conversión validada correctamente")

# Se guarda el conjunto de prueba ya procesado para el benchmark (05)
np.save(os.path.join(MODEL_DIR, "test_x.npy"), X_test)
np.save(os.path.join(MODEL_DIR, "test_y.npy"), y_test)

# 3. CUANTIZACIÓN FP32 -> INT8
# Los pesos y activaciones pasan de números de 32 bits a enteros de 8 bits:
# el modelo pesa ~4 veces menos y la CPU calcula más rápido con enteros.
# Es cuantización ESTÁTICA: se calibra con imágenes reales de entrenamiento
# para calcular el rango de valores de cada capa.
print("\n===== 3. Cuantizando a INT8 =====")
quant_pre_process(RUTA_FP32, RUTA_PRE)    # optimiza el grafo antes de cuantizar

rutas_todas = [os.path.join(DATA_DIR, c, a) for c in clases
               for a in sorted(os.listdir(os.path.join(DATA_DIR, c)))]
rutas_no_test = sorted(set(rutas_todas) - set(rutas_test))   # nunca se calibra con prueba
random.seed(SEED)
rutas_calib = random.sample(rutas_no_test, min(N_CALIBRACION, len(rutas_no_test)))

class LectorCalibracion(CalibrationDataReader):
    def __init__(self, rutas, lote=16):
        self.lotes = (rutas[i:i + lote] for i in range(0, len(rutas), lote))
    def get_next(self):
        lote = next(self.lotes, None)
        if lote is None:
            return None
        return {NOMBRE_ENTRADA: np.stack([cargar_imagen(r) for r in lote]).astype("float32")}

inicio = time.time()
quantize_static(RUTA_PRE, RUTA_INT8, LectorCalibracion(rutas_calib),
                quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
os.remove(RUTA_PRE)
print(f"Modelo ONNX INT8 guardado en {RUTA_INT8} ({time.time() - inicio:.1f} s, "
      f"{len(rutas_calib)} imágenes de calibración)")

# 4. METADATOS PARA LA API
with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump({"clases": clases, "img_size": list(IMG_SIZE),
               "entrada": NOMBRE_ENTRADA, "salida": NOMBRE_SALIDA}, f, indent=2)

print("\n===== Tamaño de los modelos =====")
for ruta in [RUTA_KERAS, RUTA_FP32, RUTA_INT8]:
    print(f"  {os.path.basename(ruta):28s} {os.path.getsize(ruta) / 1e6:6.2f} MB")