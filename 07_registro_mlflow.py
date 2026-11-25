import os
import csv
import json
import mlflow

# Registra en MLflow el entrenamiento y las 3 versiones del modelo
# (Keras FP32, ONNX FP32, ONNX INT8) con sus métricas y artefactos.
# Después de ejecutarlo, ver los resultados con:
#   mlflow ui --backend-store-uri sqlite:///mlflow.db
# y abrir http://127.0.0.1:5000

OUT_DIR = "evidencias"
MODEL_DIR = "modelo"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("deteccion-dano-autos")

with open(os.path.join(MODEL_DIR, "metadata.json"), encoding="utf-8") as f:
    meta = json.load(f)

# --- 1. Run de entrenamiento: hiperparámetros + curva por época ---
with mlflow.start_run(run_name="entrenamiento_mobilenetv2"):
    mlflow.log_params({
        "arquitectura": "MobileNetV2 (transfer learning, ImageNet)",
        "img_size": "224x224", "batch_size": 32, "seed": 42,
        "lr_fase1": 1e-3, "lr_fase2": 1e-5, "capas_descongeladas": 30,
        "particion": "70/15/15 estratificada", "clases": ",".join(meta["clases"]),
    })
    with open(os.path.join(OUT_DIR, "historial_entrenamiento.csv"), encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            paso = int(fila["epoch"])
            for clave in ["accuracy", "loss", "val_accuracy", "val_loss"]:
                mlflow.log_metric(clave, float(fila[clave]), step=paso)
    for archivo in ["04_curvas_entrenamiento.png", "05_matriz_confusion.png",
                    "reporte_clasificacion.txt", "resumen_modelo.txt"]:
        ruta = os.path.join(OUT_DIR, archivo)
        if os.path.exists(ruta):
            mlflow.log_artifact(ruta)
    print("Run de entrenamiento registrado")

# --- 2. Un run por versión optimizada del modelo ---
archivos_modelo = {"Keras FP32": "modelo_autos.keras",
                   "ONNX FP32": "modelo_autos_fp32.onnx",
                   "ONNX INT8": "modelo_autos_int8.onnx"}
with open(os.path.join(OUT_DIR, "benchmark_optimizacion.csv"), encoding="utf-8") as f:
    for fila in csv.DictReader(f):
        nombre = fila["modelo"]
        with mlflow.start_run(run_name=nombre.replace(" ", "_").lower()):
            formato, precision = nombre.split()
            mlflow.log_params({"formato": formato, "precision": precision,
                               "runtime": "TensorFlow" if formato == "Keras" else "ONNX Runtime"})
            mlflow.log_metrics({k: float(v) for k, v in fila.items() if k != "modelo"})
            mlflow.set_tag("produccion", "si" if nombre == "ONNX INT8" else "no")
            ruta = os.path.join(MODEL_DIR, archivos_modelo[nombre])
            if os.path.exists(ruta) and not ruta.endswith(".keras"):
                mlflow.log_artifact(ruta, artifact_path="modelo")
        print(f"Run registrado: {nombre}")

print("\nListo. Ejecuta:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("y abre http://127.0.0.1:5000")