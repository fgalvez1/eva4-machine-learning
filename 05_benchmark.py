import os
import csv
import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

MODEL_DIR = "modelo"
OUT_DIR = "evidencias"
N_LATENCIA = 200     # inferencias de 1 imagen para medir latencia
N_LOTES = 20         # lotes de 32 imágenes para medir rendimiento

VARIANTES = {
    "Keras FP32": os.path.join(MODEL_DIR, "modelo_autos.keras"),
    "ONNX FP32": os.path.join(MODEL_DIR, "modelo_autos_fp32.onnx"),
    "ONNX INT8": os.path.join(MODEL_DIR, "modelo_autos_int8.onnx"),
}

def medir(nombre):
    """Mide una variante en un proceso separado, para que la RAM de una no
    contamine la medición de otra."""
    import psutil
    proceso = psutil.Process()
    X = np.load(os.path.join(MODEL_DIR, "test_x.npy"))
    y = np.load(os.path.join(MODEL_DIR, "test_y.npy"))
    ram_inicial = proceso.memory_info().rss

    if nombre.startswith("Keras"):
        import tensorflow as tf
        import keras
        model = keras.models.load_model(VARIANTES[nombre])
        f = tf.function(lambda x: model(x, training=False),
                        input_signature=[tf.TensorSpec((None,) + X.shape[1:], tf.float32)])
        predecir = lambda x: f(tf.constant(x)).numpy()
    else:
        import onnxruntime as ort
        opciones = ort.SessionOptions()
        opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sesion = ort.InferenceSession(VARIANTES[nombre], opciones,
                                      providers=["CPUExecutionProvider"])
        entrada = sesion.get_inputs()[0].name
        predecir = lambda x: sesion.run(None, {entrada: np.ascontiguousarray(x)})[0]

    # Calentamiento (las primeras inferencias siempre son más lentas)
    for _ in range(10):
        predecir(X[:1])
    predecir(X[:32])

    # Accuracy sobre el conjunto de prueba
    probs = np.concatenate([predecir(X[i:i + 32]) for i in range(0, len(X), 32)])
    accuracy = float((probs.argmax(1) == y).mean())

    # Latencia: 1 imagen por petición (caso típico de una API en tiempo real)
    tiempos = []
    for i in range(N_LATENCIA):
        t = time.perf_counter()
        predecir(X[i % len(X)][None])
        tiempos.append((time.perf_counter() - t) * 1000)

    # Rendimiento: lotes de 32 imágenes (procesamiento por lotes)
    t = time.perf_counter()
    for _ in range(N_LOTES):
        predecir(X[:32])
    imgs_por_seg = 32 * N_LOTES / (time.perf_counter() - t)

    ram_mb = (proceso.memory_info().rss - ram_inicial) / 1e6
    return {
        "modelo": nombre,
        "tamano_mb": round(os.path.getsize(VARIANTES[nombre]) / 1e6, 2),
        "ram_mb": round(ram_mb, 1),
        "latencia_media_ms": round(float(np.mean(tiempos)), 2),
        "latencia_p95_ms": round(float(np.percentile(tiempos, 95)), 2),
        "imagenes_por_seg": round(imgs_por_seg, 1),
        "accuracy": round(accuracy, 4),
    }

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    resultados = []
    for nombre in VARIANTES:
        print(f"Midiendo {nombre}...")
        with ctx.Pool(1) as pool:
            resultados.append(pool.apply(medir, (nombre,)))

    # --- Tabla de resultados ---
    print(f"\n{'Modelo':<12}{'Tamaño MB':>11}{'RAM MB':>9}{'Lat. media':>12}"
          f"{'Lat. p95':>10}{'img/s':>9}{'Accuracy':>10}")
    for r in resultados:
        print(f"{r['modelo']:<12}{r['tamano_mb']:>11}{r['ram_mb']:>9}"
              f"{r['latencia_media_ms']:>10} ms{r['latencia_p95_ms']:>7} ms"
              f"{r['imagenes_por_seg']:>9}{r['accuracy']:>10.2%}")

    base = resultados[0]
    for r in resultados[1:]:
        print(f"\n{r['modelo']} vs Keras: "
              f"{base['latencia_media_ms'] / r['latencia_media_ms']:.1f}x más rápido | "
              f"{base['tamano_mb'] / r['tamano_mb']:.1f}x más liviano | "
              f"accuracy {100 * (r['accuracy'] - base['accuracy']):+.2f} puntos")

    with open(f"{OUT_DIR}/benchmark_optimizacion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=resultados[0].keys())
        w.writeheader()
        w.writerows(resultados)

    # --- Gráfico comparativo ---
    nombres = [r["modelo"] for r in resultados]
    colores = ["gray", "steelblue", "seagreen"]
    metricas = [("latencia_media_ms", "Latencia media (ms)\n↓ menor es mejor"),
                ("tamano_mb", "Tamaño del modelo (MB)\n↓ menor es mejor"),
                ("ram_mb", "RAM utilizada (MB)\n↓ menor es mejor"),
                ("accuracy", "Accuracy en prueba\n↑ mayor es mejor")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (clave, titulo) in zip(axes, metricas):
        valores = [r[clave] for r in resultados]
        barras = ax.bar(nombres, valores, color=colores)
        for b, v in zip(barras, valores):
            texto = f"{v:.1%}" if clave == "accuracy" else f"{v}"
            ax.text(b.get_x() + b.get_width() / 2, v, texto, ha="center", va="bottom")
        ax.set_title(titulo)
        ax.tick_params(axis="x", rotation=15)
        if clave == "accuracy":
            ax.set_ylim(min(valores) - 0.05, 1)
    plt.suptitle("Comparación de optimizaciones: Keras vs ONNX FP32 vs ONNX INT8", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/07_comparacion_optimizacion.png", dpi=150)
    plt.show()