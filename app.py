import os
import io
import json
import time
import asyncio
from contextlib import asynccontextmanager
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

# --- Configuración (se puede cambiar con variables de entorno, útil en Docker) ---
MODEL_DIR = os.getenv("MODEL_DIR", "modelo")
RUTA_MODELO = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "modelo_autos_int8.onnx"))
MAX_LOTE = int(os.getenv("MAX_LOTE", "16"))         # máximo de imágenes por lote dinámico
ESPERA_MS = float(os.getenv("ESPERA_MS", "5"))      # tiempo máximo para juntar un lote
MAX_MB = 10                                          # tamaño máximo por imagen
MAX_ARCHIVOS = 32                                    # máximo de imágenes en /predict/lote
TIPOS_PERMITIDOS = {"image/jpeg", "image/png", "image/webp"}
NOMBRES_ES = {"danado": "Dañado", "sin_dano": "Sin daño"}

with open(os.path.join(MODEL_DIR, "metadata.json"), encoding="utf-8") as f:
    META = json.load(f)
CLASES = META["clases"]
IMG_SIZE = tuple(META["img_size"])


# --- Batching dinámico ---
# Las peticiones que llegan casi al mismo tiempo se juntan en un solo lote y
# se procesan en una sola llamada al modelo: aprovecha mejor la CPU cuando hay
# muchos usuarios simultáneos (en el benchmark: 898 img/s en lote vs 1,3 ms por imagen).
class LoteDinamico:
    def __init__(self, sesion):
        self.sesion = sesion
        self.entrada = sesion.get_inputs()[0].name
        self.cola = asyncio.Queue()

    async def predecir(self, imagen):
        futuro = asyncio.get_running_loop().create_future()
        await self.cola.put((imagen, futuro))
        return await futuro

    async def procesar(self):
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.cola.get()]
            limite = loop.time() + ESPERA_MS / 1000
            while len(items) < MAX_LOTE:
                restante = limite - loop.time()
                if restante <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self.cola.get(), restante))
                except asyncio.TimeoutError:
                    break
            lote = np.stack([imagen for imagen, _ in items])
            try:
                inicio = time.perf_counter()
                probs = (await asyncio.to_thread(self.sesion.run, None, {self.entrada: lote}))[0]
                ms = (time.perf_counter() - inicio) * 1000
                for (_, futuro), p in zip(items, probs):
                    if not futuro.done():
                        futuro.set_result((p, ms, len(items)))
            except Exception as error:
                for _, futuro in items:
                    if not futuro.done():
                        futuro.set_exception(error)


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    # El modelo se carga UNA vez al iniciar la API, no en cada petición
    opciones = ort.SessionOptions()
    opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sesion = ort.InferenceSession(RUTA_MODELO, opciones, providers=["CPUExecutionProvider"])
    app.state.lote = LoteDinamico(sesion)
    tarea = asyncio.create_task(app.state.lote.procesar())
    print(f"Modelo cargado: {RUTA_MODELO}")
    yield
    tarea.cancel()


app = FastAPI(title="API de detección de daño en autos",
              description="Clasifica fotos de autos en 'Dañado' o 'Sin daño' "
                          "con MobileNetV2 optimizada (ONNX INT8).",
              version="1.0.0", lifespan=ciclo_de_vida)


class Prediccion(BaseModel):
    archivo: str
    clase: str
    etiqueta: str
    confianza: float
    probabilidades: dict[str, float]
    tiempo_inferencia_ms: float
    tamano_lote: int
    tiempo_total_ms: float


async def leer_imagen(archivo: UploadFile) -> np.ndarray:
    """Valida y preprocesa la imagen igual que en el entrenamiento."""
    if archivo.content_type not in TIPOS_PERMITIDOS:
        raise HTTPException(415, f"'{archivo.filename}': formato no soportado "
                                 f"({archivo.content_type}). Use JPG, PNG o WEBP.")
    datos = await archivo.read()
    if not datos:
        raise HTTPException(400, f"'{archivo.filename}': el archivo está vacío.")
    if len(datos) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"'{archivo.filename}': supera el máximo de {MAX_MB} MB.")
    try:
        img = Image.open(io.BytesIO(datos))
        img = ImageOps.exif_transpose(img).convert("RGB")   # corrige la rotación del celular
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"'{archivo.filename}': no es una imagen válida o está dañada.")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    return np.asarray(img, dtype="float32")                # rango 0-255, como en el entrenamiento


async def clasificar(archivo: UploadFile) -> Prediccion:
    inicio = time.perf_counter()
    imagen = await leer_imagen(archivo)
    probs, ms_inferencia, tamano = await app.state.lote.predecir(imagen)
    idx = int(np.argmax(probs))
    return Prediccion(
        archivo=archivo.filename or "sin_nombre",
        clase=CLASES[idx],
        etiqueta=NOMBRES_ES.get(CLASES[idx], CLASES[idx]),
        confianza=round(float(probs[idx]), 4),
        probabilidades={NOMBRES_ES.get(c, c): round(float(p), 4) for c, p in zip(CLASES, probs)},
        tiempo_inferencia_ms=round(ms_inferencia, 2),
        tamano_lote=tamano,
        tiempo_total_ms=round((time.perf_counter() - inicio) * 1000, 2),
    )


@app.get("/")
def inicio():
    return {"mensaje": "API de detección de daño en autos. Documentación en /docs"}


@app.get("/health")
def health():
    return {"estado": "ok", "modelo": os.path.basename(RUTA_MODELO),
            "clases": [NOMBRES_ES.get(c, c) for c in CLASES]}


@app.post("/predict", response_model=Prediccion)
async def predict(archivo: UploadFile = File(..., description="Foto de un auto (JPG, PNG o WEBP)")):
    return await clasificar(archivo)


@app.post("/predict/lote", response_model=list[Prediccion])
async def predict_lote(archivos: list[UploadFile] = File(..., description="Varias fotos de autos")):
    if len(archivos) > MAX_ARCHIVOS:
        raise HTTPException(400, f"Máximo {MAX_ARCHIVOS} imágenes por petición.")
    return await asyncio.gather(*(clasificar(a) for a in archivos))