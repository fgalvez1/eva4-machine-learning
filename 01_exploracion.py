import os
import random
import shutil
import hashlib
import matplotlib.pyplot as plt
from PIL import Image

KAGGLE_DIR = "kaggle_autos"   # carpeta descomprimida a mano desde Kaggle
DATA_DIR = "dataset"
OUT_DIR = "evidencias"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Organización del dataset "Car Damage Detection" (Kaggle) ---
# El dataset original viene separado en training/validation, con las carpetas
# 00-damage (auto dañado) y 01-whole (auto sin daño). Se juntan en una carpeta
# por clase, porque la partición propia 70/15/15 se hace en 02_entrenamiento.py
if not os.path.exists(DATA_DIR):
    nombres = {"00-damage": "danado", "01-whole": "sin_dano"}
    for raiz, _, archivos in os.walk(KAGGLE_DIR):
        carpeta = os.path.basename(raiz)
        if carpeta in nombres:
            destino = os.path.join(DATA_DIR, nombres[carpeta])
            os.makedirs(destino, exist_ok=True)
            split = os.path.basename(os.path.dirname(raiz))   # training / validation
            for archivo in archivos:
                shutil.copy(os.path.join(raiz, archivo), os.path.join(destino, f"{split}_{archivo}"))
    print("Dataset organizado en la carpeta", DATA_DIR)

clases = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))

# --- Depuración: detectar y eliminar imágenes corruptas y duplicadas ---
corruptas, duplicadas = 0, 0
vistas = {}
for c in clases:
    carpeta = os.path.join(DATA_DIR, c)
    for archivo in sorted(os.listdir(carpeta)):
        ruta = os.path.join(carpeta, archivo)
        try:
            with Image.open(ruta) as img:
                img.verify()
        except Exception:
            print("Imagen corrupta eliminada:", ruta)
            os.remove(ruta)
            corruptas += 1
            continue
        # Dos archivos con el mismo contenido tienen el mismo hash MD5
        with open(ruta, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h in vistas:
            print("Imagen duplicada eliminada:", ruta, "(igual a", vistas[h] + ")")
            os.remove(ruta)
            duplicadas += 1
        else:
            vistas[h] = ruta
print(f"Imágenes corruptas eliminadas: {corruptas}")
print(f"Imágenes duplicadas eliminadas: {duplicadas}")

# --- Conteo por clase ---
conteo = {c: len(os.listdir(os.path.join(DATA_DIR, c))) for c in clases}
print("\nClases:", clases)
for c, n in conteo.items():
    print(f"  {c}: {n} imágenes")
print("Total:", sum(conteo.values()))

# --- Tamaños y formato de color (a diferencia de TrashNet, aquí los tamaños varían) ---
anchos, altos, modos = [], [], {}
for c in clases:
    for archivo in os.listdir(os.path.join(DATA_DIR, c)):
        with Image.open(os.path.join(DATA_DIR, c, archivo)) as img:
            anchos.append(img.width)
            altos.append(img.height)
            modos[img.mode] = modos.get(img.mode, 0) + 1
print(f"Ancho: mín {min(anchos)} | promedio {sum(anchos) / len(anchos):.0f} | máx {max(anchos)} px")
print(f"Alto:  mín {min(altos)} | promedio {sum(altos) / len(altos):.0f} | máx {max(altos)} px")
print("Formatos de color:", modos)

# --- Gráfico de distribución ---
plt.figure(figsize=(6, 4))
plt.bar(conteo.keys(), conteo.values(), color=["firebrick", "seagreen"])
for i, n in enumerate(conteo.values()):
    plt.text(i, n + 5, str(n), ha="center")
plt.title("Cantidad de imágenes por clase")
plt.ylabel("Imágenes")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_distribucion_clases.png", dpi=150)
plt.show()

# --- Muestras de cada clase ---
random.seed(42)
fig, axes = plt.subplots(len(clases), 4, figsize=(10, 2.8 * len(clases)))
for i, c in enumerate(clases):
    archivos = random.sample(os.listdir(os.path.join(DATA_DIR, c)), 4)
    for j, a in enumerate(archivos):
        axes[i, j].imshow(Image.open(os.path.join(DATA_DIR, c, a)).convert("RGB"))
        axes[i, j].axis("off")
    axes[i, 0].set_title(c, loc="left", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_muestras_por_clase.png", dpi=150)
plt.show()