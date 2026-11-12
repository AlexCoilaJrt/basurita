"""
Reconstruye a COLOR el test de 'garbage.v1i' que uso el docente (238 fotos de TrashNet).

El docente subio TrashNet a Roboflow (proyecto 'garbage', version 1). Roboflow renombra
'cardboard101.jpg' como 'cardboard101_jpg.rf.<codigo>.jpg' y lo estira a 640x640.
Este script toma las MISMAS 238 fotos del TrashNet original (a color), les aplica el mismo
redimensionado y las guarda con el nombre exacto del examen. Si le das la carpeta en grises
del .rar del docente, comprueba foto por foto que sean las mismas (huella pHash).

Uso:
    python reconstruir_garbage_v1i.py --lista nombres_examen_docente.csv \
        --gris "resultados/_gris_garbage_v1i" --salida examen
    (si ya bajaste TrashNet, agrega:  --trashnet ruta/a/dataset-resized.zip)
"""
import argparse
import csv
import io
import os
import re
import urllib.request
import zipfile

from PIL import Image

URL_TRASHNET = "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"
DATASET = "garbage.v1i.folder"

ap = argparse.ArgumentParser()
ap.add_argument("--lista", default="nombres_examen_docente.csv")
ap.add_argument("--trashnet", default="dataset-resized.zip", help="zip de TrashNet (se descarga si no existe)")
ap.add_argument("--gris", default=None, help="carpeta _gris_garbage_v1i del .rar del docente (para verificar)")
ap.add_argument("--salida", default="examen")
ap.add_argument("--lado", type=int, default=640)
args = ap.parse_args()

if not os.path.exists(args.trashnet):
    print("Descargando TrashNet (~43 MB) desde Hugging Face...")
    urllib.request.urlretrieve(URL_TRASHNET, args.trashnet)

z = zipfile.ZipFile(args.trashnet)
originales = {os.path.basename(n): n for n in z.namelist() if n.lower().endswith(".jpg")}
print(f"TrashNet: {len(originales)} fotos en el zip")

objetivo = []
with open(args.lista, encoding="utf-8") as f:
    for fila in csv.DictReader(f):
        if fila["dataset"] == DATASET:
            objetivo.append((fila["clase_gt"], fila["archivo"]))
print(f"Examen garbage.v1i: {len(objetivo)} fotos a reconstruir")

huella = None
if args.gris:
    try:
        import imagehash
        huella = lambda im: imagehash.phash(im.convert("L"))
    except ImportError:
        print("(instala 'imagehash' para verificar contra las fotos en grises: pip install imagehash)")

hechas, faltan, iguales, distintas = 0, [], 0, []
for clase, nombre_examen in objetivo:
    m = re.match(r"(.+?)_(jpg|jpeg|png)\.rf\.", nombre_examen)
    original = f"{m.group(1)}.{m.group(2)}" if m else None
    if original not in originales:
        faltan.append(nombre_examen)
        continue
    im = Image.open(io.BytesIO(z.read(originales[original]))).convert("RGB")
    im = im.resize((args.lado, args.lado))
    destino = os.path.join(args.salida, DATASET, "test", clase)
    os.makedirs(destino, exist_ok=True)
    im.save(os.path.join(destino, nombre_examen), quality=95)
    hechas += 1
    if huella:
        ruta_gris = os.path.join(args.gris, clase, nombre_examen)
        try:
            d = huella(im) - huella(Image.open(ruta_gris))
            if d <= 8:
                iguales += 1
            else:
                distintas.append((nombre_examen, d))
        except Exception:
            pass

print(f"\nReconstruidas a color: {hechas} de {len(objetivo)}  ->  {os.path.join(args.salida, DATASET, 'test')}")
if faltan:
    print(f"No encontradas en TrashNet: {len(faltan)} (ej. {faltan[0]})")
if huella:
    total = iguales + len(distintas)
    print(f"Verificacion contra las fotos en grises del docente: {iguales} de {total} coinciden")
    if distintas:
        print(f"   {len(distintas)} no coinciden (ej. {distintas[0]}); puede que Roboflow las haya girado o recortado")
