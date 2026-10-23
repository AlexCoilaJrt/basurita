"""
Preparacion del dataset de segmentacion - NTP 900.058 (7 colores)

Lo usa notebooks/entrenamiento_seg_drive.ipynb:
  1. inventario()            -> lee las carpetas por color del Drive
  2. dividir_por_grupos()    -> train/valid/test por objeto (subcarpeta) y estratificado por color
  3. etiquetar_automatico()  -> YOLO-World ubica el objeto, SAM dibuja el contorno, la carpeta da la clase
  4. galeria_revision()      -> hojas de miniaturas para revisar las etiquetas
  5. convertir_taco()        -> TACO (COCO, contornos) a YOLO-seg, dividido por batch
  6. resumen() / escribir_yaml()
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

CLASES = [
    "azul_papel_carton",
    "amarillo_metales",
    "blanco_plasticos",
    "gris_vidrio",
    "marron_organicos",
    "rojo_peligrosos",
    "negro_generales",
]
SPLITS = ["train", "valid", "test"]
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}

# Textos para que YOLO-World ubique el objeto en la foto (la clase la da la carpeta)
PROMPTS_UBICAR = {
    "azul_papel_carton": ["cardboard box", "paper", "newspaper", "notebook", "paper bag", "carton"],
    "amarillo_metales": ["aluminum can", "tin can", "metal object", "aluminum foil", "bottle cap"],
    "blanco_plasticos": ["plastic bottle", "plastic bag", "plastic wrapper", "plastic container",
                         "plastic cup", "transparent plastic"],
    "gris_vidrio": ["glass bottle", "glass jar", "drinking glass", "broken glass"],
    "marron_organicos": ["fruit", "fruit peel", "food", "vegetable", "leaves", "eggshell"],
    "rojo_peligrosos": ["battery", "cell phone", "syringe", "pills", "face mask", "light bulb",
                        "electronic device", "cable", "spray can"],
    "negro_generales": ["cigarette butt", "tissue", "styrofoam", "snack bag", "wrapper", "trash"],
}
PROMPTS_GENERICOS = ["object", "trash", "item held in hand"]

# TACO: 60 categorias -> 7 colores (None = se descarta la anotacion)
TACO_MAPEO = {
    "Aluminium foil": "amarillo_metales",
    "Battery": "rojo_peligrosos",
    "Aluminium blister pack": "negro_generales",
    "Carded blister pack": "negro_generales",
    "Other plastic bottle": "blanco_plasticos",
    "Clear plastic bottle": "blanco_plasticos",
    "Glass bottle": "gris_vidrio",
    "Plastic bottle cap": "blanco_plasticos",
    "Metal bottle cap": "amarillo_metales",
    "Broken glass": "gris_vidrio",
    "Food Can": "amarillo_metales",
    "Aerosol": "rojo_peligrosos",
    "Drink can": "amarillo_metales",
    "Toilet tube": "azul_papel_carton",
    "Other carton": "azul_papel_carton",
    "Egg carton": "azul_papel_carton",
    "Drink carton": "azul_papel_carton",  # tetrapak: revisar segun el criterio local
    "Corrugated carton": "azul_papel_carton",
    "Meal carton": "azul_papel_carton",
    "Pizza box": "azul_papel_carton",
    "Paper cup": "negro_generales",  # papel plastificado
    "Disposable plastic cup": "blanco_plasticos",
    "Foam cup": "negro_generales",
    "Glass cup": "gris_vidrio",
    "Other plastic cup": "blanco_plasticos",
    "Food waste": "marron_organicos",
    "Glass jar": "gris_vidrio",
    "Plastic lid": "blanco_plasticos",
    "Metal lid": "amarillo_metales",
    "Other plastic": "blanco_plasticos",
    "Magazine paper": "azul_papel_carton",
    "Tissues": "negro_generales",
    "Wrapping paper": "azul_papel_carton",
    "Normal paper": "azul_papel_carton",
    "Paper bag": "azul_papel_carton",
    "Plastified paper bag": "negro_generales",
    "Plastic film": "blanco_plasticos",
    "Six pack rings": "blanco_plasticos",
    "Garbage bag": "negro_generales",
    "Other plastic wrapper": "blanco_plasticos",
    "Single-use carrier bag": "blanco_plasticos",
    "Polypropylene bag": "blanco_plasticos",
    "Crisp packet": "negro_generales",  # envoltura multicapa
    "Spread tub": "blanco_plasticos",
    "Tupperware": "blanco_plasticos",
    "Disposable food container": "blanco_plasticos",
    "Foam food container": "negro_generales",
    "Other plastic container": "blanco_plasticos",
    "Plastic glooves": "rojo_peligrosos",
    "Plastic utensils": "blanco_plasticos",
    "Pop tab": "amarillo_metales",
    "Rope & strings": "negro_generales",
    "Scrap metal": "amarillo_metales",
    "Shoe": "negro_generales",
    "Squeezable tube": "blanco_plasticos",
    "Plastic straw": "blanco_plasticos",
    "Paper straw": "azul_papel_carton",
    "Styrofoam piece": "negro_generales",
    "Unlabeled litter": "negro_generales",
    "Cigarette": "negro_generales",
}

# Colores BGR para la galeria (mismos que el detector)
COLORES = {
    "azul_papel_carton": (200, 100, 20),
    "amarillo_metales": (0, 215, 255),
    "blanco_plasticos": (245, 245, 245),
    "gris_vidrio": (150, 150, 150),
    "marron_organicos": (19, 69, 139),
    "rojo_peligrosos": (0, 0, 230),
    "negro_generales": (25, 25, 25),
}


def _limpiar(texto):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", texto).strip("_") or "x"


def normalizar_imagen(origen, destino, max_lado=1280):
    """Abre (incluido HEIC), corrige la rotacion EXIF, reduce y guarda como JPG. Devuelve (w, h)."""
    img = ImageOps.exif_transpose(Image.open(origen)).convert("RGB")
    img.thumbnail((max_lado, max_lado))
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    img.save(destino, "JPEG", quality=92)
    return img.size


# ---------------------------------------------------------------- 1. inventario
def inventario(dir_fotos):
    """Lista las fotos del Drive. Cada subcarpeta dentro de un color es un objeto (grupo)."""
    dir_fotos = Path(dir_fotos)
    items, avisos = [], []
    for carpeta in sorted(p for p in dir_fotos.iterdir() if p.is_dir()):
        if carpeta.name not in CLASES:
            avisos.append(f"Carpeta ignorada (no es una clase): {carpeta.name}")
    for clase in CLASES:
        dir_clase = dir_fotos / clase
        if not dir_clase.is_dir():
            avisos.append(f"Falta la carpeta: {clase}")
            continue
        for ruta in sorted(dir_clase.rglob("*")):
            if ruta.suffix.lower() not in EXTENSIONES or ruta.name.startswith("."):
                continue
            rel = ruta.relative_to(dir_clase)
            if len(rel.parts) > 1:
                grupo = f"{clase}/{rel.parts[0]}"
            else:  # foto suelta: es su propio grupo
                grupo = f"{clase}/{ruta.stem}"
                avisos.append(f"Foto sin subcarpeta de objeto: {clase}/{ruta.name}")
            nombre = _limpiar(f"{clase}__{'__'.join(rel.with_suffix('').parts)}")
            items.append({"ruta": str(ruta), "clase": clase, "grupo": grupo, "nombre": nombre})
    return items, avisos


# ---------------------------------------------------------------- 2. division
def dividir_por_grupos(items, proporcion=(0.7, 0.2, 0.1), semilla=42):
    """Asigna 'split' a cada item: todos los de un grupo van juntos, y se divide color por color."""
    rng = random.Random(semilla)
    avisos = []
    grupos_por_clase = defaultdict(list)
    for it in items:
        if it["grupo"] not in grupos_por_clase[it["clase"]]:
            grupos_por_clase[it["clase"]].append(it["grupo"])

    split_de_grupo = {}
    for clase, grupos in grupos_por_clase.items():
        grupos = sorted(grupos)
        rng.shuffle(grupos)
        n = len(grupos)
        if n >= 3:
            n_test = max(1, round(n * proporcion[2]))
            n_val = max(1, round(n * proporcion[1]))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train, n_val, n_test = 1, 1, n - 2
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
            avisos.append(f"{clase}: solo 2 objetos -> sin test para este color")
        else:
            n_train, n_val, n_test = n, 0, 0
            avisos.append(f"{clase}: solo {n} objeto(s) -> sin valid ni test para este color")
        for i, g in enumerate(grupos):
            split_de_grupo[g] = "train" if i < n_train else "valid" if i < n_train + n_val else "test"

    for it in items:
        it["split"] = split_de_grupo[it["grupo"]]
    return avisos


# ---------------------------------------------------------------- 3. etiquetado automatico
def _mascara_a_poligono(mascara):
    """Contorno mas grande de una mascara binaria, simplificado y normalizado a [0, 1]."""
    m = (mascara > 0.5).astype(np.uint8)
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    c = max(contornos, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(c) < 3:
        return None
    h, w = m.shape
    return (c / np.array([w, h], dtype=np.float32)).clip(0, 1)


def _caja_a_poligono(caja, w, h):
    x1, y1, x2, y2 = caja
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32) / [w, h]


def _escribir_etiqueta(ruta_lbl, id_clase, poligono):
    ruta_lbl.parent.mkdir(parents=True, exist_ok=True)
    ruta_lbl.write_text(f"{id_clase} " + " ".join(f"{v:.5f}" for v in poligono.ravel()) + "\n")


def etiquetar_automatico(items, dir_trabajo, modelo_world, modelo_sam, conf=0.05, max_lado=1280):
    """
    Por cada foto: YOLO-World encuentra el objeto mas probable, SAM dibuja su contorno
    y se guarda la etiqueta YOLO-seg con la clase de la carpeta.
    Devuelve una lista de registros con 'estado' para revisar.
    """
    dir_trabajo = Path(dir_trabajo)
    registros = []
    por_clase = defaultdict(list)
    for it in items:
        por_clase[it["clase"]].append(it)

    for clase, lista in por_clase.items():
        modelo_world.set_classes(PROMPTS_UBICAR[clase] + PROMPTS_GENERICOS)
        id_clase = CLASES.index(clase)
        for it in lista:
            ruta_img = dir_trabajo / it["split"] / "images" / f"{it['nombre']}.jpg"
            ruta_lbl = dir_trabajo / it["split"] / "labels" / f"{it['nombre']}.txt"
            reg = {**it, "imagen": str(ruta_img), "poligono": None}
            try:
                w, h = normalizar_imagen(it["ruta"], ruta_img, max_lado)
            except Exception as e:  # archivo danado o formato no soportado
                reg["estado"] = f"error_lectura: {e}"
                registros.append(reg)
                continue

            det = modelo_world.predict(str(ruta_img), conf=conf, verbose=False)[0]
            if len(det.boxes) == 0:
                # Respaldo: en estas fotos el objeto suele estar al centro -> SAM con un punto central
                seg = modelo_sam(str(ruta_img), points=[[w / 2, h / 2]], labels=[1], verbose=False)[0]
                poligono = None
                if seg.masks is not None and len(seg.masks.data):
                    poligono = _mascara_a_poligono(seg.masks.data[0].cpu().numpy())
                if poligono is None:
                    reg["estado"] = "sin_objeto"
                    ruta_img.unlink()
                    registros.append(reg)
                    continue
                reg["estado"] = "por_centro"
                _escribir_etiqueta(ruta_lbl, id_clase, poligono)
                reg["poligono"] = poligono
                registros.append(reg)
                continue
            mejor = int(det.boxes.conf.argmax())
            caja = det.boxes.xyxy[mejor].tolist()
            area = (caja[2] - caja[0]) * (caja[3] - caja[1]) / (w * h)

            seg = modelo_sam(str(ruta_img), bboxes=[caja], verbose=False)[0]
            poligono = None
            if seg.masks is not None and len(seg.masks.data):
                poligono = _mascara_a_poligono(seg.masks.data[0].cpu().numpy())
            if poligono is None:
                poligono = _caja_a_poligono(caja, w, h)
                reg["estado"] = "solo_caja"
            elif area < 0.02:
                reg["estado"] = "objeto_pequeno"
            elif len(det.boxes) > 1 and float(det.boxes.conf.sort(descending=True).values[1]) > 0.3:
                reg["estado"] = "varios_objetos"
            else:
                reg["estado"] = "ok"

            _escribir_etiqueta(ruta_lbl, id_clase, poligono)
            reg["poligono"] = poligono
            registros.append(reg)
    return registros


# ---------------------------------------------------------------- 4. galeria
def galeria_revision(registros, dir_salida, lado=220, columnas=6, filas=5):
    """Hojas JPG por color con el contorno pintado. Los casos dudosos llevan borde rojo y su estado."""
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    hojas = []
    por_clase = defaultdict(list)
    for r in registros:
        por_clase[r["clase"]].append(r)

    por_hoja = columnas * filas
    for clase, lista in por_clase.items():
        lista = sorted(lista, key=lambda r: r["estado"] == "ok")  # dudosos primero
        for n_hoja in range(0, len(lista), por_hoja):
            hoja = np.full((filas * (lado + 22), columnas * lado, 3), 255, np.uint8)
            for i, r in enumerate(lista[n_hoja:n_hoja + por_hoja]):
                miniatura = _miniatura(r, clase, lado)
                y, x = (i // columnas) * (lado + 22), (i % columnas) * lado
                hoja[y:y + lado, x:x + lado] = miniatura
                objeto = r["grupo"].split("/", 1)[1]
                texto = f"{r['estado'][:12]} {objeto[:12]}/{Path(r['ruta']).name[:12]}"
                color = (0, 120, 0) if r["estado"] == "ok" else (0, 0, 220)
                cv2.putText(hoja, texto, (x + 3, y + lado + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, color, 1, cv2.LINE_AA)
            ruta = dir_salida / f"{clase}_{n_hoja // por_hoja + 1:02d}.jpg"
            cv2.imwrite(str(ruta), hoja)
            hojas.append(str(ruta))
    return hojas


def _miniatura(r, clase, lado):
    fondo = np.full((lado, lado, 3), 200, np.uint8)
    img = cv2.imread(r["imagen"]) if Path(r["imagen"]).exists() else None
    if img is None:
        cv2.putText(fondo, "sin imagen", (10, lado // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    else:
        h, w = img.shape[:2]
        if r["poligono"] is not None:
            pts = (r["poligono"] * [w, h]).astype(np.int32)
            capa = img.copy()
            cv2.fillPoly(capa, [pts], COLORES[clase])
            img = cv2.addWeighted(capa, 0.45, img, 0.55, 0)
            cv2.polylines(img, [pts], True, (0, 255, 0), max(2, w // 200))
        esc = lado / max(h, w)
        img = cv2.resize(img, (int(w * esc), int(h * esc)))
        y0, x0 = (lado - img.shape[0]) // 2, (lado - img.shape[1]) // 2
        fondo[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    if r["estado"] != "ok":
        cv2.rectangle(fondo, (0, 0), (lado - 1, lado - 1), (0, 0, 220), 4)
    return fondo


# ---------------------------------------------------------------- 5. TACO
def convertir_taco(dir_taco, dir_trabajo, proporcion=(0.7, 0.2, 0.1), semilla=42, max_lado=1280):
    """
    TACO (annotations.json + data/batch_N/) a YOLO-seg con los 7 colores.
    Se divide por batch (fotos de una misma sesion van juntas) para no filtrar fotos parecidas.
    """
    dir_taco, dir_trabajo = Path(dir_taco), Path(dir_trabajo)
    coco = json.loads((dir_taco / "data" / "annotations.json").read_text())
    categorias = {c["id"]: TACO_MAPEO.get(c["name"]) for c in coco["categories"]}
    sin_mapeo = sorted({c["name"] for c in coco["categories"] if c["name"] not in TACO_MAPEO})

    anotaciones = defaultdict(list)
    for a in coco["annotations"]:
        if categorias.get(a["category_id"]) and a.get("segmentation"):
            anotaciones[a["image_id"]].append(a)

    batches = sorted({im["file_name"].split("/")[0] for im in coco["images"]})
    rng = random.Random(semilla)
    rng.shuffle(batches)
    n_val = max(1, round(len(batches) * proporcion[1]))
    n_test = max(1, round(len(batches) * proporcion[2]))
    split_batch = {b: "valid" if i < n_val else "test" if i < n_val + n_test else "train"
                   for i, b in enumerate(batches)}

    stats = Counter()
    for im in coco["images"]:
        anns = anotaciones.get(im["id"])
        origen = dir_taco / "data" / im["file_name"]
        if not anns:
            stats["sin_anotaciones_utiles"] += 1
            continue
        if not origen.exists():
            stats["foto_no_descargada"] += 1
            continue
        img = ImageOps.exif_transpose(Image.open(origen)).convert("RGB")
        if img.size != (im["width"], im["height"]):
            stats["rotacion_no_coincide"] += 1
            continue
        split = split_batch[im["file_name"].split("/")[0]]
        nombre = "taco__" + _limpiar(Path(im["file_name"]).with_suffix("").as_posix())
        img.thumbnail((max_lado, max_lado))
        ruta_img = dir_trabajo / split / "images" / f"{nombre}.jpg"
        ruta_img.parent.mkdir(parents=True, exist_ok=True)
        img.save(ruta_img, "JPEG", quality=92)

        lineas = []
        for a in anns:
            # Si el objeto tiene varias partes, se usa la de mayor area
            partes = [np.array(s, dtype=np.float32).reshape(-1, 2) for s in a["segmentation"] if len(s) >= 6]
            if not partes:
                continue
            p = max(partes, key=lambda q: cv2.contourArea(q))
            p = (p / [im["width"], im["height"]]).clip(0, 1)
            lineas.append(f"{CLASES.index(categorias[a['category_id']])} "
                          + " ".join(f"{v:.5f}" for v in p.ravel()))
        ruta_lbl = dir_trabajo / split / "labels" / f"{nombre}.txt"
        ruta_lbl.parent.mkdir(parents=True, exist_ok=True)
        ruta_lbl.write_text("\n".join(lineas) + "\n")
        stats[f"imagenes_{split}"] += 1
    return dict(stats), sin_mapeo


# ---------------------------------------------------------------- 6. resumen y yaml
def resumen(dir_trabajo):
    """Tabla de imagenes y objetos por color y split, con avisos si un color queda vacio."""
    dir_trabajo = Path(dir_trabajo)
    objetos = {s: Counter() for s in SPLITS}
    imagenes = {s: Counter() for s in SPLITS}
    for s in SPLITS:
        for lbl in (dir_trabajo / s / "labels").glob("*.txt"):
            clases = [CLASES[int(l.split()[0])] for l in lbl.read_text().splitlines() if l.strip()]
            objetos[s].update(clases)
            imagenes[s].update(set(clases))
    lineas = [f"{'clase':<20}" + "".join(f"{s:>16}" for s in SPLITS),
              f"{'':<20}" + "".join(f"{'img / objetos':>16}" for _ in SPLITS)]
    avisos = []
    for c in CLASES:
        lineas.append(f"{c:<20}" + "".join(f"{f'{imagenes[s][c]} / {objetos[s][c]}':>16}" for s in SPLITS))
        for s in ("valid", "test"):
            if imagenes[s][c] == 0:
                avisos.append(f"{c} no tiene imagenes en {s}: sus metricas en {s} no seran validas")
    return "\n".join(lineas), avisos


def escribir_yaml(dir_trabajo):
    dir_trabajo = Path(dir_trabajo)
    lineas = [f"path: {dir_trabajo}", "train: train/images", "val: valid/images"]
    if any((dir_trabajo / "test" / "images").glob("*.jpg")):
        lineas.append("test: test/images")
    lineas.append("names:")
    lineas += [f"  {i}: {c}" for i, c in enumerate(CLASES)]
    ruta = dir_trabajo / "data.yaml"
    ruta.write_text("\n".join(lineas) + "\n")
    return str(ruta)
