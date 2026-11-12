"""
Evalua un modelo .pt con el MISMO protocolo y la MISMA formula de nota que uso el docente.
No entrena nada: solo mide.

Estructura esperada del examen (exportado de Roboflow en formato "folder"):
    examen/garbage.v1i.folder/test/{cardboard,glass,metal,paper,plastic}/
    examen/Garbage Classification.v4i.folder/test/{Anorganik,Organik,Residu}/
    examen/Smart medical waste segregation.v3i.folder/test/{General,Infectious}/

Uso:
    python evaluar_rubrica.py --modelo modelos/clasificador.pt --examen examen
    python evaluar_rubrica.py --modelo modelos/clasificador_v1.pt --examen examen   (para comparar)
"""
import argparse
import os
import tempfile
import time

import numpy as np
from PIL import Image
from ultralytics import YOLO

EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
TODOS = {"Azul", "Amarillo", "Blanco", "Gris", "Marron", "Rojo", "Negro"}

# clase del dataset -> (dataset, grupo del docente, colores aceptados, cuenta para macro-F1)
GRUPOS = {
    "cardboard":  ("garbage", "Azul (papel/carton)", {"Azul"}, True),
    "paper":      ("garbage", "Azul (papel/carton)", {"Azul"}, True),
    "metal":      ("garbage", "Amarillo (metal)", {"Amarillo"}, True),
    "plastic":    ("garbage", "Blanco (plastico)", {"Blanco"}, True),
    "glass":      ("garbage", "Gris (vidrio)", {"Gris"}, True),
    "organik":    ("v4", "Marron (Organik, grises)", {"Marron"}, True),
    "anorganik":  ("v4", "Inorg. reciclable (Anorganik)", {"Amarillo", "Azul", "Blanco", "Gris"}, False),
    "residu":     ("v4", "Negro/Rojo (Residu)", {"Negro", "Rojo"}, False),
    "infectious": ("medico", "Rojo (biocontaminado)", {"Rojo"}, True),
    "general":    ("medico", "No-Rojo (medico general)", TODOS - {"Rojo"}, False),
}

RANKING_DOCENTE = [("15 Pastor", 16.0), ("01 Jhoel", 14.0), ("21 Jhosef", 13.6), ("08 Fernandez", 13.2),
                   ("11 Jorge", 12.9), ("02 Cabana", 12.5), ("03 Fran", 12.3), ("13 Kana", 12.0),
                   ("18 Montes", 11.9), ("17 Ivan", 11.7), ("16 Ciro", 11.2)]


def color_de(nombre_clase):
    """'marron_organicos', 'Marron', 'marrón' -> 'Marron'."""
    n = nombre_clase.lower().replace("ó", "o")
    for c in TODOS:
        if n.startswith(c.lower()):
            return c
    raise SystemExit(f"No se a que color corresponde la clase del modelo: {nombre_clase}")


def cargar_examen(raiz):
    filas = []
    for carpeta, _, archivos in os.walk(raiz):
        partes = carpeta.replace("\\", "/").split("/")
        clase = os.path.basename(carpeta).lower()
        if "test" not in partes or clase not in GRUPOS:
            continue
        for f in archivos:
            if f.lower().endswith(EXT):
                filas.append((os.path.join(carpeta, f), clase))
    if not filas:
        raise SystemExit(f"No encontre carpetas test/<clase> dentro de {raiz}")
    return filas


def predecir(modelo, rutas, colores, lote=64):
    preds = []
    for i in range(0, len(rutas), lote):
        for r in modelo.predict(rutas[i:i + lote], imgsz=224, verbose=False):
            preds.append(colores[int(r.probs.top1)])
    return preds


def f1_macro(reales, preds):
    f1s = []
    for c in sorted(set(reales)):
        tp = sum(r == c and p == c for r, p in zip(reales, preds))
        fp = sum(r != c and p == c for r, p in zip(reales, preds))
        fn = sum(r == c and p != c for r, p in zip(reales, preds))
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="modelos/clasificador.pt")
    ap.add_argument("--examen", default="examen")
    args = ap.parse_args()

    modelo = YOLO(args.modelo)
    colores = {i: color_de(n) for i, n in modelo.names.items()}
    n_colores = len(set(colores.values()))
    filas = cargar_examen(args.examen)
    rutas = [r for r, _ in filas]
    clases = [c for _, c in filas]
    print(f"Modelo: {args.modelo}  |  clases: {list(modelo.names.values())}")
    print(f"Examen: {len(rutas)} imagenes (el docente uso 1455)\n")

    t0 = time.time()
    preds = predecir(modelo, rutas, colores)
    ms = (time.time() - t0) / len(rutas) * 1000
    ok = [p in GRUPOS[c][2] for p, c in zip(preds, clases)]

    # 1. exactitud balanceada: promedio del acierto de los 9 grupos
    por_grupo = {}
    for c, o in zip(clases, ok):
        por_grupo.setdefault(GRUPOS[c][1], []).append(o)
    acc_grupo = {g: float(np.mean(v)) for g, v in por_grupo.items()}
    bal = float(np.mean(list(acc_grupo.values())))

    # 2. macro-F1 en las clases estrictas
    est = [(next(iter(GRUPOS[c][2])), p) for c, p in zip(clases, preds) if GRUPOS[c][3]]
    mf1 = f1_macro([r for r, _ in est], [p for _, p in est])

    # 3. Rojo (solo dentro del dataset medico) y organicos
    med = [(c, p) for c, p in zip(clases, preds) if GRUPOS[c][0] == "medico"]
    inf = [p for c, p in med if c == "infectious"]
    rec = np.mean([p == "Rojo" for p in inf]) if inf else 0.0
    pr = [c for c, p in med if p == "Rojo"]
    prec = np.mean([c == "infectious" for c in pr]) if pr else 0.0
    f1r = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    org = acc_grupo.get("Marron (Organik, grises)", 0.0)

    # 4. caida en grises: garbage.v1i a color vs el mismo en grises
    idx_v1i = [i for i, c in enumerate(clases) if GRUPOS[c][0] == "garbage"]
    caida = 0.0
    if idx_v1i:
        with tempfile.TemporaryDirectory() as tmp:
            rut_g = []
            for k, i in enumerate(idx_v1i):
                d = os.path.join(tmp, f"{k}.jpg")
                Image.open(rutas[i]).convert("L").convert("RGB").save(d)
                rut_g.append(d)
            pg = predecir(modelo, rut_g, colores)
        acc_c = np.mean([ok[i] for i in idx_v1i])
        acc_g = np.mean([p in GRUPOS[clases[i]][2] for p, i in zip(pg, idx_v1i)])
        caida = (acc_c - acc_g) * 100

    pts = {
        "Exactitud balanceada (8)": 8 * bal,
        "Macro-F1 (4)": 4 * mf1,
        "Cobertura 7 colores (3)": 3 * n_colores / 7,
        "Robustez (3)": 3 - 0.05 * caida - 0.015 * 0.0,
        "Seguridad: organicos + Rojo (2)": 3 * org + 28 / 9 * f1r,
    }
    nota = sum(pts.values())

    print("Acierto por grupo del docente:")
    for g, a in sorted(acc_grupo.items(), key=lambda x: x[1]):
        print(f"  {g:<34}{a:7.1%}  {'#' * int(a * 30)}")
    print(f"\nExactitud balanceada {bal:.1%} | macro-F1 {mf1:.3f} | organicos {org:.1%}")
    print(f"Rojo: recall {rec:.1%}  precision {prec:.1%}  F1 {f1r:.3f} | caida en grises {caida:.1f} pp | {ms:.1f} ms/img")
    print("\nPuntos:")
    for k, v in pts.items():
        print(f"  {k:<34}{v:6.2f}")
    print(f"  {'NOTA ESTIMADA':<34}{nota:6.1f} / 20\n")

    print("Ranking del docente con tu modelo:")
    tabla = sorted(RANKING_DOCENTE + [("TU MODELO", round(nota, 1))], key=lambda x: -x[1])
    for i, (n, v) in enumerate(tabla, 1):
        print(f"  {i:>2}. {n:<12}{v:5.1f}" + ("   <--" if n == "TU MODELO" else ""))


if __name__ == "__main__":
    main()
