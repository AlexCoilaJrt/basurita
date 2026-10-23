"""
Detector de residuos en tiempo real con la webcam - NTP 900.058 (7 colores)
Uso:
    python src/detector_residuos.py                      (usa modelos/best.pt)
    python src/detector_residuos.py --modelo otro.pt
Teclas:
    q / ESC  -> salir
    s        -> guardar captura en capturas/ (captura_YYYYMMDD_HHMMSS.jpg)
"""
import argparse
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent
MODELO_DEFECTO = RAIZ / "modelos" / "best.pt"
DIR_CAPTURAS = RAIZ / "capturas"

# Colores en formato BGR (OpenCV)
COLORES = {
    "azul_papel_carton": (200, 100, 20),
    "amarillo_metales": (0, 215, 255),
    "blanco_plasticos": (245, 245, 245),
    "gris_vidrio": (150, 150, 150),
    "marron_organicos": (19, 69, 139),
    "rojo_peligrosos": (0, 0, 230),
    "negro_generales": (25, 25, 25),
}

ETIQUETAS = {
    "azul_papel_carton": "AZUL - Papel y carton",
    "amarillo_metales": "AMARILLO - Metales",
    "blanco_plasticos": "BLANCO - Plasticos",
    "gris_vidrio": "GRIS - Vidrio",
    "marron_organicos": "MARRON - Organicos",
    "rojo_peligrosos": "ROJO - Peligrosos",
    "negro_generales": "NEGRO - Generales",
}

# Colores claros llevan texto/borde oscuro; los oscuros, texto/borde blanco
CLAROS = {"amarillo_metales", "blanco_plasticos", "gris_vidrio"}


def contraste(clase):
    return (0, 0, 0) if clase in CLAROS else (255, 255, 255)


def dibujar_caja(frame, x1, y1, x2, y2, clase, conf):
    color = COLORES.get(clase, (0, 255, 0))
    borde = contraste(clase)
    # Doble borde para que se vea sobre cualquier fondo
    cv2.rectangle(frame, (x1, y1), (x2, y2), borde, 5)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    texto = f"{ETIQUETAS.get(clase, clase)} {conf:.0%}"
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(texto, fuente, 0.6, 2)
    y_txt = max(y1, th + 10)
    cv2.rectangle(frame, (x1, y_txt - th - 10), (x1 + tw + 10, y_txt), color, -1)
    cv2.rectangle(frame, (x1, y_txt - th - 10), (x1 + tw + 10, y_txt), borde, 1)
    cv2.putText(frame, texto, (x1 + 5, y_txt - 6), fuente, 0.6, borde, 2, cv2.LINE_AA)


def dibujar_leyenda(frame, conteo, fps):
    x, y = 10, 10
    alto = 24 * (len(COLORES) + 1) + 10
    panel = frame.copy()
    cv2.rectangle(panel, (x, y), (x + 270, y + alto), (40, 40, 40), -1)
    cv2.addWeighted(panel, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"NTP 900.058 | {fps:.1f} FPS", (x + 8, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (clase, color) in enumerate(COLORES.items()):
        yy = y + 24 * (i + 1) + 14
        cv2.rectangle(frame, (x + 8, yy - 12), (x + 24, yy + 2), color, -1)
        cv2.rectangle(frame, (x + 8, yy - 12), (x + 24, yy + 2), (255, 255, 255), 1)
        n = conteo.get(clase, 0)
        txt = ETIQUETAS[clase] + (f"  x{n}" if n else "")
        cv2.putText(frame, txt, (x + 32, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(MODELO_DEFECTO), help="ruta al modelo entrenado")
    ap.add_argument("--camara", type=int, default=0, help="indice de la camara")
    ap.add_argument("--conf", type=float, default=0.45, help="confianza minima")
    ap.add_argument("--imgsz", type=int, default=480, help="tamano de inferencia (416/480/640)")
    args = ap.parse_args()

    if not Path(args.modelo).exists():
        raise SystemExit(f"No se encontro el modelo: {args.modelo}\n"
                         "Entrena en Colab (notebooks/) y copia best.pt a modelos/.")
    modelo = YOLO(args.modelo)
    cam = cv2.VideoCapture(args.camara)
    if not cam.isOpened():
        raise SystemExit("No se pudo abrir la camara. Revisa el permiso de Camara para la Terminal.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps, t_prev = 0.0, time.time()
    print("Detector iniciado. 'q' para salir, 's' para guardar captura.")

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        r = modelo.predict(frame, imgsz=args.imgsz, conf=args.conf,
                           device="cpu", verbose=False)[0]
        conteo = Counter()
        for caja in r.boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            clase = r.names[int(caja.cls[0])]
            conf = float(caja.conf[0])
            conteo[clase] += 1
            dibujar_caja(frame, x1, y1, x2, y2, clase, conf)

        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        dibujar_leyenda(frame, conteo, fps)

        cv2.imshow("Detector de residuos - NTP 900.058", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("s"):
            DIR_CAPTURAS.mkdir(exist_ok=True)
            nombre = str(DIR_CAPTURAS / f"captura_{datetime.now():%Y%m%d_%H%M%S}.jpg")
            cv2.imwrite(nombre, frame)
            print("Guardada:", nombre)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
