"""
Clasificador de residuos con la webcam - NTP 900.058 (7 colores) - modelo YOLO11n-cls
Coloca el objeto dentro del recuadro central; el recuadro toma el color del tacho.
Al iniciar, deja el recuadro VACIO 2 segundos: se guarda el fondo para saber cuando no hay objeto
(un clasificador siempre elige un color, incluso con una pared vacia).

Uso (desde la carpeta del proyecto, con el entorno activado):
    python clasificador_webcam_yolo.py --modelo modelos/clasificador.pt
Teclas:
    q / ESC -> salir
    s       -> guardar captura en capturas/
    f       -> volver a tomar el fondo (con el recuadro vacio)
"""
import argparse
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# Colores BGR (OpenCV) y textos; las claves son los nombres de carpeta del dataset
INFO = {
    "azul": ((200, 100, 20), "AZUL - Papel y carton"),
    "amarillo": ((0, 215, 255), "AMARILLO - Metales"),
    "blanco": ((245, 245, 245), "BLANCO - Plasticos"),
    "gris": ((150, 150, 150), "GRIS - Vidrio"),
    "marron": ((19, 69, 139), "MARRON - Organicos"),
    "rojo": ((0, 0, 230), "ROJO - Peligrosos"),
    "negro": ((25, 25, 25), "NEGRO - Generales"),
}
CLAROS = {"amarillo", "blanco", "gris"}


def clave(nombre):
    """Convierte 'Marron', 'marron_organicos', 'Marrón' -> 'marron'."""
    n = nombre.lower().replace("ó", "o")
    for k in INFO:
        if n.startswith(k):
            return k
    return None


def texto(frame, txt, org, escala, color, grosor=1):
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="modelos/clasificador.pt")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--umbral", type=float, default=0.55, help="confianza minima para decidir")
    ap.add_argument("--suavizado", type=int, default=8, help="frames promediados")
    ap.add_argument("--recuadro", type=float, default=0.6, help="tamano del recuadro (0-1)")
    ap.add_argument("--umbral_fondo", type=float, default=14, help="diferencia minima con el fondo para decir que hay objeto")
    args = ap.parse_args()

    modelo = YOLO(args.modelo)
    nombres = [modelo.names[i] for i in sorted(modelo.names)]
    claves = [clave(n) for n in nombres]
    if None in claves:
        raise SystemExit(f"Clases no reconocidas en el modelo: {nombres}")
    print("Clases del modelo:", nombres)

    cam = cv2.VideoCapture(args.camara)
    if not cam.isOpened():
        raise SystemExit("No se pudo abrir la camara. Revisa el permiso de Camara para la Terminal.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    historial = deque(maxlen=args.suavizado)
    fondo, fondo_frames, fondo_inicio = None, [], time.time()
    fps, t_prev = 0.0, time.time()
    os.makedirs("capturas", exist_ok=True)
    print("Deja el recuadro VACIO 2 segundos (se guarda el fondo). Luego pon el objeto. 'q' salir, 's' captura, 'f' fondo.")

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # efecto espejo
        h, w = frame.shape[:2]
        lado = int(min(h, w) * args.recuadro)
        x1, y1 = (w - lado) // 2, (h - lado) // 2
        x2, y2 = x1 + lado, y1 + lado

        recorte = frame[y1:y2, x1:x2]
        mini = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY), (64, 64)), (5, 5), 0).astype(np.float32)

        # 1) Tomar el fondo (recuadro vacio) durante los primeros 2 segundos o al presionar 'f'
        if fondo is None:
            fondo_frames.append(mini)
            if time.time() - fondo_inicio >= 2:
                fondo = np.median(fondo_frames, axis=0)
                print("Fondo guardado. Ya puedes poner objetos.")
        hay_objeto = fondo is not None and float(np.abs(mini - fondo).mean()) >= args.umbral_fondo

        # 2) Clasificar solo si hay un objeto distinto al fondo
        if hay_objeto:
            r = modelo.predict(recorte, imgsz=224, device="cpu", verbose=False)[0]
            historial.append(r.probs.data.cpu().numpy())
        else:
            historial.clear()
        prom = np.mean(historial, axis=0) if historial else np.zeros(len(claves))
        idx = int(prom.argmax())
        k, conf = claves[idx], float(prom[idx])
        decidido = hay_objeto and conf >= args.umbral

        color = INFO[k][0] if decidido else (200, 200, 200)
        borde = (0, 0, 0) if (decidido and k in CLAROS) else (255, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), borde, 8)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)

        if fondo is None:
            etiqueta = "Deja el recuadro VACIO... guardando fondo"
        elif not hay_objeto:
            etiqueta = "Sin objeto - pon uno en el recuadro"
        elif not decidido:
            etiqueta = "No estoy seguro - acerca mas el objeto"
        else:
            etiqueta = f"{INFO[k][1]}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x1, y1 - th - 22), (x1 + tw + 20, y1), color, -1)
        texto(frame, etiqueta, (x1 + 10, y1 - 11), 0.9, borde if decidido else (0, 0, 0), 2)

        # Panel: probabilidad de cada color, ordenado de mayor a menor
        panel = frame.copy()
        cv2.rectangle(panel, (10, 10), (300, 40 + 26 * len(claves)), (40, 40, 40), -1)
        cv2.addWeighted(panel, 0.65, frame, 0.35, 0, frame)
        t_now = time.time()
        fps = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        texto(frame, f"NTP 900.058 | {fps:.1f} FPS", (20, 32), 0.5, (255, 255, 255))
        for rango, i in enumerate(np.argsort(-prom)):
            fila = 50 + 26 * rango
            ki = claves[i]
            cv2.rectangle(frame, (20, fila), (20 + int(150 * prom[i]), fila + 14), INFO[ki][0], -1)
            cv2.rectangle(frame, (20, fila), (170, fila + 14), (255, 255, 255), 1)
            texto(frame, f"{nombres[i]} {prom[i]:.0%}", (180, fila + 12), 0.45, (255, 255, 255))

        cv2.imshow("Clasificador de residuos - NTP 900.058", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("s"):
            ruta = os.path.join("capturas", f"captura_{datetime.now():%Y%m%d_%H%M%S}.jpg")
            cv2.imwrite(ruta, frame)
            print("Guardada:", ruta)
        if tecla == ord("f"):
            fondo, fondo_frames, fondo_inicio = None, [], time.time()
            historial.clear()

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
