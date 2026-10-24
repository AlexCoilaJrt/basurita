"""
Detector de residuos en tiempo real con la webcam - NTP 900.058 (7 colores)
Uso:
    python src/detector_residuos.py                      (usa modelos/best.pt o, si no existe, YOLO-World)
    Con YOLO-World, si existe modelos/clasificador.pt, el clasificador decide el color de cada objeto.
    python src/detector_residuos.py --modelo otro.pt
Teclas:
    q / ESC  -> salir
    s        -> guardar captura en capturas/ (captura_YYYYMMDD_HHMMSS.jpg)
    c        -> con --recolectar Blanco: guarda la foto limpia en datos/recolectadas/Blanco/
                (para sumarla a "data basurita/Blanco" y volver a entrenar)
"""
import argparse
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO, YOLOWorld

RAIZ = Path(__file__).resolve().parent.parent
MODELO_DEFECTO = RAIZ / "modelos" / "best.pt"
MODELO_WORLD = RAIZ / "modelos" / "yolov8s-worldv2.pt"
MODELO_CLASIFICADOR = RAIZ / "modelos" / "clasificador.pt"
MAX_RECORTES = 5  # objetos por cuadro que revisa el clasificador (CPU)
CONF_CENTRO = 0.7  # plan B: confianza minima cuando el clasificador mira el centro
DIR_CAPTURAS = RAIZ / "capturas"
DIR_RECOLECTADAS = RAIZ / "datos" / "recolectadas"
CARPETAS_DRIVE = ("Azul", "Amarillo", "Blanco", "Gris", "Marron", "Rojo", "Negro")

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

# Modelo generico (yolo11n.pt, 80 clases COCO): objeto -> (nombre, color NTP).
# Es una aproximacion: no distingue material (botella de vidrio sale como plastico).
# Con best.pt entrenado no se usa, porque el modelo ya devuelve las 7 clases.
MAPEO_COCO = {
    "bottle": ("Botella", "blanco_plasticos"),
    "cup": ("Vaso", "blanco_plasticos"),
    "toothbrush": ("Cepillo de dientes", "blanco_plasticos"),
    "wine glass": ("Copa", "gris_vidrio"),
    "vase": ("Florero", "gris_vidrio"),
    "fork": ("Tenedor", "amarillo_metales"),
    "knife": ("Cuchillo", "amarillo_metales"),
    "spoon": ("Cuchara", "amarillo_metales"),
    "scissors": ("Tijera", "amarillo_metales"),
    "book": ("Libro / papel", "azul_papel_carton"),
    "banana": ("Platano", "marron_organicos"),
    "apple": ("Manzana", "marron_organicos"),
    "orange": ("Naranja", "marron_organicos"),
    "broccoli": ("Brocoli", "marron_organicos"),
    "carrot": ("Zanahoria", "marron_organicos"),
    "sandwich": ("Sandwich", "marron_organicos"),
    "hot dog": ("Hot dog", "marron_organicos"),
    "pizza": ("Pizza", "marron_organicos"),
    "donut": ("Dona", "marron_organicos"),
    "cake": ("Torta", "marron_organicos"),
    "cell phone": ("Celular", "rojo_peligrosos"),
    "laptop": ("Laptop", "rojo_peligrosos"),
    "mouse": ("Mouse", "rojo_peligrosos"),
    "remote": ("Control remoto", "rojo_peligrosos"),
    "keyboard": ("Teclado", "rojo_peligrosos"),
    "hair drier": ("Secadora", "rojo_peligrosos"),
    "handbag": ("Bolso", "negro_generales"),
    "backpack": ("Mochila", "negro_generales"),
    "umbrella": ("Paraguas", "negro_generales"),
    "teddy bear": ("Peluche", "negro_generales"),
}


# YOLO-World (sin entrenar): detecta lo que se le describe en texto (en ingles).
# Agrega o quita lineas para ajustar que residuos busca.
MAPEO_WORLD = {
    # Plasticos
    "plastic bottle": ("Botella plastica", "blanco_plasticos"),
    "plastic cup": ("Vaso plastico", "blanco_plasticos"),
    "plastic bag": ("Bolsa plastica", "blanco_plasticos"),
    "plastic container": ("Envase plastico", "blanco_plasticos"),
    "plastic straw": ("Sorbete", "blanco_plasticos"),
    "plastic bottle cap": ("Tapa plastica", "blanco_plasticos"),
    "plastic wrapper": ("Envoltura plastica", "blanco_plasticos"),
    "plastic packaging": ("Empaque plastico", "blanco_plasticos"),
    "transparent plastic": ("Plastico transparente", "blanco_plasticos"),
    "plastic food container": ("Taper / envase de comida", "blanco_plasticos"),
    "plastic tray": ("Bandeja plastica", "blanco_plasticos"),
    "plastic lid": ("Tapa plastica", "blanco_plasticos"),
    "yogurt cup": ("Envase de yogurt", "blanco_plasticos"),
    "shampoo bottle": ("Frasco de shampoo", "blanco_plasticos"),
    "plastic sheet": ("Lamina plastica", "blanco_plasticos"),
    "plastic film": ("Film plastico", "blanco_plasticos"),
    "clear plastic sheet": ("Lamina transparente", "blanco_plasticos"),
    "colored plastic sheet": ("Lamina plastica de color", "blanco_plasticos"),
    "plastic folder": ("Folder / mica plastica", "blanco_plasticos"),
    # Vidrio
    "glass bottle": ("Botella de vidrio", "gris_vidrio"),
    "glass jar": ("Frasco de vidrio", "gris_vidrio"),
    "drinking glass": ("Vaso de vidrio", "gris_vidrio"),
    # Metales
    "aluminum can": ("Lata de aluminio", "amarillo_metales"),
    "tin can": ("Lata de conserva", "amarillo_metales"),
    "aluminum foil": ("Papel aluminio", "amarillo_metales"),
    "metal spoon": ("Cubierto de metal", "amarillo_metales"),
    # Papel y carton
    "cardboard box": ("Caja de carton", "azul_papel_carton"),
    "printed paper document": ("Hoja de papel", "azul_papel_carton"),
    "newspaper": ("Periodico", "azul_papel_carton"),
    "notebook": ("Cuaderno", "azul_papel_carton"),
    "paper bag": ("Bolsa de papel", "azul_papel_carton"),
    "egg carton": ("Caja de huevos", "azul_papel_carton"),
    # Organicos
    "banana": ("Platano", "marron_organicos"),
    "banana peel": ("Cascara de platano", "marron_organicos"),
    "apple": ("Manzana", "marron_organicos"),
    "orange": ("Naranja", "marron_organicos"),
    "fruit peel": ("Cascara de fruta", "marron_organicos"),
    "food leftovers": ("Restos de comida", "marron_organicos"),
    "vegetable": ("Verdura", "marron_organicos"),
    "eggshell": ("Cascara de huevo", "marron_organicos"),
    "dry leaves": ("Hojas secas", "marron_organicos"),
    # Peligrosos
    "battery": ("Pila / bateria", "rojo_peligrosos"),
    "cell phone": ("Celular", "rojo_peligrosos"),
    "syringe": ("Jeringa", "rojo_peligrosos"),
    "medicine pills": ("Medicamentos", "rojo_peligrosos"),
    "face mask": ("Mascarilla", "rojo_peligrosos"),
    "light bulb": ("Foco", "rojo_peligrosos"),
    "aerosol spray can": ("Aerosol", "rojo_peligrosos"),
    "computer mouse": ("Mouse", "rojo_peligrosos"),
    "charger cable": ("Cargador / cable", "rojo_peligrosos"),
    # Generales (no aprovechables)
    "cigarette butt": ("Colilla", "negro_generales"),
    "used tissue": ("Papel higienico / servilleta", "negro_generales"),
    "styrofoam container": ("Tecnopor", "negro_generales"),
    "chip bag": ("Envoltura de snack", "negro_generales"),
    "candy wrapper": ("Envoltura de golosina", "negro_generales"),
    "disposable paper cup": ("Vaso descartable de papel", "negro_generales"),
}


def clasificar(objeto):
    """Devuelve (nombre a mostrar, clase NTP) o None si no es un residuo."""
    if objeto in COLORES:  # modelo entrenado: ya es una de las 7 clases
        return ETIQUETAS[objeto].split(" - ")[1], objeto
    for clase in COLORES:  # clasificador entrenado con carpetas "Azul", "Rojo", ...
        if clase.split("_")[0] == objeto.lower():
            return ETIQUETAS[clase].split(" - ")[1], clase
    return MAPEO_WORLD.get(objeto) or MAPEO_COCO.get(objeto)


def reclasificar(frame, detecciones, clasificador, conf_clas):
    """El clasificador decide el color mirando el recorte de cada objeto."""
    h, w = frame.shape[:2]
    validas, recortes = [], []
    for d in detecciones:
        x1, y1, x2, y2 = d[:4]
        mx, my = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)  # margen del 10%
        recorte = frame[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
        if recorte.size:
            validas.append(d)
            recortes.append(recorte)
    if not recortes:
        return []

    salida = []
    for d, r in zip(validas, clasificador.predict(recortes, imgsz=224, device="cpu", verbose=False)):
        resultado = clasificar(r.names[r.probs.top1])
        conf = float(r.probs.top1conf)
        if resultado is None or conf < conf_clas:
            continue
        d[4], d[5], d[6] = resultado[0], resultado[1], conf
        salida.append(d)
    return salida


def clasificar_centro(frame, clasificador, conf_min):
    """Plan B: si no se ubico ningun objeto, el clasificador mira el centro de la imagen."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = w // 4, h // 6, w * 3 // 4, h * 5 // 6
    r = clasificador.predict(frame[y1:y2, x1:x2], imgsz=224, device="cpu", verbose=False)[0]
    resultado = clasificar(r.names[r.probs.top1])
    conf = float(r.probs.top1conf)
    if resultado is None or conf < conf_min:
        return []
    return [(x1, y1, x2, y2, f"{resultado[0]} (centro)", resultado[1], conf, None)]


def detectar(frame, modelo, conf_min, imgsz, clasificador=None, conf_clas=0.4):
    """Devuelve [(x1, y1, x2, y2, nombre, clase, conf, contorno)] de los residuos del cuadro."""
    r = modelo.predict(frame, imgsz=imgsz, conf=conf_min, device="cpu", verbose=False,
                       agnostic_nms=clasificador is not None)[0]
    contornos = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
    detecciones = []
    for caja, contorno in zip(r.boxes, contornos):
        resultado = clasificar(r.names[int(caja.cls[0])])
        if resultado is None:  # personas, sillas, etc. no son residuos
            continue
        x1, y1, x2, y2 = map(int, caja.xyxy[0])
        detecciones.append([x1, y1, x2, y2, *resultado, float(caja.conf[0]), contorno])
    if clasificador is not None:
        detecciones = reclasificar(frame, detecciones[:MAX_RECORTES], clasificador, conf_clas)
        if not detecciones:
            return clasificar_centro(frame, clasificador, CONF_CENTRO)
    return [tuple(d) for d in detecciones]


def contraste(clase):
    return (0, 0, 0) if clase in CLAROS else (255, 255, 255)


def dibujar_caja(frame, x1, y1, x2, y2, nombre, clase, conf):
    color = COLORES.get(clase, (0, 255, 0))
    borde = contraste(clase)
    # Doble borde para que se vea sobre cualquier fondo
    cv2.rectangle(frame, (x1, y1), (x2, y2), borde, 5)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    texto = f"{nombre} -> {ETIQUETAS[clase].split(' - ')[0]} {conf:.0%}"
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(texto, fuente, 0.6, 2)
    y_txt = max(y1, th + 10)
    cv2.rectangle(frame, (x1, y_txt - th - 10), (x1 + tw + 10, y_txt), color, -1)
    cv2.rectangle(frame, (x1, y_txt - th - 10), (x1 + tw + 10, y_txt), borde, 1)
    cv2.putText(frame, texto, (x1 + 5, y_txt - 6), fuente, 0.6, borde, 2, cv2.LINE_AA)


def dibujar_mascaras(frame, detecciones):
    """Rellena el contorno de cada residuo con el color de su tacho (modelos -seg)."""
    con_contorno = [(clase, c) for *_, clase, _, c in detecciones if c is not None and len(c) >= 3]
    if not con_contorno:
        return
    capa = frame.copy()
    for clase, contorno in con_contorno:
        cv2.fillPoly(capa, [contorno.astype("int32")], COLORES[clase])
    cv2.addWeighted(capa, 0.45, frame, 0.55, 0, frame)
    for clase, contorno in con_contorno:
        cv2.polylines(frame, [contorno.astype("int32")], True, contraste(clase), 2, cv2.LINE_AA)


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
        if n:  # resalta los colores detectados en este cuadro
            cv2.putText(frame, ">", (x - 2, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, txt, (x + 32, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2 if n else 1, cv2.LINE_AA)


def dibujar_veredicto(frame, principal):
    """Barra inferior grande con el tacho donde va el objeto mas seguro."""
    h, w = frame.shape[:2]
    alto = 90
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    if principal is None:
        panel = frame.copy()
        cv2.rectangle(panel, (0, h - alto), (w, h), (40, 40, 40), -1)
        cv2.addWeighted(panel, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "Acerca un residuo a la camara...", (20, h - 35),
                    fuente, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return

    nombre, clase, conf = principal
    color, texto_color = COLORES[clase], contraste(clase)
    tacho, categoria = ETIQUETAS[clase].split(" - ")
    cv2.rectangle(frame, (0, h - alto), (w, h), color, -1)
    cv2.rectangle(frame, (0, h - alto), (w, h), texto_color, 2)
    cv2.putText(frame, f"TACHO {tacho}", (20, h - 45), fuente, 1.4,
                texto_color, 3, cv2.LINE_AA)
    cv2.putText(frame, f"{nombre}  |  {categoria}  |  {conf:.0%}", (20, h - 14),
                fuente, 0.8, texto_color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=None,
                    help="ruta al modelo (por defecto modelos/best.pt o, si no existe, YOLO-World)")
    ap.add_argument("--camara", type=int, default=0, help="indice de la camara")
    ap.add_argument("--conf", type=float, default=None,
                    help="confianza minima (0.45; 0.15 con YOLO-World)")
    ap.add_argument("--imgsz", type=int, default=480, help="tamano de inferencia (416/480/640)")
    ap.add_argument("--clasificador", default=None,
                    help="clasificador de color para YOLO-World (por defecto modelos/clasificador.pt)")
    ap.add_argument("--conf-clas", type=float, default=0.4,
                    help="confianza minima del clasificador de color")
    ap.add_argument("--recolectar", choices=CARPETAS_DRIVE, default=None,
                    help="con la tecla 'c' guarda fotos limpias en datos/recolectadas/<color>/")
    args = ap.parse_args()

    if args.modelo:
        ruta = Path(args.modelo)
    else:
        ruta = MODELO_DEFECTO if MODELO_DEFECTO.exists() else MODELO_WORLD
    if not ruta.exists():
        raise SystemExit(f"No se encontro el modelo: {ruta}\n"
                         "Entrena en Colab (notebooks/) y copia best.pt a modelos/.")

    es_world = "world" in ruta.name
    if es_world:
        modelo = YOLOWorld(str(ruta))
        modelo.set_classes(list(MAPEO_WORLD))
    else:
        modelo = YOLO(str(ruta))
    conf_min = args.conf if args.conf is not None else (0.15 if es_world else 0.45)

    # Con YOLO-World, si hay clasificador: YOLO-World ubica y el clasificador decide el color
    clasificador = None
    ruta_clas = Path(args.clasificador) if args.clasificador else MODELO_CLASIFICADOR
    if args.clasificador and not ruta_clas.exists():
        raise SystemExit(f"No se encontro el clasificador: {ruta_clas}")
    if es_world and ruta_clas.exists():
        clasificador = YOLO(str(ruta_clas))
    print(f"Modelo: {ruta.name} | confianza minima: {conf_min}"
          + (f" | color: {ruta_clas.name}" if clasificador else ""))
    cam = cv2.VideoCapture(args.camara)
    if not cam.isOpened():
        raise SystemExit("No se pudo abrir la camara. Revisa el permiso de Camara para la Terminal.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps, t_prev = 0.0, time.time()
    print("Detector iniciado. 'q' para salir, 's' para guardar captura.")
    if args.recolectar:
        dir_recolectar = DIR_RECOLECTADAS / args.recolectar
        dir_recolectar.mkdir(parents=True, exist_ok=True)
        print(f"Recolectando '{args.recolectar}': 'c' guarda la foto en {dir_recolectar}")

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        limpio = frame.copy()  # sin dibujos, para recolectar fotos de entrenamiento

        detecciones = detectar(frame, modelo, conf_min, args.imgsz, clasificador, args.conf_clas)
        conteo = Counter(d[5] for d in detecciones)
        principal = max(detecciones, key=lambda d: d[6])[4:7] if detecciones else None

        dibujar_mascaras(frame, detecciones)
        for x1, y1, x2, y2, nombre, clase, conf, _ in detecciones:
            dibujar_caja(frame, x1, y1, x2, y2, nombre, clase, conf)

        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        dibujar_leyenda(frame, conteo, fps)
        dibujar_veredicto(frame, principal)

        cv2.imshow("Detector de residuos - NTP 900.058", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("s"):
            DIR_CAPTURAS.mkdir(exist_ok=True)
            nombre = str(DIR_CAPTURAS / f"captura_{datetime.now():%Y%m%d_%H%M%S}.jpg")
            cv2.imwrite(nombre, frame)
            print("Guardada:", nombre)
        if tecla == ord("c") and args.recolectar:
            nombre = dir_recolectar / f"{args.recolectar}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            cv2.imwrite(str(nombre), limpio)
            print(f"Recolectada ({len(list(dir_recolectar.glob('*.jpg')))}):", nombre)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
