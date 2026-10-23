# Detector de residuos – NTP 900.058 (7 colores)

Detección de residuos en tiempo real con la webcam usando **YOLO11n** (Ultralytics).
Cada objeto se clasifica según el código de colores de la **NTP 900.058** (Perú):

| Color | Clase | Residuos |
|---|---|---|
| Azul | `azul_papel_carton` | Papel y cartón |
| Amarillo | `amarillo_metales` | Metales |
| Blanco | `blanco_plasticos` | Plásticos |
| Gris | `gris_vidrio` | Vidrio |
| Marrón | `marron_organicos` | Orgánicos |
| Rojo | `rojo_peligrosos` | Peligrosos |
| Negro | `negro_generales` | Generales (no aprovechables) |

## Estructura

```
detecciondebasura/
├── notebooks/entrenamiento_residuos_ntp900058.ipynb  # entrenamiento en Google Colab (GPU)
├── src/detector_residuos.py                         # detector en vivo con la webcam (Mac, CPU)
├── modelos/        # aquí va best.pt (descargado de Colab)
├── resultados/     # zip con métricas/gráficos del entrenamiento
├── capturas/       # fotos guardadas con la tecla 's'
├── datos/webcam/   # fotos propias para crear tu dataset en Roboflow
└── requirements.txt
```

## Flujo de trabajo

### 1. Entrenar (Google Colab)
1. Sube `notebooks/entrenamiento_residuos_ntp900058.ipynb` a Colab y elige **GPU T4**.
2. Pega tu API key de Roboflow en la celda 2 (no la subas a git).
3. Ejecuta todo. En la celda 4 revisa la tabla de mapeo: las clases marcadas `SIN MAPEAR` se descartan.
4. Al final se descargan `best.pt` y `resultados_residuos7.zip`.

### 2. Copiar resultados
- `best.pt` → `modelos/best.pt`
- `resultados_residuos7.zip` → `resultados/`

> **Sin `best.pt` también funciona:** el detector usa **YOLO-World** (`modelos/yolov8s-worldv2.pt`,
> se descarga solo), que busca ~43 residuos descritos en texto en `MAPEO_WORLD`
> (`src/detector_residuos.py`). Es una aproximación; el modelo entrenado es más preciso.

### 3. Ejecutar el detector (Mac Intel)
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/detector_residuos.py
```
Opciones: `--camara 1`, `--conf 0.5` (por defecto 0.45, o 0.15 con YOLO-World), `--imgsz 416` (más rápido) o `640` (más preciso).
La barra inferior muestra en qué tacho va el objeto detectado con mayor confianza.
Teclas: `q`/`ESC` salir, `s` guardar captura.

> Si la cámara no abre: *Ajustes del Sistema → Privacidad y seguridad → Cámara* y habilita la Terminal / VS Code.

### 4. (Opcional) Mejorar con fotos propias
1. Guarda fotos de residuos reales en `datos/webcam/`.
2. Súbelas a Roboflow y etiquétalas con los 7 nombres de clase de la tabla.
3. Descomenta la línea `"propias"` en la lista `DATASETS` del notebook y vuelve a entrenar.
