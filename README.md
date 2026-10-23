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
├── notebooks/entrenamiento_seg_drive.ipynb          # segmentación con tus fotos del Drive + TACO (Colab)
├── notebooks/entrenamiento_residuos_ntp900058.ipynb  # detección con datasets de Roboflow (Colab)
├── src/datos_residuos.py                            # etiquetado automático, división y TACO
├── src/detector_residuos.py                         # detector en vivo con la webcam (Mac, CPU)
├── modelos/        # aquí va best.pt (descargado de Colab)
├── resultados/     # zip con métricas/gráficos del entrenamiento
├── capturas/       # fotos guardadas con la tecla 's'
├── datos/webcam/   # fotos propias para crear tu dataset en Roboflow
└── requirements.txt
```

## Flujo de trabajo

### Recomendado: segmentación con tus fotos del Drive
`notebooks/entrenamiento_seg_drive.ipynb` (usa `src/datos_residuos.py`):
1. En el Drive, crea `MyDrive/Residuos/<color>/<objeto>/fotos...` con los 7 nombres de clase de la tabla.
2. Abre el notebook en Colab con GPU T4 y ejecútalo en orden:
   - **Etiquetado automático**: YOLO-World ubica el objeto, SAM dibuja el contorno y la carpeta da el color.
   - **Revisión**: hojas de miniaturas en `MyDrive/Residuos_resultados/revision/` (los dudosos salen con borde rojo).
   - **División por objeto**: todas las fotos de un objeto van al mismo split, color por color (70/20/10).
   - **TACO** (opcional), dividido por batch.
   - **Entrenamiento en 2 etapas**: `yolo11n-seg` con la base congelada (`freeze=10`) y luego un ajuste fino con `lr0=0.001`.
   - Se elige el mejor modelo con *valid* y se evalúa una sola vez en *test*.
3. Copia `best.pt` (también queda en `MyDrive/Residuos_resultados/best.pt`) a `modelos/`. El detector pinta el contorno de cada residuo con su color.

### Alternativa: detección con datasets de Roboflow

#### 1. Entrenar (Google Colab)
1. Sube `notebooks/entrenamiento_residuos_ntp900058.ipynb` a Colab y elige **GPU T4**.
2. Pega tu API key de Roboflow en la celda 2 (no la subas a git).
3. Ejecuta todo. En la celda 4 revisa la tabla de mapeo: las clases marcadas `SIN MAPEAR` se descartan.
4. Al final se descargan `best.pt` y `resultados_residuos7.zip`.

#### 2. Copiar resultados
- `best.pt` → `modelos/best.pt`
- `resultados_residuos7.zip` → `resultados/`

> **Sin `best.pt` también funciona:** el detector usa **YOLO-World** (`modelos/yolov8s-worldv2.pt`,
> se descarga solo), que busca ~50 residuos descritos en texto en `MAPEO_WORLD`
> (`src/detector_residuos.py`). Es una aproximación; el modelo entrenado es más preciso.

## Ejecutar el detector (Mac Intel)
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

## (Opcional) Fotos propias en Roboflow (alternativa de detección)
1. Guarda fotos de residuos reales en `datos/webcam/`.
2. Súbelas a Roboflow y etiquétalas con los 7 nombres de clase de la tabla.
3. Descomenta la línea `"propias"` en la lista `DATASETS` del notebook y vuelve a entrenar.
