# Detector de residuos – NTP 900.058 (7 colores)

Detección de residuos en tiempo real con la webcam (Ultralytics).
Cada objeto se clasifica según el código de colores de la **NTP 900.058** (Perú).

**Cómo funciona:** YOLO-World (preentrenado, sin modificar) **ubica** cada objeto y nuestro
clasificador **YOLO11n-cls** (`modelos/clasificador.pt`, entrenado con `data basurita`) **decide el color**.
**Plan B:** si no queda ningún objeto (YOLO-World no encontró nada, o el clasificador descartó todos por
baja confianza), el clasificador mira el centro de la imagen y muestra el color solo si está seguro al 70% o más.
Como el clasificador siempre elige uno de los 7 colores (no tiene la opción "no hay nada"), una pared con
textura o una cara podrían mostrar un tacho; para la demo se puede apagar con `--sin-plan-b`.

| Color | Acierto en test |
|---|---|
| Marrón | 99,0% |
| Azul | 94,0% |
| Gris | 92,0% |
| Amarillo / Negro | 91,3% |
| Rojo | 85,7% |
| Blanco | 82,7% |

Acierto general en val: **90,8%** (top-1). El dataset se dividió foto por foto al azar, así que estos
valores pueden ser algo optimistas respecto a la cámara real.

Documentación detallada en `docs/`:
- `Explicacion_detector_residuos.docx`: todo el proyecto, el entrenamiento (optimizadores, pérdida, aumento de datos, épocas) y los resultados.
- `Preentrenado_y_nuestro_entrenamiento.docx`: qué estaba entrenado, qué entrenamos nosotros y cómo mejorar con más datos.

## Los 7 colores

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
├── src/detector_residuos.py                         # detector en vivo con la webcam (Mac, CPU)      [usado]
├── notebooks/dataset_ntp900058_splits.ipynb        # arma el dataset de clasificación en el Drive  [usado]
├── notebooks/entrenamiento_clasificador.ipynb      # entrena el clasificador de color (Colab)      [usado]
├── notebooks/entrenamiento_seg_drive.ipynb          # segmentación con fotos del Drive + TACO        [alternativa]
├── notebooks/entrenamiento_residuos_ntp900058.ipynb  # detección con datasets de Roboflow           [alternativa]
├── src/datos_residuos.py                            # herramientas del notebook de segmentación      [alternativa]
├── docs/           # documentación en Word
├── modelos/        # clasificador.pt (en git, ~3 MB); yolov8s-worldv2.pt se descarga la primera vez
├── datos/          # recolectadas/<Color>/ (tecla 'c') y pruebas/ (fuera de git)
├── capturas/       # fotos guardadas con la tecla 's'
├── resultados/     # métricas y copias de Colab (fuera de git)
└── requirements.txt
```

## Flujo de trabajo

### Usado: clasificador de color con `data basurita` (Drive)
1. `notebooks/dataset_ntp900058_splits.ipynb` arma `MyDrive/data basurita/splits` y `dataset_ntp900058.zip`
   (Garbage Dataset V2 + TrashBox + tus fotos de `data basurita/<Color>/`).
2. `notebooks/entrenamiento_clasificador.ipynb` entrena **YOLO11n-cls** en 2 etapas (base congelada + ajuste fino)
   y guarda `clasificador.pt` en `MyDrive/data basurita/modelos/`.
3. Copia `clasificador.pt` a `modelos/`. El detector queda así: **YOLO-World ubica el objeto y el clasificador decide el color**.

Los datos se quedan en el Drive; en este proyecto solo van el código y los modelos `.pt`.

### Alternativa (no usada aún): segmentación con tus fotos del Drive
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
> se descarga en `modelos/` la primera vez que ejecutas el detector), que busca ~50 residuos descritos en texto en `MAPEO_WORLD`
> (`src/detector_residuos.py`). Es una aproximación; el modelo entrenado es más preciso.

## Ejecutar el detector
Requisitos: **Python 3.11** (64 bits) y **git** (para instalar `clip`). La primera ejecución necesita
internet: descarga YOLO-World (~26 MB, en `modelos/`) y el modelo de CLIP (~340 MB).

### Mac
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/detector_residuos.py
```
Si la Mac no tiene git, macOS ofrece instalar las *Command Line Tools* (acepta).

### Windows
Instala [Python 3.11](https://www.python.org/downloads/) (marca *Add python.exe to PATH*) y [Git for Windows](https://git-scm.com/download/win). En PowerShell:
```powershell
git clone https://github.com/AlexCoilaJrt/basurita.git
cd basurita
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\detector_residuos.py
```
- Si PowerShell no deja ejecutar `activate`: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- Si la cámara no abre: *Configuración → Privacidad y seguridad → Cámara* → permitir a las aplicaciones de escritorio.

### Opciones
`--camara 1`, `--conf 0.5` (por defecto 0.45, o 0.15 con YOLO-World), `--imgsz 416` (más rápido) o `640` (más preciso),
`--sin-plan-b` (no mira el centro cuando no queda ningún objeto).
La barra inferior muestra en qué tacho va el objeto detectado con mayor confianza.
Teclas: `q`/`ESC` salir, `s` guardar captura.

### Recolectar fotos para mejorar el clasificador
```bash
python src/detector_residuos.py --recolectar Blanco
```
Con la tecla `c` se guarda la foto **sin dibujos** en `datos/recolectadas/Blanco/`. Súbelas a
`data basurita/Blanco/` en el Drive y vuelve a ejecutar los dos notebooks. Entrena siempre con los
datos anteriores **y** los nuevos juntos; si solo usas los nuevos, el modelo olvida los demás colores.

> Si la cámara no abre en Mac: *Ajustes del Sistema → Privacidad y seguridad → Cámara* y habilita la Terminal / VS Code.

## (Opcional) Fotos propias en Roboflow (alternativa de detección)
1. Guarda fotos de residuos reales en `datos/webcam/`.
2. Súbelas a Roboflow y etiquétalas con los 7 nombres de clase de la tabla.
3. Descomenta la línea `"propias"` en la lista `DATASETS` del notebook y vuelve a entrenar.
