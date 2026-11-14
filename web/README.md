# ReciclaIA: versión web

Hay dos versiones que hacen lo mismo, con el mismo modelo (**YOLO11n-cls "v9 limpio"**), la misma regla de decisión
que la app móvil y los mismos textos de cada tacho:

- **70 % o más:** confirmado.
- **40–70 %:** dudando (muestra las 2 opciones más probables).
- **Menos de 40 %:** no estoy seguro.

| | `navegador/` (pública) | `app.py` (local) |
|---|---|---|
| Dónde corre el modelo | En el navegador del usuario (ONNX Runtime Web) | En tu computadora (Python + PyTorch) |
| Link | **https://alexcoilajrt.github.io/basurita/** (GitHub Pages, gratis) | http://127.0.0.1:7860 |
| Cámara | En vivo, con recuadro central (promedio de 5 cuadros) | Foto desde la webcam |
| Instalación | Ninguna: abrir el link | Python 3.11 y `requirements.txt` |

## Versión pública (`navegador/`)
Página estática: `index.html`, `preproceso.js`, `clasificador.onnx` (el `.pt` convertido a ONNX) y `clases.json`
(nombres de las clases leídos del modelo). La foto no se envía a ningún servidor.

- **Se publica sola:** cada push a `main` la publica con `.github/workflows/pagina_web.yml`.
  La primera vez hay que activar *Settings → Pages → Source: GitHub Actions* en el repositorio.
- **Mismo resultado que el `.pt`:** `preproceso.js` reproduce el preprocesamiento de Ultralytics (escalado
  BILINEAR de PIL hasta 224 en el lado corto, recorte central, RGB 0..1). Resultados de la verificación:
  - 200 imágenes del examen (cuadradas y rectangulares): los píxeles difieren como máximo en 1/255 y el color
    coincide con `clasificador.pt` en **200 de 200**.
  - En Chrome real: **28 de 28** iguales, con una diferencia de confianza menor a 1 punto.
- **Probarla en la computadora:** necesita un servidor, porque abrir el archivo con doble clic no funciona:
  ```bash
  cd web/navegador
  python3 -m http.server 8000      # abre http://localhost:8000
  ```
- **Si cambias de modelo:** `python herramientas/exportar_web.py` vuelve a generar `clasificador.onnx` y `clases.json`.

## Versión local con Gradio (`app.py`)
Mac:
```bash
cd web
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Windows (PowerShell):
```powershell
cd web
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Abre http://127.0.0.1:7860 y detenla con Ctrl+C. `clasificador.pt` es una copia de `modelos/clasificador.pt`
(si falta, usa `../modelos/clasificador.pt`).

> Nota: Hugging Face Spaces ahora exige la suscripción de pago PRO para alojar apps de Gradio, por eso la versión
> pública es la del navegador (GitHub Pages es gratis).
