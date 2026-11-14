---
title: ReciclaIA
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Clasifica residuos en los 7 tachos de la NTP 900.058
---

# ReciclaIA: versión web

Sube o toma una foto de un residuo y la página te dice en qué tacho va según la **NTP 900.058** (Perú).
Usa el mismo modelo del proyecto, **YOLO11n-cls "v9 limpio"** (`clasificador.pt`, md5
`35b25249faee5906f9eec61fd3787aa0`), en CPU. También aplica la misma regla de decisión y los mismos textos que la app móvil.

- **Pestañas:** *Subir foto* y *Cámara* (webcam de la PC o cámara del celular).
- **Resultado:** tarjeta del color del tacho, con el % de confianza, el estado y qué va / qué no va / un consejo, más un gráfico de barras con los 7 tachos.
  - 70 % o más: **confirmado**.
  - Entre 40 y 70 %: **dudando** (muestra las 2 opciones más probables).
  - Menos de 40 %: **no estoy seguro**.
- **"Recortar al centro (70 %)":** imita el recuadro de la app móvil.

Código completo, app Android y versión para Raspberry Pi: https://github.com/AlexCoilaJrt/basurita

## Ejecutar en la computadora
```bash
cd web
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Abre http://127.0.0.1:7860. Para detenerlo, presiona Ctrl+C.

| Archivo | Qué es |
|---|---|
| `app.py` | La interfaz (Gradio). Carga el modelo una sola vez al iniciar |
| `clasificador.pt` | Copia de `modelos/clasificador.pt` (si falta, usa `../modelos/clasificador.pt`) |
| `requirements.txt` | PyTorch solo CPU, Ultralytics y Gradio |
| `packages.txt` | Librerías del sistema que OpenCV necesita en Hugging Face Spaces |
