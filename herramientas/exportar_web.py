"""Convierte modelos/clasificador.pt a ONNX para la pagina web (web/navegador/), sin modificar el .pt.

Uso (con un entorno que tenga ultralytics; instala antes: pip install "onnx<1.18" onnxslim "numpy<2"):
    python herramientas/exportar_web.py
Crea web/navegador/clasificador.onnx y web/navegador/clases.json (nombres leidos del modelo).
"""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent
PT = RAIZ / "modelos" / "clasificador.pt"
DESTINO = RAIZ / "web" / "navegador"

with tempfile.TemporaryDirectory() as tmp:
    copia = Path(tmp) / "clasificador.pt"
    shutil.copy(PT, copia)  # se exporta una copia: el original no se toca
    modelo = YOLO(str(copia), task="classify")
    onnx = modelo.export(format="onnx", imgsz=224, opset=17, simplify=True, dynamic=False)
    shutil.copy(onnx, DESTINO / "clasificador.onnx")
    clases = [modelo.names[i] for i in sorted(modelo.names)]

md5 = hashlib.md5(PT.read_bytes()).hexdigest()
(DESTINO / "clases.json").write_text(
    json.dumps({"modelo": "v9 limpio", "md5_pt": md5, "clases": clases}, ensure_ascii=False, indent=1),
    encoding="utf-8")
print("Listo:", DESTINO / "clasificador.onnx", "| clases:", clases, "| md5 del .pt:", md5)
