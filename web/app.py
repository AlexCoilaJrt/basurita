"""ReciclaIA web: clasifica una foto de un residuo en los 7 tachos de la NTP 900.058.

Usa el mismo modelo del proyecto (YOLO11n-cls, "v9 limpio") y la misma regla que la app movil:
  >= 70 %  confirmado
  40-70 %  dudando (muestra las 2 opciones mas probables)
  < 40 %   no estoy seguro

Local:  python web/app.py   ->  http://127.0.0.1:7860
"""
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps
from ultralytics import YOLO

AQUI = Path(__file__).resolve().parent
REPO = "https://github.com/AlexCoilaJrt/basurita"
CONF_CONFIRMADO = 0.70  # igual que confMinima en app_movil/lib/escaner.dart
CONF_DUDA = 0.40  # igual que confDuda
FRACCION_RECUADRO = 0.70  # igual que fraccionRecuadro en app_movil/lib/clasificador.dart

# Los 7 tachos (copiados de app_movil/lib/tachos.dart para que la web y la app digan lo mismo)
TACHOS = {
    "amarillo": dict(color="AMARILLO", residuo="Metales", icono="🥫", fondo="#FFD600", texto="#000000",
                     si=["Latas de bebida y de conserva", "Tapas metálicas", "Papel aluminio limpio",
                         "Chatarra pequeña"],
                     no=["Aerosoles y pilas (van al Rojo)", "Latas con restos de comida"],
                     consejo="Enjuaga las latas y aplástalas."),
    "azul": dict(color="AZUL", residuo="Papel y cartón", icono="📄", fondo="#1565C0", texto="#FFFFFF",
                 si=["Hojas, cuadernos y sobres", "Periódicos y revistas", "Cajas de cartón"],
                 no=["Papel higiénico o servilletas usadas (Negro)", "Papel con grasa o comida (Negro)",
                     "Vasos de papel plastificados (Negro)"],
                 consejo="Dobla las cajas y mantenlo seco."),
    "blanco": dict(color="BLANCO", residuo="Plásticos", icono="🧴", fondo="#F5F5F5", texto="#000000",
                   si=["Botellas PET", "Envases de shampoo y detergente", "Tapas de plástico",
                       "Bolsas y láminas limpias"],
                   no=["Tecnopor (Negro)", "Envolturas metalizadas de snacks (Negro)",
                       "Plástico con restos de comida"],
                   consejo="Enjuaga las botellas y aplástalas."),
    "gris": dict(color="GRIS", residuo="Vidrio", icono="🍾", fondo="#9E9E9E", texto="#000000",
                 si=["Botellas de vidrio", "Frascos", "Envases de vidrio"],
                 no=["Focos y fluorescentes (Rojo)", "Cerámica y porcelana (Negro)", "Espejos"],
                 consejo="Sin tapa y enjuagado. Cuidado con los bordes rotos."),
    "marron": dict(color="MARRÓN", residuo="Orgánicos", icono="🍎", fondo="#6D4C41", texto="#FFFFFF",
                   si=["Restos de frutas y verduras", "Cáscaras y restos de comida", "Hojas y ramas",
                       "Café y bolsitas de té"],
                   no=["Bolsas o envases (sepáralos)", "Aceite usado (llévalo a un punto de acopio)"],
                   consejo="Sin bolsas ni envases."),
    "negro": dict(color="NEGRO", residuo="Generales (no aprovechables)", icono="🗑️", fondo="#212121",
                  texto="#FFFFFF",
                  si=["Papel higiénico y servilletas usadas", "Tecnopor", "Envolturas de snacks", "Colillas",
                      "Cerámica rota"],
                  no=["Pilas y medicinas (Rojo)", "Residuos reciclables limpios (usa su tacho)"],
                  consejo="Si se puede reciclar, no va aquí."),
    "rojo": dict(color="ROJO", residuo="Peligrosos", icono="☣️", fondo="#C62828", texto="#FFFFFF",
                 si=["Pilas y baterías", "Medicinas vencidas", "Jeringas, guantes y mascarillas usadas",
                     "Aerosoles", "Focos, fluorescentes y termómetros"],
                 no=["Basura común (no llenes este tacho con otros residuos)"],
                 consejo="Manipula con cuidado. Las jeringas, en un envase rígido."),
}


def _buscar_modelo():
    """web/clasificador.pt (Hugging Face Spaces) o, si no esta, modelos/clasificador.pt del repositorio."""
    for ruta in (AQUI / "clasificador.pt", AQUI.parent / "modelos" / "clasificador.pt"):
        if ruta.exists():
            return ruta
    raise SystemExit("No encuentro clasificador.pt (ni en web/ ni en modelos/).")


def _clave(nombre):
    """'marron_organicos' -> 'marron' (para buscar el tacho)."""
    n = nombre.lower().replace("ó", "o")
    for k in TACHOS:
        if n.startswith(k):
            return k
    raise SystemExit(f"Clase del modelo no reconocida: {nombre}")


# El modelo se carga una sola vez, al iniciar
RUTA_MODELO = _buscar_modelo()
MODELO = YOLO(str(RUTA_MODELO), task="classify")
NOMBRES = [MODELO.names[i] for i in sorted(MODELO.names)]  # se leen del modelo, no a mano
CLAVES = [_clave(n) for n in NOMBRES]
print(f"Modelo cargado: {RUTA_MODELO} | clases: {NOMBRES}")


def recortar_centro(imagen, fraccion=FRACCION_RECUADRO):
    """Cuadrado central de lado = fraccion del lado corto (como el recuadro de la app movil)."""
    w, h = imagen.size
    lado = round(fraccion * min(w, h))
    x0, y0 = (w - lado) // 2, (h - lado) // 2
    return imagen.crop((x0, y0, x0 + lado, y0 + lado))


def _lista(items):
    return "".join(f"<li>{t}</li>" for t in items)


def tarjeta(probs):
    orden = sorted(range(len(probs)), key=lambda i: -probs[i])
    i1, i2 = orden[0], orden[1]
    t, conf = TACHOS[CLAVES[i1]], probs[i1]
    if conf >= CONF_CONFIRMADO:
        estado, fondo, texto = "✅ Confirmado", t["fondo"], t["texto"]
        titulo = f"{t['icono']} Tacho {t['color']}: {t['residuo']}"
        detalle = (f"<div class='rc-cols'><div><b>Sí va</b><ul>{_lista(t['si'])}</ul></div>"
                   f"<div><b>No va</b><ul>{_lista(t['no'])}</ul></div></div>"
                   f"<p class='rc-consejo'>💡 {t['consejo']}</p>")
    elif conf >= CONF_DUDA:
        t2 = TACHOS[CLAVES[i2]]
        estado, fondo, texto = "🤔 Dudando", "#FFF3E0", "#000000"
        titulo = f"¿{t['residuo']} o {t2['residuo']}?"
        detalle = (f"<p>Puede ir al tacho <b>{t['color']}</b> ({conf:.0%}) o al <b>{t2['color']}</b> "
                   f"({probs[i2]:.0%}). Acerca el objeto, centra la foto o usa mejor luz.</p>")
    else:
        estado, fondo, texto = "❓ No estoy seguro", "#ECEFF1", "#000000"
        titulo = "No estoy seguro"
        detalle = "<p>Acerca el objeto, que ocupe el centro de la foto, o usa mejor luz.</p>"
    return (f"<div class='rc-tarjeta' style='background:{fondo};color:{texto}'>"
            f"<div class='rc-estado'>{estado} · {conf:.1%} de confianza</div>"
            f"<div class='rc-titulo'>{titulo}</div>{detalle}</div>")


def barras(probs):
    filas = []
    for i in sorted(range(len(probs)), key=lambda i: -probs[i]):
        t = TACHOS[CLAVES[i]]
        color = "#BDBDBD" if t["color"] == "BLANCO" else t["fondo"]  # el blanco no se veria
        filas.append(f"<div class='rc-fila'><span class='rc-nombre'>{t['icono']} {t['color'].title()}</span>"
                     f"<span class='rc-pista'><span style='width:{probs[i] * 100:.1f}%;background:{color}'>"
                     f"</span></span><span class='rc-pct'>{probs[i]:.1%}</span></div>")
    return "<div class='rc-barras'>" + "".join(filas) + "</div>"


def clasificar(imagen, recortar):
    if imagen is None:
        return "<p>Sube una foto o toma una con la cámara.</p>", "", None
    imagen = ImageOps.exif_transpose(imagen).convert("RGB")  # fotos de celular giradas
    if recortar:
        imagen = recortar_centro(imagen)
    r = MODELO.predict(imagen, imgsz=224, device="cpu", verbose=False)[0]
    probs = [float(p) for p in r.probs.data.cpu().numpy()]
    return tarjeta(probs), barras(probs), imagen


CSS = """
.rc-tarjeta{border-radius:18px;padding:22px 24px;border:1px solid rgba(0,0,0,.12)}
.rc-estado{font-size:15px;opacity:.85;margin-bottom:6px}
.rc-titulo{font-size:28px;font-weight:800;margin-bottom:10px}
.rc-cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.rc-cols ul{margin:6px 0 0 18px;padding:0}
.rc-consejo{margin-top:12px;font-weight:600}
.rc-fila{display:flex;align-items:center;gap:10px;margin:6px 0}
.rc-nombre{width:110px;flex:none}
.rc-pista{flex:1;height:16px;border-radius:8px;background:rgba(128,128,128,.18);overflow:hidden}
.rc-pista span{display:block;height:100%;border-radius:8px}
.rc-pct{width:58px;text-align:right;font-variant-numeric:tabular-nums}
@media (max-width:600px){.rc-cols{grid-template-columns:1fr}.rc-titulo{font-size:22px}}
"""

ACERCA = f"""
### Acerca del modelo
- **Modelo:** YOLO11n-cls (Ultralytics), versión final *v9 limpio*, **1,54 M parámetros**, entrada 224×224.
- **Acierto:** **89,3 %** en la prueba limpia.
- **Clases:** los 7 colores de tachos de la **NTP 900.058** (Perú), leídos del propio modelo: {", ".join(NOMBRES)}.
- **Regla de decisión** (igual que la app móvil): 70 % o más = confirmado; 40–70 % = dudando (muestra las 2 opciones
  más probables); menos de 40 % = no estoy seguro.
- Funciona en CPU. Código, app Android y versión para Raspberry Pi: [{REPO}]({REPO})
"""

with gr.Blocks(title="ReciclaIA", css=CSS) as demo:
    gr.Markdown("# ♻️ ReciclaIA\nToma o sube una foto de un residuo y te digo en qué tacho va según la **NTP 900.058**.")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Subir foto"):
                    foto = gr.Image(type="pil", sources=["upload", "clipboard"], label="Foto del residuo")
                    boton_foto = gr.Button("Clasificar", variant="primary")
                with gr.Tab("Cámara"):
                    camara = gr.Image(type="pil", sources=["webcam"], label="Toma una foto")
                    boton_camara = gr.Button("Clasificar", variant="primary")
            recortar = gr.Checkbox(value=False, label="Recortar al centro (70 %), como el recuadro de la app móvil")
            vista = gr.Image(label="Lo que ve el modelo", interactive=False, height=200)
        with gr.Column():
            resultado = gr.HTML("<p>Sube una foto o toma una con la cámara.</p>")
            gr.Markdown("**Probabilidad de cada tacho**")
            grafico = gr.HTML()
    gr.Markdown(ACERCA)

    salidas = [resultado, grafico, vista]
    boton_foto.click(clasificar, [foto, recortar], salidas)
    foto.upload(clasificar, [foto, recortar], salidas)
    boton_camara.click(clasificar, [camara, recortar], salidas)
    camara.change(clasificar, [camara, recortar], salidas)

if __name__ == "__main__":
    demo.launch()  # local: http://127.0.0.1:7860 (en Spaces, Gradio usa su propia direccion)
