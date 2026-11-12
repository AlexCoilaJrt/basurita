# Procedencia del modelo final (v9 limpio)

md5 `35b25249faee5906f9eec61fd3787aa0` · YOLO11n-cls · 7 clases NTP 900.058 · imgsz 224

| Paso | Qué es | Notebook / archivo (Drive `data basurita`) |
|---|---|---|
| v1 | Entrenado con la partición congelada `dataset_ntp900058.zip` (1400/300/300 por color) | `modelos/clasificador_v1.pt` |
| v3 → v5 | Ajustes finos con datos públicos (HH, RealWaste, TACO, EPP médico, frutas podridas, etc.) | `clasificador_v5.ipynb` |
| v6n | Desde v5n + TrashNet, Waste Classification (O) y Medical Waste 4.0, sin fotos del examen | `clasificador_v6.ipynb` |
| v7n | Desde v1 con todo lo anterior + más datos médicos/orgánicos | `clasificador_v7.ipynb` |
| mezcla30 | Promedio de pesos 0.7·v6n + 0.3·v7n (sin entrenar) | `mezcla_v6_v7.ipynb` → `revision_v8_mezcla/mezcla30.pt` |
| v9 | Ajuste fino desde mezcla30: Medical Waste 4.0 recortado al objeto + 30% del train sobre fondos lisos | `clasificador_v9_rapido.ipynb` |
| **v9 limpio (FINAL)** | Igual que v9, **quitando las 2.3% de fotos nuevas casi iguales a val/test** (pHash ≤ 8). 10 épocas, SGD lr0 0.0005 | `verificacion_final.ipynb` → `runs_clasificador/v9r_limpio/weights/best.pt` = `modelos/clasificador.pt` |

## Datos de entrenamiento del modelo final
- Partición congelada (train): 9 800 fotos.
- Nuevas: TrashNet, Waste Classification clase O y Medical Waste 4.0 (recortado al objeto): ~4 400, menos las quitadas por fuga.
- Copias sobre fondo liso: 30% del train de todos los colores. Grises: 40% del train.
- Excluidas: 1 806 por nombre/sesión del examen (±20 s), 1 244 por pHash ≤ 12 vs examen, y las casi iguales a val/test.

## Qué declarar en el informe
1. TrashNet, Waste Classification y Medical Waste 4.0 son del **mismo origen que el examen**; se usaron **sin** sus fotos de test.
2. Algunas decisiones intermedias (v7 a v8b) se tomaron mirando el examen; el modelo final se eligió con **validación limpia
   y reglas fijas**.
3. Se verificó fuga de datos con pHash: se eliminó el 2.3% en v9 y se reportan métricas sobre val/test **limpios**.
4. La medida honesta es el **test limpio (89.3%)** y las fotos reales de celular; la nota estimada (21.7) es optimista.
