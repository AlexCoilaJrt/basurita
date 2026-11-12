# ReciclaIA – Resultados finales

**Modelo final:** v9 limpio · YOLO11n-cls (1.5 M parámetros, 3.2 MB) · 7 clases NTP 900.058 · imgsz 224
**md5:** `35b25249faee5906f9eec61fd3787aa0`

## 1. Verificación de fuga de datos (pHash, distancia ≤ 4 = casi igual)
| Comparación | Fotos | Casi iguales | % |
|---|---|---|---|
| val vs fotos nuevas de v9 | 2100 | 48 | 2.29 |
| test vs fotos nuevas de v9 | 2100 | 39 | 1.86 |
| examen vs fotos nuevas de v9 | 1455 | 0 | 0.00 |
| val vs train original (dataset v1) | 2100 | 212 | 10.10 |
| test vs train original (dataset v1) | 2100 | 214 | 10.19 |
| examen vs train original (dataset v1) | 1455 | 35 | 2.41 |

- La fuga de v9 (2.3%) se **eliminó**: se quitaron esas fotos del train y se reentrenó → **v9 limpio**.
- El dataset original (v1) traía ~10% de duplicados entre sus propios splits y 35 fotos parecidas al examen (TrashNet).
  Afecta a todas las versiones; por eso las métricas se reportan sobre **val limpio (1842 fotos)** y **test limpio (1847 fotos)**,
  sin ninguna foto parecida a train.

## 2. Resultados (datos limpios)
| Prueba | v5 | v8b | v9 | **v9 limpio** |
|---|---|---|---|---|
| Val limpio a color | 89.9% | 90.9% | 90.7% | 90.5% |
| Val limpio en grises | 86.2% | 87.8% | 86.9% | 86.9% |
| Val limpio con objeto sobre fondo liso | 44.4% | 46.3% | 74.4% | **76.5%** |
| **Test limpio** | 89.3% | 89.9% | 88.5% | **89.3%** |
| Falsos Rojo (no peligroso sobre fondo → "Rojo") | 15.4% | 29.4% | 4.9% | **4.0%** |
| Fondos vacíos llamados Rojo | 35.7% | 70.0% | 65.3% | **32.3%** |
| Fotos reales de celular | 57.0% · 8/10 | 51.9% · 7/10 | 62.0% · 9/10 | 58.2% · 8/10 |
| Nota estimada (fórmula del docente) | 17.7 | 21.4 | 21.4 | **21.7** |

Test limpio por clase (v9 limpio): Azul 92% · Amarillo 89% · Blanco 85% · Gris 89% · Marrón 99% · Rojo 83% · Negro 93%.

Examen del docente por grupo (v9 limpio): papel/cartón 99% · metal 91% · plástico 92% · vidrio 96.5% · orgánicos 89% ·
Rojo 77% · médico general 99% · Residu 60% · Anorganik 46%.

## 3. Cómo leer estos números
- La medida más honesta es el **test limpio (89.3%)** y las **fotos reales de celular (8/10 objetos)**.
- La nota estimada (21.7) es **optimista**: se usaron fuentes del mismo origen que el examen (sin sus fotos de test).
- **Limitaciones:** Anorganik (46%) y Residu (60%) son grupos mezclados y en grises; con una superficie vacía el modelo
  todavía puede responder un color (el programa de cámara muestra "Sin objeto"); test de celular pequeño (79 fotos).
