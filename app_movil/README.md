# Basurita: app Android

Clasifica residuos con la cámara según los 7 colores de la NTP 900.058. Usa **el mismo clasificador**
del proyecto (`modelos/clasificador.pt`, **v9 limpio**) convertido a TFLite (`assets/clasificador.tflite`).
No usa YOLO-World y funciona sin internet.

## Pantallas
- **Bienvenida** (solo la primera vez; se vuelve a ver con el botón **?** del escáner): 3 pasos y por qué se usa la cámara.
- **Escanear** (pestaña principal):
  - Esquinas animadas mientras busca: *"Acerca el residuo al recuadro"*.
  - **Dudando** (40–70 % de confianza): muestra las dos opciones más probables, *"¿Plásticos o Vidrio?"*.
  - **Confirmado** (70 % o más, estable medio segundo): el recuadro se pinta del color del tacho, el celular vibra
    y sube una tarjeta con el ícono, el tacho, la barra de confianza y un consejo. Al tocarla se abre la ficha.
  - Botón de **linterna**.
  - **Modo técnico**: toca el logo *Basurita* 5 veces. Muestra las 7 probabilidades, cuadros por segundo,
    milisegundos por cuadro, versión del modelo y cuántas fotos corregidas hay.
- **Ficha del tacho**: qué sí va, qué no va y un consejo. Desde el escáner tiene **"¿Me equivoqué?"**: eliges el tacho
  correcto y se guarda la foto del recuadro en
  `Android/data/pe.basurita.basurita_app/files/correcciones/<Color>/` (las carpetas tienen los mismos nombres
  que en `data basurita`: Azul, Amarillo, Blanco, Gris, Marron, Rojo, Negro). Se copian conectando el celular
  por cable y sirven para reentrenar.
- **Guía**: los 7 tachos; al tocar uno se abre su ficha.
- **Historial**: residuos confirmados hoy y en total, por color, y los últimos 7 días.

Modo claro y oscuro automáticos, fuente Nunito (incluida en la app, licencia en `assets/fuentes/OFL.txt`).

## Instalar en el celular
**Opción 1: el APK.** Copia el `.apk` al celular (por cable, Drive o WhatsApp) y ábrelo.
Android pedirá permitir "instalar apps de origen desconocido"; acepta. Al abrir la app, permite la cámara.
Si ya tenías una versión anterior, se instala encima.

**Opción 2: por cable desde la Mac.**
1. En el celular: *Ajustes → Acerca del teléfono* → toca 7 veces *Número de compilación*.
   Luego, en *Opciones de desarrollador*, activa *Depuración USB*.
2. Conecta el cable y, en la Terminal:
   ```bash
   cd app_movil
   flutter run --release
   ```

## Generar el APK
```bash
cd app_movil
flutter build apk --release
```
Sale en `build/app/outputs/flutter-apk/app-release.apk`.

## Si cambias de modelo
La app trae una **copia** del modelo, así que no se actualiza sola. Conviértelo de nuevo (mejor en Colab,
porque TensorFlow da problemas en Mac Intel):
```python
from ultralytics import YOLO
YOLO("clasificador.pt").export(format="tflite", imgsz=224)
```
Copia `clasificador_saved_model/clasificador_float32.tflite` a `app_movil/assets/clasificador.tflite`,
cambia `versionModelo` en `lib/clasificador.dart` y genera el APK otra vez. Si cambian las clases, actualiza
también la lista `tachos` en `lib/tachos.dart` (debe seguir el orden alfabético de las clases del modelo).

## Cómo funciona
| Archivo | Qué hace |
|---|---|
| `lib/clasificador.dart` | Recorta el cuadrado central (70 % del lado corto) de cada cuadro YUV420, lo endereza según el sensor, lo reduce a 224×224 promediando 2×2 muestras por píxel (como Ultralytics) y ejecuta el modelo TFLite en un hilo aparte |
| `lib/escaner.dart` | Promedia los últimos 5 cuadros y decide el estado: buscando, dudando o confirmado |
| `lib/tachos.dart` | Los 7 tachos: color, ícono, qué va, qué no va y consejo |
| `lib/almacen.dart` | Historial y fotos corregidas |
| `lib/ficha.dart`, `guia.dart`, `historial.dart`, `bienvenida.dart` | Las demás pantallas |

Verificado en la Mac: el modelo TFLite da el mismo color y la misma probabilidad que `clasificador.pt` (v9 limpio)
en las 1455 imágenes del examen. Con fotos simuladas como cuadros de cámara Android (YUV, sensor girado 90°),
la app da el mismo color en 7 de 8; la otra es un caso dudoso (papel arrugado: 43 % en la Mac, 51 % en la app),
en el que la app no confirma ningún tacho y muestra las dos opciones. Ultralytics usa la licencia AGPL-3.0.
