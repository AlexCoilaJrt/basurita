// El clasificador del proyecto (modelos/clasificador.pt, v9 limpio) exportado a TFLite.
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

const versionModelo = 'v9 limpio';
const tamano = 224; // entrada del modelo (imgsz=224)
const fraccionRecuadro = 0.7; // lado del recuadro respecto al lado corto de la imagen

/// Copia de un cuadro de la camara (para guardarlo despues como foto).
class Cuadro {
  final Uint8List y, u, v;
  final int filaY, filaUV, pasoUV, ancho, alto, rotacion;
  Cuadro(CameraImage im, this.rotacion)
      : y = Uint8List.fromList(im.planes[0].bytes),
        u = Uint8List.fromList(im.planes[1].bytes),
        v = Uint8List.fromList(im.planes[2].bytes),
        filaY = im.planes[0].bytesPerRow,
        filaUV = im.planes[1].bytesPerRow,
        pasoUV = im.planes[1].bytesPerPixel ?? 1,
        ancho = im.width,
        alto = im.height;
}

/// Recorre el cuadrado central de una imagen YUV420, enderezado segun la orientacion del sensor,
/// en una cuadricula de lado x lado, y entrega cada pixel en RGB (0..255).
/// Cada pixel promedia muestras x muestras puntos de su celda (suaviza al reducir, como Ultralytics).
void _recorrerRecuadro(Uint8List y, int filaY, Uint8List u, Uint8List v, int filaUV, int pasoUV,
    int w, int h, int rotacion, int salidaLado, int muestras, void Function(double r, double g, double b) pixel) {
  final lado = (fraccionRecuadro * (w < h ? w : h)).round();
  final x0 = (w - lado) ~/ 2, y0 = (h - lado) ~/ 2;
  final n = muestras * muestras;
  for (var fila = 0; fila < salidaLado; fila++) {
    for (var col = 0; col < salidaLado; col++) {
      var r = 0.0, g = 0.0, bl = 0.0;
      for (var sy = 0; sy < muestras; sy++) {
        for (var sx = 0; sx < muestras; sx++) {
          // (a, b): posicion en el recuadro ya enderezado; (px, py): en la imagen del sensor
          final a = (col + (sx + 0.5) / muestras) * lado ~/ salidaLado;
          final b = (fila + (sy + 0.5) / muestras) * lado ~/ salidaLado;
          final (px, py) = switch (rotacion) {
            90 => (b, lado - 1 - a),
            180 => (lado - 1 - a, lado - 1 - b),
            270 => (lado - 1 - b, a),
            _ => (a, b),
          };
          final x = px + x0, yy = py + y0;
          final luz = y[yy * filaY + x];
          final iUV = (yy >> 1) * filaUV + (x >> 1) * pasoUV;
          final cb = u[iUV] - 128, cr = v[iUV] - 128;
          r += (luz + 1.402 * cr).clamp(0, 255);
          g += (luz - 0.344136 * cb - 0.714136 * cr).clamp(0, 255);
          bl += (luz + 1.772 * cb).clamp(0, 255);
        }
      }
      pixel(r / n, g / n, bl / n);
    }
  }
}

/// Recorta el cuadrado central, lo endereza, lo reduce a 224x224 y lo deja en RGB 0..1 (como Ultralytics).
void prepararEntrada(Uint8List y, int filaY, Uint8List u, Uint8List v, int filaUV, int pasoUV,
    int w, int h, int rotacion, Float32List salida) {
  var k = 0;
  _recorrerRecuadro(y, filaY, u, v, filaUV, pasoUV, w, h, rotacion, tamano, 2, (r, g, b) {
    salida[k++] = r / 255.0;
    salida[k++] = g / 255.0;
    salida[k++] = b / 255.0;
  });
}

/// El recuadro central como JPEG (para "¿Me equivoqué?").
Uint8List recuadroJpeg(Cuadro c, {int lado = 448}) {
  final rgb = Uint8List(lado * lado * 3);
  var k = 0;
  _recorrerRecuadro(c.y, c.filaY, c.u, c.v, c.filaUV, c.pasoUV, c.ancho, c.alto, c.rotacion, lado, 1, (r, g, b) {
    rgb[k++] = r.round();
    rgb[k++] = g.round();
    rgb[k++] = b.round();
  });
  final imagen = img.Image.fromBytes(width: lado, height: lado, bytes: rgb.buffer, numChannels: 3);
  return img.encodeJpg(imagen, quality: 92);
}

class Clasificador {
  final Interpreter _interprete;
  final IsolateInterpreter _modelo;
  final _entrada = Float32List(tamano * tamano * 3);
  Clasificador._(this._interprete, this._modelo);

  static Future<Clasificador> cargar() async {
    final interprete = await Interpreter.fromAsset('assets/clasificador.tflite');
    final modelo = await IsolateInterpreter.create(address: interprete.address);
    return Clasificador._(interprete, modelo);
  }

  /// Probabilidades de los 7 tachos para el recuadro central de la imagen.
  Future<List<double>> clasificar(CameraImage imagen, int rotacion) async {
    final pY = imagen.planes[0], pU = imagen.planes[1], pV = imagen.planes[2];
    prepararEntrada(pY.bytes, pY.bytesPerRow, pU.bytes, pV.bytes, pU.bytesPerRow, pU.bytesPerPixel ?? 1,
        imagen.width, imagen.height, rotacion, _entrada);
    final salida = [List<double>.filled(7, 0)];
    await _modelo.run(_entrada.buffer, salida);
    return salida[0];
  }

  bool _cerrado = false;

  /// Libera el modelo (solo la primera vez). Llamar solo cuando no hay un analisis en curso.
  Future<void> cerrar() async {
    if (_cerrado) return;
    _cerrado = true;
    await _modelo.close();
    _interprete.close();
  }
}
