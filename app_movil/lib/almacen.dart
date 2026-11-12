// Lo que la app guarda en el celular: si ya vio la bienvenida, el historial y las fotos corregidas.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'tachos.dart';

class Almacen {
  static const _claveBienvenida = 'bienvenida_vista';
  static const _claveHistorial = 'historial'; // {"2026-11-08": [n0, ..., n6]}

  static Future<bool> bienvenidaVista() async =>
      (await SharedPreferences.getInstance()).getBool(_claveBienvenida) ?? false;

  static Future<void> marcarBienvenida() async =>
      (await SharedPreferences.getInstance()).setBool(_claveBienvenida, true);

  static String _hoy() => DateTime.now().toIso8601String().substring(0, 10);

  static Future<Map<String, List<int>>> historial() async {
    final texto = (await SharedPreferences.getInstance()).getString(_claveHistorial);
    if (texto == null) return {};
    final datos = jsonDecode(texto) as Map<String, dynamic>;
    return datos.map((dia, n) => MapEntry(dia, List<int>.from(n as List)));
  }

  static Future<void> sumar(int indice) async {
    final h = await historial();
    final dia = h.putIfAbsent(_hoy(), () => List.filled(tachos.length, 0));
    dia[indice]++;
    await (await SharedPreferences.getInstance()).setString(_claveHistorial, jsonEncode(h));
  }

  static Future<void> borrarHistorial() async =>
      (await SharedPreferences.getInstance()).remove(_claveHistorial);

  /// Carpeta de las fotos corregidas: Android/data/pe.basurita.basurita_app/files/correcciones
  /// (se ve al conectar el celular por cable). Dentro, una carpeta por color como en el Drive.
  static Future<Directory> carpetaCorrecciones() async {
    final base = await getExternalStorageDirectory() ?? await getApplicationDocumentsDirectory();
    return Directory('${base.path}/correcciones');
  }

  static Future<String> guardarCorreccion(Uint8List jpeg, int correcto, int predicho) async {
    final carpeta = Directory('${(await carpetaCorrecciones()).path}/${tachos[correcto].carpeta}');
    await carpeta.create(recursive: true);
    final marca = DateTime.now().toIso8601String().replaceAll(':', '-').split('.').first;
    final archivo = File('${carpeta.path}/app_${marca}_era_${tachos[predicho].carpeta}.jpg');
    await archivo.writeAsBytes(jpeg);
    return archivo.path;
  }

  static Future<int> contarCorrecciones() async {
    final carpeta = await carpetaCorrecciones();
    if (!await carpeta.exists()) return 0;
    return carpeta.list(recursive: true).where((e) => e.path.endsWith('.jpg')).length;
  }
}
