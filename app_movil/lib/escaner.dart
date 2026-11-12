// Pantalla principal: la camara clasifica el residuo del recuadro central.
import 'dart:async';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'almacen.dart';
import 'clasificador.dart';
import 'ficha.dart';
import 'tachos.dart';
import 'tema.dart';

const confMinima = 0.7; // confirma el tacho (igual que CONF_CENTRO del detector de la PC)
const confDuda = 0.4; // entre 0.4 y 0.7 muestra las dos opciones mas probables
const confSalida = 0.55; // un tacho confirmado se mantiene hasta bajar de aqui (evita parpadeos)
const cuadrosPromedio = 5; // suaviza las probabilidades entre cuadros
const estable = Duration(milliseconds: 500); // tiempo con el mismo tacho antes de confirmar

enum Estado { buscando, dudando, confirmado }

class Escaner extends StatefulWidget {
  final VoidCallback alPedirAyuda;
  const Escaner({super.key, required this.alPedirAyuda});

  @override
  State<Escaner> createState() => _EscanerState();
}

class _EscanerState extends State<Escaner> with WidgetsBindingObserver, SingleTickerProviderStateMixin {
  CameraController? _camara;
  Clasificador? _clasificador;
  String? _error;
  bool _ocupado = false, _linterna = false, _tecnico = false, _iniciando = false, _cerrando = false;
  late final AnimationController _respiracion =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 1400))..repeat(reverse: true);

  final _historial = <List<double>>[];
  List<double> _probs = List.filled(tachos.length, 0);
  Estado _estado = Estado.buscando;
  int _mejor = 0, _segundo = 1;
  int? _confirmado, _candidato;
  DateTime _desde = DateTime.now();
  Cuadro? _cuadro; // foto del ultimo tacho confirmado (para "¿Me equivoqué?")

  // modo tecnico
  final _toques = <DateTime>[];
  double _msInferencia = 0, _fps = 0;
  DateTime _ultimoCuadro = DateTime.now();
  int _correcciones = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _iniciar();
  }

  Future<void> _iniciar() async {
    if (_iniciando) return; // p. ej. al volver del permiso de camara mientras ya se estaba iniciando
    _iniciando = true;
    try {
      if (_clasificador == null) {
        final clasificador = await Clasificador.cargar();
        if (_cerrando) {
          await clasificador.cerrar(); // se salio del escaner mientras cargaba
          return;
        }
        _clasificador = clasificador;
      }
      final camaras = await availableCameras();
      final trasera = camaras.firstWhere((c) => c.lensDirection == CameraLensDirection.back,
          orElse: () => camaras.first);
      final camara = CameraController(trasera, ResolutionPreset.medium,
          enableAudio: false, imageFormatGroup: ImageFormatGroup.yuv420);
      await camara.initialize();
      if (_cerrando) {
        await camara.dispose();
        await _clasificador?.cerrar();
        return;
      }
      await camara.startImageStream(_procesar);
      if (!mounted) {
        await camara.dispose();
        return;
      }
      setState(() {
        _camara = camara;
        _linterna = false;
        _error = null;
      });
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    } finally {
      _iniciando = false;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState estado) {
    final camara = _camara;
    if (estado == AppLifecycleState.inactive && camara != null) {
      _camara = null;
      camara.dispose();
    } else if (estado == AppLifecycleState.resumed && _camara == null) {
      _iniciar();
    }
  }

  Future<void> _procesar(CameraImage imagen) async {
    final camara = _camara, clasificador = _clasificador;
    if (_ocupado || _cerrando || camara == null || clasificador == null) return;
    _ocupado = true;
    try {
      final t0 = DateTime.now();
      final rotacion = camara.description.sensorOrientation;
      final p = await clasificador.clasificar(imagen, rotacion);
      final ahora = DateTime.now();
      _msInferencia = 0.8 * _msInferencia + 0.2 * ahora.difference(t0).inMilliseconds;
      _fps = 0.8 * _fps + 0.2 * (1000 / math.max(1, ahora.difference(_ultimoCuadro).inMilliseconds));
      _ultimoCuadro = ahora;

      _historial.add(p);
      if (_historial.length > cuadrosPromedio) _historial.removeAt(0);
      final prom = List<double>.generate(
          tachos.length, (i) => _historial.map((h) => h[i]).reduce((a, b) => a + b) / _historial.length);
      final orden = List<int>.generate(tachos.length, (i) => i)..sort((a, b) => prom[b].compareTo(prom[a]));
      _actualizarEstado(prom, orden[0], orden[1], imagen, rotacion);
    } finally {
      _ocupado = false;
    }
  }

  void _actualizarEstado(List<double> prom, int mejor, int segundo, CameraImage imagen, int rotacion) {
    final conf = prom[mejor];
    var confirmado = _confirmado;
    // un tacho confirmado se mantiene mientras siga primero y no baje de confSalida
    if (confirmado != null && (mejor != confirmado || conf < confSalida)) confirmado = null;
    if (conf >= confMinima) {
      if (_candidato != mejor) {
        _candidato = mejor;
        _desde = DateTime.now();
      }
      if (confirmado == null && DateTime.now().difference(_desde) >= estable) {
        confirmado = mejor;
        _cuadro = Cuadro(imagen, rotacion);
        HapticFeedback.mediumImpact();
        Almacen.sumar(mejor);
      }
    } else {
      _candidato = null;
    }
    final estado = confirmado != null
        ? Estado.confirmado
        : conf >= confDuda
            ? Estado.dudando
            : Estado.buscando;
    if (!mounted) return;
    setState(() {
      _probs = prom;
      _mejor = mejor;
      _segundo = segundo;
      _confirmado = confirmado;
      _estado = estado;
    });
  }

  Future<void> _alternarLinterna() async {
    final camara = _camara;
    if (camara == null) return;
    try {
      await camara.setFlashMode(_linterna ? FlashMode.off : FlashMode.torch);
      setState(() => _linterna = !_linterna);
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(const SnackBar(content: Text('Este celular no tiene linterna disponible.')));
      }
    }
  }

  Future<void> _tocarLogo() async {
    final ahora = DateTime.now();
    _toques
      ..add(ahora)
      ..removeWhere((t) => ahora.difference(t) > const Duration(seconds: 3));
    if (_toques.length >= 5) {
      _toques.clear();
      final n = await Almacen.contarCorrecciones();
      setState(() {
        _tecnico = !_tecnico;
        _correcciones = n;
      });
      HapticFeedback.selectionClick();
    }
  }

  Future<void> _abrirFicha(int indice, {Cuadro? cuadro}) async {
    final camara = _camara;
    if (camara != null && camara.value.isStreamingImages) await camara.stopImageStream();
    if (!mounted) return;
    await Navigator.push(context, MaterialPageRoute<void>(builder: (_) => Ficha(indice: indice, cuadro: cuadro)));
    _historial.clear();
    if (_camara == camara && camara != null && !camara.value.isStreamingImages) {
      await camara.startImageStream(_procesar);
    }
    if (_tecnico) _correcciones = await Almacen.contarCorrecciones();
  }

  @override
  void dispose() {
    _cerrando = true;
    WidgetsBinding.instance.removeObserver(this);
    _respiracion.dispose();
    unawaited(_liberar(_camara, _clasificador));
    _camara = null;
    super.dispose();
  }

  /// Apaga la camara y libera el modelo SOLO cuando termina el cuadro que se esta analizando:
  /// si se libera a mitad de un analisis, el hilo del modelo usa memoria ya borrada y Android cierra la app.
  Future<void> _liberar(CameraController? camara, Clasificador? clasificador) async {
    try {
      if (camara != null && camara.value.isStreamingImages) await camara.stopImageStream();
    } catch (_) {}
    await camara?.dispose();
    for (var i = 0; _ocupado && i < 150; i++) {
      await Future<void>.delayed(const Duration(milliseconds: 20));
    }
    await clasificador?.cerrar();
  }

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: Colors.black,
      child: SafeArea(
        bottom: false,
        child: Column(children: [
          _encabezado(),
          Expanded(child: _vista()),
          _tarjeta(),
        ]),
      ),
    );
  }

  Widget _encabezado() => Padding(
        padding: const EdgeInsets.fromLTRB(16, 4, 4, 4),
        child: Row(children: [
          GestureDetector(
            onTap: _tocarLogo,
            child: Row(children: [
              const Text('🌱', style: TextStyle(fontSize: 26)),
              const SizedBox(width: 6),
              Text('Basurita', style: letra(24, FontWeight.w900, Colors.white)),
            ]),
          ),
          const Spacer(),
          IconButton(
            tooltip: 'Linterna',
            onPressed: _alternarLinterna,
            icon: Icon(_linterna ? Icons.flashlight_on : Icons.flashlight_off,
                color: _linterna ? Colors.amber : Colors.white),
          ),
          IconButton(
            tooltip: 'Cómo se usa',
            onPressed: widget.alPedirAyuda,
            icon: const Icon(Icons.help_outline, color: Colors.white),
          ),
        ]),
      );

  Widget _vista() {
    final camara = _camara;
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            const Icon(Icons.no_photography, color: Colors.white70, size: 48),
            const SizedBox(height: 12),
            Text('No se pudo abrir la cámara.\nRevisa el permiso de cámara de la app.',
                textAlign: TextAlign.center, style: letra(17, FontWeight.w600, Colors.white)),
            const SizedBox(height: 16),
            FilledButton(onPressed: _iniciar, child: const Text('Reintentar')),
          ]),
        ),
      );
    }
    if (camara == null) return const Center(child: CircularProgressIndicator());
    final color = _estado == Estado.confirmado ? tachos[_confirmado!].fondo : Colors.white;
    return Center(
      child: AspectRatio(
        aspectRatio: 1 / camara.value.aspectRatio,
        child: LayoutBuilder(builder: (context, c) {
          final lado = fraccionRecuadro * c.biggest.shortestSide;
          return Stack(alignment: Alignment.center, children: [
            CameraPreview(camara),
            AnimatedBuilder(
              animation: _respiracion,
              builder: (context, _) {
                final pulso = _estado == Estado.buscando ? _respiracion.value : 0.0;
                return CustomPaint(
                  size: Size.square(lado * (1 + 0.03 * pulso)),
                  painter: _Esquinas(color: color.withValues(alpha: 1 - 0.35 * pulso),
                      relleno: _estado == Estado.confirmado),
                );
              },
            ),
            if (_estado == Estado.buscando)
              Positioned(
                bottom: 16,
                child: _Pastilla(texto: 'Acerca el residuo al recuadro'),
              ),
            if (_tecnico) Positioned(top: 8, left: 8, right: 8, child: _panelTecnico()),
          ]);
        }),
      ),
    );
  }

  Widget _tarjeta() {
    final Widget contenido;
    final Key clave;
    switch (_estado) {
      case Estado.confirmado:
        final i = _confirmado!, t = tachos[i];
        clave = ValueKey('ok_$i');
        contenido = Material(
          color: t.fondo,
          borderRadius: BorderRadius.circular(24),
          child: InkWell(
            borderRadius: BorderRadius.circular(24),
            onTap: () => _abrirFicha(i, cuadro: _cuadro),
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 18),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Row(children: [
                  Hero(tag: 'icono_$i', child: Text(t.icono, style: const TextStyle(fontSize: 44))),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text('TACHO ${t.color}', style: letra(28, FontWeight.w900, t.texto)),
                      Text(t.residuo, style: letra(17, FontWeight.w600, t.texto)),
                    ]),
                  ),
                  Icon(Icons.chevron_right, color: t.texto),
                ]),
                const SizedBox(height: 12),
                Row(children: [
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: TweenAnimationBuilder<double>(
                        tween: Tween(end: _probs[i]),
                        duration: duracion,
                        builder: (_, v, _) => LinearProgressIndicator(
                          value: v,
                          minHeight: 10,
                          color: t.texto,
                          backgroundColor: t.texto.withValues(alpha: 0.2),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Text('${(_probs[i] * 100).round()}%', style: letra(17, FontWeight.w800, t.texto)),
                ]),
                const SizedBox(height: 10),
                Text('💡 ${t.consejo}', style: letra(15, FontWeight.w600, t.texto)),
              ]),
            ),
          ),
        );
      case Estado.dudando:
        final a = tachos[_mejor], b = tachos[_segundo];
        clave = ValueKey('duda_${_mejor}_$_segundo');
        contenido = Card(
          color: const Color(0xFF2B2B2B),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(children: [
              Text('¿${a.residuo} o ${b.residuo}?', textAlign: TextAlign.center,
                  style: letra(20, FontWeight.w800, Colors.white)),
              const SizedBox(height: 4),
              Text('Acércalo un poco más y con buena luz', style: letra(14, FontWeight.w500, Colors.white70)),
              const SizedBox(height: 12),
              Row(children: [
                Expanded(child: _Opcion(indice: _mejor, prob: _probs[_mejor], alTocar: () => _abrirFicha(_mejor))),
                const SizedBox(width: 10),
                Expanded(
                    child: _Opcion(indice: _segundo, prob: _probs[_segundo], alTocar: () => _abrirFicha(_segundo))),
              ]),
            ]),
          ),
        );
      case Estado.buscando:
        clave = const ValueKey('buscando');
        contenido = Card(
          color: const Color(0xFF2B2B2B),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Row(children: [
              const Text('📷', style: TextStyle(fontSize: 34)),
              const SizedBox(width: 14),
              Expanded(
                child: Text('Pon un residuo dentro del recuadro, de cerca y con fondo liso',
                    style: letra(17, FontWeight.w700, Colors.white)),
              ),
            ]),
          ),
        );
    }
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
      child: AnimatedSwitcher(
        duration: duracion,
        transitionBuilder: (hijo, anim) => FadeTransition(
          opacity: anim,
          child: SlideTransition(
            position: Tween(begin: const Offset(0, 0.25), end: Offset.zero)
                .animate(CurvedAnimation(parent: anim, curve: Curves.easeOutCubic)),
            child: hijo,
          ),
        ),
        child: KeyedSubtree(key: clave, child: contenido),
      ),
    );
  }

  Widget _panelTecnico() => Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(color: Colors.black.withValues(alpha: 0.7), borderRadius: BorderRadius.circular(12)),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
          Text('MODO TÉCNICO · modelo $versionModelo (YOLO11n-cls, TFLite)',
              style: letra(12, FontWeight.w800, Colors.amber)),
          Text('${_fps.toStringAsFixed(1)} cuadros/s · ${_msInferencia.round()} ms por cuadro · '
              'promedio de $cuadrosPromedio · correcciones guardadas: $_correcciones',
              style: letra(11, FontWeight.w600, Colors.white70)),
          const SizedBox(height: 6),
          for (var i = 0; i < tachos.length; i++)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 1.5),
              child: Row(children: [
                SizedBox(width: 74, child: Text(tachos[i].color, style: letra(11, FontWeight.w800, Colors.white))),
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: _probs[i],
                      minHeight: 8,
                      color: tachos[i].fondo,
                      backgroundColor: Colors.white12,
                    ),
                  ),
                ),
                SizedBox(
                  width: 44,
                  child: Text('${(_probs[i] * 100).toStringAsFixed(0)}%', textAlign: TextAlign.right,
                      style: letra(11, FontWeight.w700, Colors.white)),
                ),
              ]),
            ),
        ]),
      );
}

class _Opcion extends StatelessWidget {
  final int indice;
  final double prob;
  final VoidCallback alTocar;
  const _Opcion({required this.indice, required this.prob, required this.alTocar});

  @override
  Widget build(BuildContext context) {
    final t = tachos[indice];
    return Material(
      color: t.fondo,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: alTocar,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
          child: Column(children: [
            Text(t.icono, style: const TextStyle(fontSize: 28)),
            Text(t.color, style: letra(15, FontWeight.w900, t.texto)),
            Text('${(prob * 100).round()}%', style: letra(13, FontWeight.w700, t.texto)),
          ]),
        ),
      ),
    );
  }
}

class _Pastilla extends StatelessWidget {
  final String texto;
  const _Pastilla({required this.texto});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(20)),
        child: Text(texto, style: letra(15, FontWeight.w700, Colors.white)),
      );
}

/// Las 4 esquinas del recuadro (con relleno suave del color del tacho al confirmar).
class _Esquinas extends CustomPainter {
  final Color color;
  final bool relleno;
  _Esquinas({required this.color, required this.relleno});

  @override
  void paint(Canvas canvas, Size size) {
    final r = RRect.fromRectAndRadius(Offset.zero & size, const Radius.circular(22));
    if (relleno) canvas.drawRRect(r, Paint()..color = color.withValues(alpha: 0.18));
    final largo = size.width * 0.18;
    final lapiz = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = relleno ? 7 : 5
      ..strokeCap = StrokeCap.round;
    final sombra = Paint()
      ..color = Colors.black38
      ..style = PaintingStyle.stroke
      ..strokeWidth = lapiz.strokeWidth + 3
      ..strokeCap = StrokeCap.round;
    final w = size.width, h = size.height, k = 22.0;
    final trazos = [
      Path()..moveTo(0, largo)..lineTo(0, k)..quadraticBezierTo(0, 0, k, 0)..lineTo(largo, 0),
      Path()..moveTo(w - largo, 0)..lineTo(w - k, 0)..quadraticBezierTo(w, 0, w, k)..lineTo(w, largo),
      Path()..moveTo(w, h - largo)..lineTo(w, h - k)..quadraticBezierTo(w, h, w - k, h)..lineTo(w - largo, h),
      Path()..moveTo(largo, h)..lineTo(k, h)..quadraticBezierTo(0, h, 0, h - k)..lineTo(0, h - largo),
    ];
    for (final p in trazos) {
      canvas.drawPath(p, sombra);
      canvas.drawPath(p, lapiz);
    }
  }

  @override
  bool shouldRepaint(_Esquinas viejo) => viejo.color != color || viejo.relleno != relleno;
}
