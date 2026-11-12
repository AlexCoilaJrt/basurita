// Historial: cuantos residuos se clasificaron hoy y en total, por color.
import 'package:flutter/material.dart';

import 'almacen.dart';
import 'tachos.dart';
import 'tema.dart';

class Historial extends StatefulWidget {
  const Historial({super.key});

  @override
  State<Historial> createState() => _HistorialState();
}

class _HistorialState extends State<Historial> {
  Map<String, List<int>>? _datos;

  @override
  void initState() {
    super.initState();
    _cargar();
  }

  Future<void> _cargar() async {
    final d = await Almacen.historial();
    if (mounted) setState(() => _datos = d);
  }

  Future<void> _borrar() async {
    final si = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('¿Borrar el historial?'),
        content: const Text('Se pierden los conteos. Las fotos corregidas no se borran.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancelar')),
          FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Borrar')),
        ],
      ),
    );
    if (si != true) return;
    await Almacen.borrarHistorial();
    _cargar();
  }

  @override
  Widget build(BuildContext context) {
    final datos = _datos;
    if (datos == null) return const Center(child: CircularProgressIndicator());
    final hoy = datos[DateTime.now().toIso8601String().substring(0, 10)] ?? List.filled(tachos.length, 0);
    final total = List<int>.generate(tachos.length, (i) => datos.values.fold(0, (s, d) => s + d[i]));
    final nHoy = hoy.fold(0, (a, b) => a + b), nTotal = total.fold(0, (a, b) => a + b);
    final dias = datos.keys.toList()..sort((a, b) => b.compareTo(a));
    final texto = Theme.of(context).textTheme;

    return SafeArea(
      child: RefreshIndicator(
        onRefresh: _cargar,
        child: ListView(padding: const EdgeInsets.all(20), children: [
          Row(children: [
            Expanded(child: Text('Historial', style: texto.headlineMedium)),
            if (nTotal > 0) IconButton(tooltip: 'Borrar historial', onPressed: _borrar, icon: const Icon(Icons.delete_outline)),
          ]),
          const SizedBox(height: 12),
          Card(
            color: verde,
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Row(children: [
                const Text('♻️', style: TextStyle(fontSize: 44)),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(nHoy == 0 ? 'Hoy aún no clasificaste residuos' : 'Hoy clasificaste',
                        style: letra(16, FontWeight.w700, Colors.white)),
                    if (nHoy > 0)
                      Text('$nHoy ${nHoy == 1 ? 'residuo' : 'residuos'}', style: letra(32, FontWeight.w900, Colors.white)),
                  ]),
                ),
              ]),
            ),
          ),
          const SizedBox(height: 20),
          Text('Hoy por color', style: texto.titleLarge),
          const SizedBox(height: 8),
          _Barras(valores: hoy),
          const SizedBox(height: 24),
          Text('Total: $nTotal', style: texto.titleLarge),
          const SizedBox(height: 8),
          _Barras(valores: total),
          if (dias.isNotEmpty) ...[
            const SizedBox(height: 24),
            Text('Últimos días', style: texto.titleLarge),
            const SizedBox(height: 4),
            for (final dia in dias.take(7))
              ListTile(
                contentPadding: EdgeInsets.zero,
                title: Text(dia),
                trailing: Text('${datos[dia]!.fold(0, (a, b) => a + b)}', style: texto.titleMedium),
                subtitle: Wrap(spacing: 6, children: [
                  for (var i = 0; i < tachos.length; i++)
                    if (datos[dia]![i] > 0) Text('${tachos[i].icono} ${datos[dia]![i]}'),
                ]),
              ),
          ],
          if (nTotal == 0) ...[
            const SizedBox(height: 24),
            Text('Cada residuo que el escáner confirma se suma aquí.',
                textAlign: TextAlign.center, style: texto.bodyLarge),
          ],
        ]),
      ),
    );
  }
}

class _Barras extends StatelessWidget {
  final List<int> valores;
  const _Barras({required this.valores});

  @override
  Widget build(BuildContext context) {
    final maximo = valores.fold(1, (a, b) => a > b ? a : b);
    return Column(children: [
      for (var i = 0; i < tachos.length; i++)
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 4),
          child: Row(children: [
            SizedBox(width: 34, child: Text(tachos[i].icono, style: const TextStyle(fontSize: 20))),
            SizedBox(width: 86, child: Text(tachos[i].color, style: Theme.of(context).textTheme.labelLarge)),
            Expanded(
              child: TweenAnimationBuilder<double>(
                tween: Tween(begin: 0, end: valores[i] / maximo),
                duration: const Duration(milliseconds: 600),
                curve: Curves.easeOutCubic,
                builder: (_, v, _) => ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: LinearProgressIndicator(
                    value: v,
                    minHeight: 14,
                    color: tachos[i].barra,
                    backgroundColor: Theme.of(context).colorScheme.surfaceContainerHighest,
                  ),
                ),
              ),
            ),
            SizedBox(
                width: 40,
                child: Text('${valores[i]}', textAlign: TextAlign.right, style: Theme.of(context).textTheme.titleMedium)),
          ]),
        ),
    ]);
  }
}
