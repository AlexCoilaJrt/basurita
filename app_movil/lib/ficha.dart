// Ficha de un tacho: que va, que no va y un consejo. Desde el escaner, tambien "¿Me equivoqué?".
import 'package:flutter/material.dart';

import 'almacen.dart';
import 'clasificador.dart';
import 'tachos.dart';
import 'tema.dart';

class Ficha extends StatelessWidget {
  final int indice;
  final Cuadro? cuadro; // foto del escaner (solo si se abrio desde un resultado)
  const Ficha({super.key, required this.indice, this.cuadro});

  @override
  Widget build(BuildContext context) {
    final t = tachos[indice];
    return Scaffold(
      body: CustomScrollView(slivers: [
        SliverAppBar(
          expandedHeight: 230,
          pinned: true,
          backgroundColor: t.fondo,
          foregroundColor: t.texto,
          flexibleSpace: FlexibleSpaceBar(
            background: SafeArea(
              child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                Hero(tag: 'icono_$indice', child: Text(t.icono, style: const TextStyle(fontSize: 72))),
                const SizedBox(height: 8),
                Text('TACHO ${t.color}', style: letra(30, FontWeight.w900, t.texto)),
                Text(t.residuo, style: letra(18, FontWeight.w600, t.texto)),
              ]),
            ),
          ),
        ),
        SliverPadding(
          padding: const EdgeInsets.all(20),
          sliver: SliverList.list(children: [
            _Lista(titulo: 'Sí va', icono: Icons.check_circle, color: Colors.green.shade600, items: t.siVa),
            const SizedBox(height: 16),
            _Lista(titulo: 'No va', icono: Icons.cancel, color: Colors.red.shade600, items: t.noVa),
            const SizedBox(height: 16),
            Card(
              color: Theme.of(context).colorScheme.secondaryContainer,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(children: [
                  const Text('💡', style: TextStyle(fontSize: 28)),
                  const SizedBox(width: 12),
                  Expanded(child: Text(t.consejo, style: letra(17, FontWeight.w700))),
                ]),
              ),
            ),
            if (cuadro != null) ...[
              const SizedBox(height: 24),
              OutlinedButton.icon(
                onPressed: () => _corregir(context),
                icon: const Text('👎', style: TextStyle(fontSize: 20)),
                label: const Text('¿Me equivoqué?'),
                style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(56)),
              ),
              const SizedBox(height: 6),
              Text('Guarda la foto con el tacho correcto para mejorar el modelo.',
                  textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodySmall),
            ],
          ]),
        ),
      ]),
    );
  }

  Future<void> _corregir(BuildContext context) async {
    final correcto = await showModalBottomSheet<int>(
      context: context,
      showDragHandle: true,
      builder: (context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Text('¿En qué tacho va?', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 12),
            GridView.count(
              crossAxisCount: 4,
              shrinkWrap: true,
              mainAxisSpacing: 8,
              crossAxisSpacing: 8,
              children: [
                for (var i = 0; i < tachos.length; i++)
                  if (i != indice)
                    InkWell(
                      borderRadius: BorderRadius.circular(16),
                      onTap: () => Navigator.pop(context, i),
                      child: Ink(
                        decoration: BoxDecoration(color: tachos[i].fondo, borderRadius: BorderRadius.circular(16),
                            border: Border.all(color: Colors.black26)),
                        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                          Text(tachos[i].icono, style: const TextStyle(fontSize: 26)),
                          Text(tachos[i].color, style: letra(11, FontWeight.w800, tachos[i].texto)),
                        ]),
                      ),
                    ),
              ],
            ),
          ]),
        ),
      ),
    );
    if (correcto == null || !context.mounted) return;
    final ruta = await Almacen.guardarCorreccion(recuadroJpeg(cuadro!), correcto, indice);
    if (!context.mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text('¡Gracias! Foto guardada como ${tachos[correcto].color}.'),
      action: SnackBarAction(label: 'Ver ruta', onPressed: () => _mostrarRuta(context, ruta)),
    ));
  }

  void _mostrarRuta(BuildContext context, String ruta) => showDialog<void>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Foto guardada'),
          content: SelectableText(ruta),
          actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK'))],
        ),
      );
}

class _Lista extends StatelessWidget {
  final String titulo;
  final IconData icono;
  final Color color;
  final List<String> items;
  const _Lista({required this.titulo, required this.icono, required this.color, required this.items});

  @override
  Widget build(BuildContext context) => Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Icon(icono, color: color),
              const SizedBox(width: 8),
              Text(titulo, style: Theme.of(context).textTheme.titleMedium),
            ]),
            const SizedBox(height: 8),
            for (final item in items)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 3),
                child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('•  ', style: TextStyle(color: color, fontWeight: FontWeight.bold)),
                  Expanded(child: Text(item, style: Theme.of(context).textTheme.bodyLarge)),
                ]),
              ),
          ]),
        ),
      );
}
