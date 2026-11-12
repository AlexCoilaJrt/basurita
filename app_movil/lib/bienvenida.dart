// Bienvenida: 3 pantallas la primera vez (y desde el boton de ayuda del escaner).
import 'package:flutter/material.dart';

import 'tachos.dart';
import 'tema.dart';

class Bienvenida extends StatefulWidget {
  final VoidCallback alTerminar;
  const Bienvenida({super.key, required this.alTerminar});

  @override
  State<Bienvenida> createState() => _BienvenidaState();
}

class _Pagina {
  final String icono, titulo, texto;
  const _Pagina(this.icono, this.titulo, this.texto);
}

const _paginas = [
  _Pagina('🌱', 'Bienvenido a Basurita',
      'Te dice en qué tacho va cada residuo, según los 7 colores de la NTP 900.058.'),
  _Pagina('📷', 'Apunta al residuo',
      'Pon un solo residuo dentro del recuadro, de cerca, con buena luz y sobre un fondo liso.'),
  _Pagina('✅', 'Espera el color y bótalo',
      'Cuando el recuadro se pinte y el celular vibre, ese es su tacho.\n\n'
          'La app usa la cámara solo para ver el residuo. Todo se procesa en tu celular, sin internet.'),
];

class _BienvenidaState extends State<Bienvenida> {
  final _paginador = PageController();
  int _pagina = 0;

  @override
  void dispose() {
    _paginador.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ultima = _pagina == _paginas.length - 1;
    final esquema = Theme.of(context).colorScheme;
    return Scaffold(
      body: SafeArea(
        child: Column(children: [
          Align(
            alignment: Alignment.centerRight,
            child: TextButton(onPressed: widget.alTerminar, child: const Text('Saltar')),
          ),
          Expanded(
            child: PageView.builder(
              controller: _paginador,
              itemCount: _paginas.length,
              onPageChanged: (p) => setState(() => _pagina = p),
              itemBuilder: (context, i) => Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                  Text(_paginas[i].icono, style: const TextStyle(fontSize: 96)),
                  const SizedBox(height: 24),
                  Text(_paginas[i].titulo, textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.headlineMedium),
                  const SizedBox(height: 16),
                  Text(_paginas[i].texto, textAlign: TextAlign.center, style: letra(18, FontWeight.w500)),
                  if (i == 0) ...[
                    const SizedBox(height: 28),
                    Wrap(spacing: 8, runSpacing: 8, alignment: WrapAlignment.center, children: [
                      for (final t in tachos)
                        Container(
                          width: 40,
                          height: 40,
                          alignment: Alignment.center,
                          decoration: BoxDecoration(
                              color: t.fondo, shape: BoxShape.circle, border: Border.all(color: Colors.black12)),
                          child: Text(t.icono, style: const TextStyle(fontSize: 20)),
                        ),
                    ]),
                  ],
                ]),
              ),
            ),
          ),
          Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            for (var i = 0; i < _paginas.length; i++)
              AnimatedContainer(
                duration: duracion,
                margin: const EdgeInsets.all(4),
                width: i == _pagina ? 24 : 8,
                height: 8,
                decoration: BoxDecoration(
                  color: i == _pagina ? esquema.primary : esquema.outlineVariant,
                  borderRadius: BorderRadius.circular(4),
                ),
              ),
          ]),
          Padding(
            padding: const EdgeInsets.all(24),
            child: FilledButton(
              style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(56)),
              onPressed: ultima
                  ? widget.alTerminar
                  : () => _paginador.nextPage(duration: duracion, curve: Curves.easeOutCubic),
              child: Text(ultima ? 'Empezar' : 'Siguiente', style: letra(18, FontWeight.w800)),
            ),
          ),
        ]),
      ),
    );
  }
}
