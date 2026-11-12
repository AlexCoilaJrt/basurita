// Basurita: clasifica residuos con la camara segun la NTP 900.058 (7 colores).
// El modelo es el clasificador del proyecto (modelos/clasificador.pt, v9 limpio) exportado a TFLite.
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'almacen.dart';
import 'bienvenida.dart';
import 'escaner.dart';
import 'guia.dart';
import 'historial.dart';
import 'tema.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  final vista = await Almacen.bienvenidaVista();
  runApp(Basurita(bienvenidaVista: vista));
}

class Basurita extends StatelessWidget {
  final bool bienvenidaVista;
  const Basurita({super.key, required this.bienvenidaVista});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Basurita',
        debugShowCheckedModeBanner: false,
        theme: tema(Brightness.light),
        darkTheme: tema(Brightness.dark),
        home: bienvenidaVista ? const Inicio() : const _PrimeraVez(),
      );
}

class _PrimeraVez extends StatelessWidget {
  const _PrimeraVez();

  @override
  Widget build(BuildContext context) => Bienvenida(alTerminar: () async {
        await Almacen.marcarBienvenida();
        if (!context.mounted) return;
        Navigator.pushReplacement(context, MaterialPageRoute<void>(builder: (_) => const Inicio()));
      });
}

class Inicio extends StatefulWidget {
  const Inicio({super.key});

  @override
  State<Inicio> createState() => _InicioState();
}

class _InicioState extends State<Inicio> {
  int _pestana = 0;

  void _ayuda() => Navigator.push(context,
      MaterialPageRoute<void>(builder: (context) => Bienvenida(alTerminar: () => Navigator.pop(context))));

  @override
  Widget build(BuildContext context) {
    // Solo la pestaña visible existe: al salir del escaner se apaga la camara
    final pantalla = switch (_pestana) {
      0 => Escaner(alPedirAyuda: _ayuda),
      1 => const Guia(),
      _ => const Historial(),
    };
    return Scaffold(
      body: AnimatedSwitcher(duration: duracion, child: KeyedSubtree(key: ValueKey(_pestana), child: pantalla)),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _pestana,
        onDestinationSelected: (i) => setState(() => _pestana = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.center_focus_weak), selectedIcon: Icon(Icons.center_focus_strong),
              label: 'Escanear'),
          NavigationDestination(icon: Icon(Icons.grid_view_outlined), selectedIcon: Icon(Icons.grid_view_rounded),
              label: 'Guía'),
          NavigationDestination(icon: Icon(Icons.bar_chart_outlined), selectedIcon: Icon(Icons.bar_chart),
              label: 'Historial'),
        ],
      ),
    );
  }
}
