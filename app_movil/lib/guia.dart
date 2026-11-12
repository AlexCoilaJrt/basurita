// Guia de la NTP 900.058: los 7 tachos; al tocar uno se abre su ficha.
import 'package:flutter/material.dart';

import 'ficha.dart';
import 'tachos.dart';
import 'tema.dart';

class Guia extends StatelessWidget {
  const Guia({super.key});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: CustomScrollView(slivers: [
        SliverPadding(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
          sliver: SliverList.list(children: [
            Text('Guía de colores', style: Theme.of(context).textTheme.headlineMedium),
            const SizedBox(height: 4),
            Text('NTP 900.058 · Toca un tacho para ver qué va en él',
                style: Theme.of(context).textTheme.bodyLarge),
          ]),
        ),
        SliverPadding(
          padding: const EdgeInsets.all(16),
          sliver: SliverGrid.count(
            crossAxisCount: 2,
            mainAxisSpacing: 12,
            crossAxisSpacing: 12,
            childAspectRatio: 1.05,
            children: [
              for (var i = 0; i < tachos.length; i++)
                Material(
                  color: tachos[i].fondo,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(24), side: const BorderSide(color: Colors.black12)),
                  elevation: 1,
                  child: InkWell(
                    borderRadius: BorderRadius.circular(24),
                    onTap: () => Navigator.push(
                        context, MaterialPageRoute<void>(builder: (_) => Ficha(indice: i))),
                    child: Padding(
                      padding: const EdgeInsets.all(14),
                      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Hero(tag: 'icono_$i', child: Text(tachos[i].icono, style: const TextStyle(fontSize: 40))),
                        const Spacer(),
                        Text(tachos[i].color, style: letra(20, FontWeight.w900, tachos[i].texto)),
                        Text(tachos[i].residuo, maxLines: 2,
                            style: letra(14, FontWeight.w600, tachos[i].texto)),
                      ]),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ]),
    );
  }
}
