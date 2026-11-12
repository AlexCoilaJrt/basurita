// Tema de la app: neutro (para que resalten los 7 colores de la NTP), modo claro y oscuro, fuente Nunito.
import 'package:flutter/material.dart';

const verde = Color(0xFF2E7D32);
const duracion = Duration(milliseconds: 250);

/// Nunito es una fuente variable: el grosor se pide con FontVariation ademas de fontWeight.
TextStyle letra(double tamano, FontWeight peso, [Color? color]) => TextStyle(
      fontFamily: 'Nunito',
      fontSize: tamano,
      fontWeight: peso,
      fontVariations: [FontVariation('wght', peso.value.toDouble())],
      color: color,
    );

TextTheme _conPesos(TextTheme t) {
  TextStyle? p(TextStyle? s) => s?.copyWith(
      fontFamily: 'Nunito',
      fontVariations: [FontVariation('wght', (s.fontWeight ?? FontWeight.w400).value.toDouble())]);
  return t.copyWith(
    displayLarge: p(t.displayLarge), displayMedium: p(t.displayMedium), displaySmall: p(t.displaySmall),
    headlineLarge: p(t.headlineLarge?.copyWith(fontWeight: FontWeight.w800)),
    headlineMedium: p(t.headlineMedium?.copyWith(fontWeight: FontWeight.w800)),
    headlineSmall: p(t.headlineSmall?.copyWith(fontWeight: FontWeight.w800)),
    titleLarge: p(t.titleLarge?.copyWith(fontWeight: FontWeight.w800)),
    titleMedium: p(t.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
    titleSmall: p(t.titleSmall?.copyWith(fontWeight: FontWeight.w700)),
    bodyLarge: p(t.bodyLarge), bodyMedium: p(t.bodyMedium), bodySmall: p(t.bodySmall),
    labelLarge: p(t.labelLarge?.copyWith(fontWeight: FontWeight.w700)),
    labelMedium: p(t.labelMedium?.copyWith(fontWeight: FontWeight.w700)),
    labelSmall: p(t.labelSmall),
  );
}

ThemeData tema(Brightness brillo) {
  final base = ThemeData(
    useMaterial3: true,
    fontFamily: 'Nunito',
    colorScheme: ColorScheme.fromSeed(seedColor: verde, brightness: brillo),
  );
  return base.copyWith(
    textTheme: _conPesos(base.textTheme),
    cardTheme: const CardThemeData(margin: EdgeInsets.zero, clipBehavior: Clip.antiAlias),
  );
}
