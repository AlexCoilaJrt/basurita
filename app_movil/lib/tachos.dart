// Los 7 tachos de la NTP 900.058, en el mismo orden que las clases del modelo
// (amarillo, azul, blanco, gris, marron, negro, rojo).
import 'package:flutter/material.dart';

class Tacho {
  final String color; // "BLANCO"
  final String carpeta; // carpeta de "data basurita" en el Drive: "Blanco"
  final String residuo; // "Plásticos"
  final String icono;
  final Color fondo, texto;
  final List<String> siVa, noVa;
  final String consejo;
  const Tacho({
    required this.color,
    required this.carpeta,
    required this.residuo,
    required this.icono,
    required this.fondo,
    required this.texto,
    required this.siVa,
    required this.noVa,
    required this.consejo,
  });

  /// Color para barras sobre fondo claro (el blanco no se veria).
  Color get barra => fondo.computeLuminance() > 0.8 ? const Color(0xFFBDBDBD) : fondo;
}

const tachos = [
  Tacho(
    color: 'AMARILLO',
    carpeta: 'Amarillo',
    residuo: 'Metales',
    icono: '🥫',
    fondo: Color(0xFFFFD600),
    texto: Colors.black,
    siVa: ['Latas de bebida y de conserva', 'Tapas metálicas', 'Papel aluminio limpio', 'Chatarra pequeña'],
    noVa: ['Aerosoles y pilas (van al Rojo)', 'Latas con restos de comida'],
    consejo: 'Enjuaga las latas y aplástalas.',
  ),
  Tacho(
    color: 'AZUL',
    carpeta: 'Azul',
    residuo: 'Papel y cartón',
    icono: '📄',
    fondo: Color(0xFF1565C0),
    texto: Colors.white,
    siVa: ['Hojas, cuadernos y sobres', 'Periódicos y revistas', 'Cajas de cartón'],
    noVa: ['Papel higiénico o servilletas usadas (Negro)', 'Papel con grasa o comida (Negro)',
      'Vasos de papel plastificados (Negro)'],
    consejo: 'Dobla las cajas y mantenlo seco.',
  ),
  Tacho(
    color: 'BLANCO',
    carpeta: 'Blanco',
    residuo: 'Plásticos',
    icono: '🧴',
    fondo: Color(0xFFF5F5F5),
    texto: Colors.black,
    siVa: ['Botellas PET', 'Envases de shampoo y detergente', 'Tapas de plástico', 'Bolsas y láminas limpias'],
    noVa: ['Tecnopor (Negro)', 'Envolturas metalizadas de snacks (Negro)', 'Plástico con restos de comida'],
    consejo: 'Enjuaga las botellas y aplástalas.',
  ),
  Tacho(
    color: 'GRIS',
    carpeta: 'Gris',
    residuo: 'Vidrio',
    icono: '🍾',
    fondo: Color(0xFF9E9E9E),
    texto: Colors.black,
    siVa: ['Botellas de vidrio', 'Frascos', 'Envases de vidrio'],
    noVa: ['Focos y fluorescentes (Rojo)', 'Cerámica y porcelana (Negro)', 'Espejos'],
    consejo: 'Sin tapa y enjuagado. Cuidado con los bordes rotos.',
  ),
  Tacho(
    color: 'MARRÓN',
    carpeta: 'Marron',
    residuo: 'Orgánicos',
    icono: '🍎',
    fondo: Color(0xFF6D4C41),
    texto: Colors.white,
    siVa: ['Restos de frutas y verduras', 'Cáscaras y restos de comida', 'Hojas y ramas', 'Café y bolsitas de té'],
    noVa: ['Bolsas o envases (sepáralos)', 'Aceite usado (llévalo a un punto de acopio)'],
    consejo: 'Sin bolsas ni envases.',
  ),
  Tacho(
    color: 'NEGRO',
    carpeta: 'Negro',
    residuo: 'Generales (no aprovechables)',
    icono: '🗑️',
    fondo: Color(0xFF212121),
    texto: Colors.white,
    siVa: ['Papel higiénico y servilletas usadas', 'Tecnopor', 'Envolturas de snacks', 'Colillas', 'Cerámica rota'],
    noVa: ['Pilas y medicinas (Rojo)', 'Residuos reciclables limpios (usa su tacho)'],
    consejo: 'Si se puede reciclar, no va aquí.',
  ),
  Tacho(
    color: 'ROJO',
    carpeta: 'Rojo',
    residuo: 'Peligrosos',
    icono: '☣️',
    fondo: Color(0xFFC62828),
    texto: Colors.white,
    siVa: ['Pilas y baterías', 'Medicinas vencidas', 'Jeringas, guantes y mascarillas usadas', 'Aerosoles',
      'Focos, fluorescentes y termómetros'],
    noVa: ['Basura común (no llenes este tacho con otros residuos)'],
    consejo: 'Manipula con cuidado. Las jeringas, en un envase rígido.',
  ),
];
