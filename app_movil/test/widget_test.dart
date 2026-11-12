import 'package:basurita_app/tachos.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('7 tachos en el orden de las clases del modelo', () {
    expect(tachos.map((t) => t.color).toList(),
        ['AMARILLO', 'AZUL', 'BLANCO', 'GRIS', 'MARRÓN', 'NEGRO', 'ROJO']);
    expect(tachos.map((t) => t.carpeta).toList(),
        ['Amarillo', 'Azul', 'Blanco', 'Gris', 'Marron', 'Negro', 'Rojo']);
  });
}
