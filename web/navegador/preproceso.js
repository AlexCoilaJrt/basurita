// Prepara una imagen igual que Ultralytics para YOLO11n-cls (imgsz=224):
//   1) escala con PIL BILINEAR hasta que el lado corto mida 224 (torchvision Resize(224))
//   2) recorta el centro 224x224 (torchvision CenterCrop)
//   3) RGB 0..1 en formato NCHW (ToTensor; media 0 y desviacion 1)
// Se reproduce el filtro de PIL (triangulo con soporte escalado, dos pasadas con redondeo a 8 bits)
// para que el navegador de el mismo resultado que web/app.py y que modelos/clasificador.pt.
(function (global) {
  "use strict";
  const LADO = 224;

  // round() de Python: los empates van al par
  function redondeoPython(x) {
    const f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }

  // Coeficientes de PIL (Resample.c, precompute_coeffs) para pasar de "entrada" a "salida" pixeles.
  // Solo se calculan las posiciones de salida desde "desde" (inclusive) hasta "hasta" (exclusivo).
  function coeficientes(entrada, salida, desde, hasta) {
    const escala = entrada / salida;
    const escalaFiltro = Math.max(escala, 1);
    const soporte = escalaFiltro; // BILINEAR: soporte 1
    const lista = [];
    for (let i = desde; i < hasta; i++) {
      const centro = (i + 0.5) * escala;
      const xmin = Math.max(Math.trunc(centro - soporte + 0.5), 0);
      const xmax = Math.min(Math.trunc(centro + soporte + 0.5), entrada);
      const pesos = [];
      let suma = 0;
      for (let x = xmin; x < xmax; x++) {
        const t = Math.abs((x - centro + 0.5) / escalaFiltro);
        const w = t < 1 ? 1 - t : 0;
        pesos.push(w);
        suma += w;
      }
      for (let k = 0; k < pesos.length; k++) pesos[k] = suma ? pesos[k] / suma : 0;
      lista.push({ xmin, pesos });
    }
    return lista;
  }

  function a8bits(v) {
    const r = Math.round(v);
    return r < 0 ? 0 : r > 255 ? 255 : r;
  }

  /**
   * rgba: Uint8ClampedArray (ImageData.data) de una imagen de ancho w y alto h.
   * region: {x, y, lado_x, lado_y} parte de la imagen a usar (toda, o el recuadro central).
   * Devuelve Float32Array(3*224*224) en NCHW.
   */
  function prepararTensor(rgba, w, h, region) {
    const r = region || { x: 0, y: 0, ancho: w, alto: h };
    const rw = r.ancho, rh = r.alto;
    // 1) tamano tras Resize(224): el lado corto pasa a 224, el largo se trunca como en torchvision
    let nw, nh;
    if (rw <= rh) { nw = LADO; nh = Math.trunc(LADO * rh / rw); } else { nh = LADO; nw = Math.trunc(LADO * rw / rh); }
    // 2) CenterCrop: solo se calculan las filas y columnas que quedan dentro del recorte
    const arriba = redondeoPython((nh - LADO) / 2), izquierda = redondeoPython((nw - LADO) / 2);
    const cx = coeficientes(rw, nw, izquierda, izquierda + LADO);
    const cy = coeficientes(rh, nh, arriba, arriba + LADO);
    // Filas de entrada que necesita la pasada vertical
    const fila0 = cy[0].xmin;
    const filaN = cy[LADO - 1].xmin + cy[LADO - 1].pesos.length;
    // Pasada horizontal (redondeada a 8 bits, como PIL)
    const horiz = new Uint8Array((filaN - fila0) * LADO * 3);
    for (let yy = fila0; yy < filaN; yy++) {
      const base = ((yy + r.y) * w + r.x) * 4;
      const fuera = (yy - fila0) * LADO * 3;
      for (let i = 0; i < LADO; i++) {
        const { xmin, pesos } = cx[i];
        let sr = 0, sg = 0, sb = 0;
        for (let k = 0; k < pesos.length; k++) {
          const p = base + (xmin + k) * 4, q = pesos[k];
          sr += rgba[p] * q; sg += rgba[p + 1] * q; sb += rgba[p + 2] * q;
        }
        horiz[fuera + i * 3] = a8bits(sr);
        horiz[fuera + i * 3 + 1] = a8bits(sg);
        horiz[fuera + i * 3 + 2] = a8bits(sb);
      }
    }
    // Pasada vertical y salida NCHW 0..1
    const area = LADO * LADO;
    const salida = new Float32Array(3 * area);
    for (let j = 0; j < LADO; j++) {
      const { xmin, pesos } = cy[j];
      for (let i = 0; i < LADO; i++) {
        let sr = 0, sg = 0, sb = 0;
        for (let k = 0; k < pesos.length; k++) {
          const p = ((xmin + k - fila0) * LADO + i) * 3, q = pesos[k];
          sr += horiz[p] * q; sg += horiz[p + 1] * q; sb += horiz[p + 2] * q;
        }
        const o = j * LADO + i;
        salida[o] = a8bits(sr) / 255;
        salida[area + o] = a8bits(sg) / 255;
        salida[2 * area + o] = a8bits(sb) / 255;
      }
    }
    return salida;
  }

  /** Recuadro central: lado = fraccion del lado corto (igual que la app movil y web/app.py). */
  function recuadroCentral(w, h, fraccion) {
    const lado = redondeoPython(fraccion * Math.min(w, h));
    return { x: Math.floor((w - lado) / 2), y: Math.floor((h - lado) / 2), ancho: lado, alto: lado };
  }

  const api = { LADO, prepararTensor, recuadroCentral };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else global.Preproceso = api;
})(typeof window !== "undefined" ? window : globalThis);
