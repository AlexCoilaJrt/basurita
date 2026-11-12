"""
Verifica si un dataset que bajaste de Roboflow es EXACTAMENTE el que uso el docente,
comparando los nombres de archivo (incluido el codigo despues de '.rf.') con la lista del .rar.

Uso:
    python verificar_examen.py --lista nombres_examen_docente.csv --carpeta "ruta/al/dataset_descargado"
"""
import argparse
import csv
import os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--lista", default="nombres_examen_docente.csv")
ap.add_argument("--carpeta", required=True, help="carpeta descomprimida del dataset de Roboflow")
args = ap.parse_args()

esperados = defaultdict(set)   # dataset -> nombres
clase_de = {}
with open(args.lista, encoding="utf-8") as f:
    for fila in csv.DictReader(f):
        esperados[fila["dataset"]].add(fila["archivo"])
        clase_de[fila["archivo"]] = (fila["dataset"], fila["clase_gt"])

encontrados = {}   # nombre -> (split, carpeta de clase)
for raiz, _, archivos in os.walk(args.carpeta):
    for a in archivos:
        partes = os.path.relpath(raiz, args.carpeta).replace("\\", "/").split("/")
        split = next((p for p in partes if p in ("train", "valid", "test")), "?")
        encontrados[a] = (split, os.path.basename(raiz))

print(f"Archivos en la carpeta: {len(encontrados)}\n")
for ds, nombres in esperados.items():
    hallados = nombres & set(encontrados)
    print(f"{ds}: {len(hallados)} de {len(nombres)} imagenes del examen encontradas")
    if hallados:
        splits = defaultdict(int)
        for n in hallados:
            splits[encontrados[n][0]] += 1
        print(f"   en que split estan: {dict(splits)}")
        mal = [n for n in hallados if encontrados[n][1].lower() != clase_de[n][1].lower()]
        if mal:
            print(f"   AVISO: {len(mal)} estan en otra carpeta de clase (ej. {mal[0]})")
        if len(hallados) == len(nombres) and set(splits) == {"test"}:
            print("   ==> ES EL MISMO EXAMEN. Copia su carpeta test dentro de examen/")
    print()
