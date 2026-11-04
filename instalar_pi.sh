#!/bin/bash
# Instala el detector en una Raspberry Pi 4/5 con Raspberry Pi OS de 64 bits.
# Uso (dentro de la carpeta del proyecto):  bash instalar_pi.sh
set -e
cd "$(dirname "$0")"

if [ "$(uname -m)" != "aarch64" ]; then
    echo "ERROR: se necesita Raspberry Pi OS de 64 bits (este sistema es $(uname -m))."
    echo "Graba 'Raspberry Pi OS (64-bit)' con Raspberry Pi Imager."
    exit 1
fi

echo "== 1/4 Paquetes del sistema =="
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip libgl1

echo "== 2/4 Entorno de Python ($(python3 --version)) =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "== 3/4 Librerias (tarda unos 15-30 minutos) =="
pip install -r requirements.txt

echo "== 4/4 Prueba del modelo =="
python -c "
import numpy as np
from ultralytics import YOLO
r = YOLO('modelos/clasificador.pt')(np.zeros((224, 224, 3), np.uint8), verbose=False)[0]
print('Clasificador OK:', len(r.names), 'clases')
"

memoria=$(free -m | awk '/^Mem:/ {print $2}')
echo
echo "Instalacion terminada. Memoria de la Pi: ${memoria} MB."
echo "Para iniciar:  bash iniciar_pi.sh            (modo escaner, recomendado)"
echo "               bash iniciar_pi.sh --normal   (YOLO-World + clasificador, lento)"
if [ "$memoria" -lt 3500 ]; then
    echo "Con menos de 4 GB de memoria usa solo el modo escaner."
fi
