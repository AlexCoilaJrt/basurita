#!/bin/bash
# Inicia el detector en la Raspberry Pi (antes: bash instalar_pi.sh).
#   bash iniciar_pi.sh            modo escaner: solo el clasificador (rapido, poca memoria)
#   bash iniciar_pi.sh --normal   YOLO-World + clasificador (lento; mejor con 4 GB o mas)
# Se pueden agregar opciones del detector, por ejemplo: bash iniciar_pi.sh --camara 1
cd "$(dirname "$0")"
source .venv/bin/activate

if [ "$1" = "--normal" ]; then
    shift
    python src/detector_residuos.py --sin-plan-b --imgsz 416 "$@"
else
    python src/detector_residuos.py --escaner "$@"
fi
