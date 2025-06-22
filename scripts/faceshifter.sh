#!/bin/bash

set -e

function=$1
function=`echo $function | tr '[:upper:]' '[:lower:]'`

if [ $# == 2 ] && [ "$2" == "debug" ]; then
    debug="_debug"
else
    debug=""
fi

ROOT=$(dirname $(dirname $(realpath "$0")))

if [[ $function == 'metric' ]]
then
    PYTHONPATH="$ROOT/src:$ROOT/third_party/FaceShifter/ModelC:$ROOT/third_party/FaceShifter/ModelC/face_modules" python "$ROOT/src/faceshifter/main.py" third_party=faceshifter log_suffix=$debug third_party.function=$function
else
    echo "⚠️ Oops! That function doesn't exist"
fi