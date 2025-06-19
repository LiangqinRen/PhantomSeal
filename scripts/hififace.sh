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

PYTHONPATH="$ROOT/src:$ROOT/third_party/HifiFace" python "$ROOT/src/hififace/main.py" third_party=hififace log_suffix=$debug third_party.function=$function