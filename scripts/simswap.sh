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


if [[ $function == 'sample' ]]
then
    PYTHONPATH="$ROOT/src:$ROOT/third_party/SimSwap" python "$ROOT/src/simswap/main.py" third_party=simswap log_suffix=$debug third_party.function=$function
elif [[ $function == 'metric' ]]
then
    PYTHONPATH="$ROOT/src:$ROOT/third_party/SimSwap" python "$ROOT/src/simswap/main.py" third_party=simswap log_suffix=$debug third_party.function=$function third_party.dataset.metric_pairs=100 third_party.dataset.use_224=false
else
    echo "⚠️ Oops! That function doesn't exist"
fi