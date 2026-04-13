#!/bin/bash

set -e

function=$1
function=`echo $function | tr '[:upper:]' '[:lower:]'`

log_level="info"
suffix=""
if [ $# == 2 ] && [ "$2" == "debug" ]; then
    suffix="-debug"
fi

ROOT=$(dirname $(dirname $(realpath "$0")))

run() {
    PYTHONPATH="$ROOT" python -m src.diffswap.main \
        third_party=diffswap \
        evaluate=evaluate_local \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

multirun() {
    PYTHONPATH="$ROOT" python -m src.diffswap.main -m \
        third_party=diffswap \
        evaluate=evaluate_local \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

if [[ $function == 'swap' || $function == 'sample' || $function == 'metric' ]]
then
    if [[ $function == 'swap' ]]; then
        run \
        evaluate.effectiveness.perturb=false \
        evaluate.effectiveness.ASRo=true \
        evaluate.effectiveness.ASRp=false \
        evaluate.effectiveness.TSR=false
    else
        run
    fi
else
    echo "⚠️ Oops! Function '$function' is not supported."
fi
