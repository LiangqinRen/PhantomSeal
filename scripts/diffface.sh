#!/bin/bash

set -e

function=$1
function=$(echo "$function" | tr '[:upper:]' '[:lower:]')

log_level="info"
suffix=""
if [ $# == 2 ] && [ "$2" == "debug" ]; then
    suffix="-debug"
fi

ROOT=$(dirname $(dirname $(realpath "$0")))

run() {
    PYTHONPATH="$ROOT" python -m src.diffface.main \
        third_party=diffface \
        evaluate=evaluate_local \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

multirun() {
    PYTHONPATH="$ROOT" python -m src.diffface.main -m \
        third_party=diffface \
        evaluate=evaluate_local \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

if [[ $function == 'swap' || $function == 'sample' || $function == 'metric' || $function == 'lowkey' ]]
then
    if [[ $function == 'swap' ]]; then
        run \
        third_party.dataset.batch_size=10 \
        third_party.dataset.metric_pairs=10 \
        evaluate.effectiveness.perturb=false \
        evaluate.effectiveness.ASRp=false
    else
        run \
        evaluate.effectiveness.perturb=false
    fi
else
    echo "Oops! Function '$function' is not supported."
fi
