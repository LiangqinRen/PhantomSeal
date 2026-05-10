#!/bin/bash

set -e

function=$1
function=`echo $function | tr '[:upper:]' '[:lower:]'`

log_level="info"
suffix=""
if [ $# == 2 ] && [ "$2" == "debug" ]; then
    suffix="-debug"
    log_level="debug"
    extra_args=()
elif [ $# -ge 2 ] && [ "$2" == "debug" ]; then
    suffix="-debug"
    log_level="debug"
    extra_args=("${@:3}")
else
    extra_args=("${@:2}")
fi

ROOT=$(dirname $(dirname $(realpath "$0")))

run() {
    PYTHONPATH="$ROOT" python -m src.cuma.main \
        third_party=cuma \
        evaluate=evaluate_local \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

if [[ $function == 'metric' ]]
then
    run "${extra_args[@]}"
else
    echo "Function '$function' is not supported."
fi
