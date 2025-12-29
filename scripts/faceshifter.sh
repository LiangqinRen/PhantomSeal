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
    PYTHONPATH="$ROOT" python -m src.faceshifter.main \
        third_party=faceshifter \
        evaluate=evaluate_local \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

multirun() {
    PYTHONPATH="$ROOT" python -m src.faceshifter.main -m \
        third_party=faceshifter \
        evaluate=evaluate_local \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

if [[ $function == 'metric' ]]
then
    run
else
    echo "⚠️ Oops! That function doesn't exist."
fi