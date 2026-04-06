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
    PYTHONPATH="$ROOT" python -m src.blackbox.main \
        third_party=blackbox \
        evaluate=evaluate_local \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

multirun() {
    PYTHONPATH="$ROOT" python -m src.blackbox.main -m \
        third_party=blackbox \
        evaluate=evaluate_local \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

if [[ $function == 'metric' ]]
then
    # run \
    # third_party.defense.target=uniface \
    # third_party.defense.batch_size=60

    # run \
    # third_party.defense.target=infoswap \
    # third_party.defense.batch_size=20

    run \
    third_party.defense.target=e4s \
    third_party.defense.batch_size=20
else
    echo "⚠️ Oops! Function '$function' is not supported."
fi
