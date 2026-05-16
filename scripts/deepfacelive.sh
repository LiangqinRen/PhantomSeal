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
    PYTHONPATH="$ROOT" python -m src.deepfacelive.main \
        third_party=deepfacelive \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

if [[ $function == 'perturb' ]]
then
    run
else
    echo "⚠️ Oops! Function '$function' is not supported. Use: perturb"
fi
