#!/bin/bash

set -e

function=$1
function=`echo $function | tr '[:upper:]' '[:lower:]'`

log_level="info"
suffix=""
if [ $# == 2 ] && [ "$2" == "debug" ]; then
    suffix="-debug"
fi

ROOT=$(dirname "$(dirname "$(realpath "$0")")")

valid_functions=("extract" "train" "test" "metric")

if [[ " ${valid_functions[@]} " =~ " ${function} " ]]; then
    PYTHONPATH="$ROOT" python -m src.faceswap.main \
        third_party=faceswap \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level"
else
    echo "⚠️ Oops! Function '$function' is not supported."
fi