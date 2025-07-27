#!/bin/bash

set -e

function=$1
function=$(echo "$function" | tr '[:upper:]' '[:lower:]')

if [ $# == 2 ] && [ "$2" == "debug" ]; then
    suffix="-debug"
    log_level="debug"
else
    suffix=""
    log_level="info"
fi

ROOT=$(dirname "$(dirname "$(realpath "$0")")")

valid_functions=("extract" "train" "test" "metric")

if [[ " ${valid_functions[@]} " =~ " ${function} " ]]; then
    PYTHONPATH="$ROOT/src" python "$ROOT/src/faceswap/main.py" \
        third_party=faceswap \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level"
else
    echo "⚠️ Oops! Function '$function' is not supported."
fi