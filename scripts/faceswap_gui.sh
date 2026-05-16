#!/bin/bash

set -e

function=$1
function=`echo $function | tr '[:upper:]' '[:lower:]'`

log_level="info"
suffix=""
if [ "$2" == "debug" ]; then
    suffix="-debug"
    log_level="debug"
fi
if [ "$2" == "debug" ]; then
    extra_args=("${@:3}")
else
    extra_args=("${@:2}")
fi

ROOT=$(dirname "$(dirname "$(realpath "$0")")")

run() {
    PYTHONPATH="$ROOT" python -m src.faceswap_gui.main \
        third_party=faceswap_gui \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

if [[ $function == 'perturb' ]]
then
    run \
        third_party.dataset.input_dir=/home/liangqinren/PhantomSeal/data/faceswap_gui/1_256 \
        "${extra_args[@]}"
else
    echo "⚠️ Oops! Function '$function' is not supported. Use: perturb"
fi
