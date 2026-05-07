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
    PYTHONPATH="$ROOT" python -m src.denoiser.main \
        third_party=denoiser \
        evaluate=evaluate_local \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

multirun() {
    PYTHONPATH="$ROOT" python -m src.denoiser.main -m \
        third_party=denoiser \
        evaluate=evaluate_local \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

if [[ $function == 'train' ]]
then
    run \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    third_party.dataset.max_train_images='range(10,510,10)'
elif [[ $function == 'test' ]]
then
    run \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    third_party.test.checkpoint_path=checkpoints/denoiser/500.pth \
    third_party.test.max_images=2000 \
    evaluate.facepp.enable=true
else
    echo "⚠️ Oops! Function '$function' is not supported."
fi
