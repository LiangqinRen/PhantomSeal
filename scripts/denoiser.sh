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

checkpoint_sweep() {
    local paths=()
    local n

    for n in $(seq 10 10 500); do
        paths+=("checkpoints/denoiser/${n}.pth")
    done

    local IFS=,
    echo "${paths[*]}"
}

if [[ $function == 'train' ]]
then
    multirun \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    third_party.dataset.max_train_images='range(10,510,10)'
elif [[ $function == 'test' ]]
then
    checkpoint_paths=$(checkpoint_sweep)
    multirun \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    third_party.test.checkpoint_path="$checkpoint_paths" \
    third_party.test.max_images=2000
else
    echo "⚠️ Oops! Function '$function' is not supported."
fi
