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
    PYTHONPATH="$ROOT" python -m src.simswap.main \
        third_party=simswap \
        third_party.function="$function" \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

multirun() {
    PYTHONPATH="$ROOT" python -m src.simswap.main -m \
        third_party=simswap \
        third_party.function=metric \
        log.file_suffix="$suffix" \
        log.record_level="$log_level" \
        "$@"
}

if [[ $function == 'sample' ]]
then
    run
elif [[ $function == 'metric' ]]
then
    run
elif [[ $function == 'ai_beauty' ]]
then
    run \
    third_party.robustness.ai_beauty=true \
    third_party.robustness.ai_beauty_tool=ai_lab_tools # ai_lab_tools or tencent_cloud
elif [[ $function == 'failure_tracing' ]]
then
    run \ 
    third_party.defense.failure_defense_tracing=true
elif [[ $function == 'protection_robustness_sample' ]]
then
    run \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    third_party.defense.epochs=335
elif [[ $function == 'protection_robustness_metric' ]]
then
    run \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    third_party.defense.epochs=335
elif [[ $function == 'forensics_robustness_sample' ]]
then
    run \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    third_party.defense.epochs=335
elif [[ $function == 'forensics_robustness_metric' ]]
then
    run \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    third_party.defense.epochs=335
elif [[ $function == 'image_robustness_metric' ]]
then
    run \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false
elif [[ $function == 'adaptive_attack_with_self_image' ]]
then
    run \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false
elif [[ $function == 'adaptive_attack_with_other_image' ]]
then
    run \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false
else
    echo "⚠️ Oops! That function doesn't exist."
fi