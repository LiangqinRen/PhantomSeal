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

if [[ $function == 'sample' ]]
then
    PYTHONPATH="$ROOT" python -m src.simswap.main third_party=simswap third_party.function=$function log.file_suffix=$suffix log.record_level=$log_level
elif [[ $function == 'metric' || $function == 'adaptive_attack' || $function == 'adaptive_attack_self' ]]
then
    PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    third_party.function=$function \
    log.file_suffix=$suffix \
    log.record_level=$log_level
elif [[ $function == 'ai_beauty' ]]
then
    PYTHONPATH="$ROOT" python -m src.simswap.main third_party=simswap third_party.function=metric log.file_suffix=$suffix log.record_level=$log_level third_party.robustness.ai_beauty=true third_party.robustness.ai_beauty_tool=ai_lab_tools 
    # third_party.robustness.ai_beauty_tool=ai_lab_tools or tencent_cloud
elif [[ $function == 'failure_tracing' ]]
then
    PYTHONPATH="$ROOT" python -m src.simswap.main -m third_party=simswap third_party.function=metric log.file_suffix=$suffix log.record_level=$log_level third_party.defense.failure_defense_tracing=true evaluate.facepp.use=false third_party.dataset.cloak_index=0,10,20,30
    PYTHONPATH="$ROOT" python -m src.simswap.main -m third_party=simswap third_party.function=metric log.file_suffix=$suffix log.record_level=$log_level third_party.defense.failure_defense_tracing=true evaluate.facenet_512.use=false third_party.dataset.cloak_index=0,10,20,30
elif [[ $function == 'robustness_sample' || $function == 'robustness_metric' ]]
then
    PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap third_party.function=$function \
    log.file_suffix=$suffix log.record_level=$log_level \
    evaluate.effectiveness.perturb=false evaluate.effectiveness.ASRo=false \
    evaluate.effectiveness.TSR=false \
    third_party.defense.epochs=335
elif [[ $function == 'robustness_forensics_sample' || $function == 'robustness_forensics_metric' ]]
then
    PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap third_party.function=$function \
    log.file_suffix=$suffix log.record_level=$log_level \
    evaluate.effectiveness.perturb=false evaluate.effectiveness.ASRo=false \
    third_party.defense.epochs=335
elif [[ $function == 'image_robustness_metric' ]]
then
    PYTHONPATH="$ROOT" python -m src.simswap.main third_party=simswap third_party.function=$function log.file_suffix=$suffix log.record_level=$log_level third_party.defense.epochs=1000
elif [[ $function == 'lowkey' ]]
then
    PYTHONPATH="$ROOT" python -m src.simswap.main -m third_party=simswap third_party.function=$function log.file_suffix=$suffix log.record_level=$log_level third_party.lowkey.weight.identity='range(31000,35000,1000)'
elif [[ $function == 'misclassify' ]]
then
    PYTHONPATH="$ROOT:$ROOT/src:$ROOT/third_party/SimSwap" python "$ROOT/src/simswap/main.py" third_party=simswap third_party.function=$function log.file_suffix=$suffix log.record_level=$log_level
else
    echo "⚠️ Oops! That function doesn't exist."
fi