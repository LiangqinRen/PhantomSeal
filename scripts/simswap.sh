#!/bin/bash

set -e

function=$1
function=`echo $function | tr '[:upper:]' '[:lower:]'`

if [ $# == 2 ] && [ "$2" == "debug" ]; then
    debug="_debug"
else
    debug=""
fi

ROOT=$(dirname $(dirname $(realpath "$0")))


if [[ $function == 'sample' ]]
then
    PYTHONPATH="$ROOT/src:$ROOT/third_party/SimSwap" python "$ROOT/src/simswap/main.py" third_party=simswap log_suffix=$debug third_party.function=$function
elif [[ $function == 'metric' ]]
then
    PYTHONPATH="$ROOT/src:$ROOT/third_party/SimSwap" python "$ROOT/src/simswap/main.py" third_party=simswap log_suffix=$debug third_party.function=$function
elif [[ $function == 'ai_beauty' ]]
then
    PYTHONPATH="$ROOT/src:$ROOT/third_party/SimSwap" python "$ROOT/src/simswap/main.py" third_party=simswap log_suffix=$debug third_party.function=metric third_party.robustness.ai_beauty=true third_party.robustness.ai_beauty_tool=ai_lab_tools
    # third_party.robustness.ai_beauty_tool=ai_lab_tools or tencent_cloud
else
    echo "⚠️ Oops! That function doesn't exist"
fi