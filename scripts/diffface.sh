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

if [[ $function == 'metric' || $function == 'validate' ]]
then
    PYTHONPATH="$ROOT" python -m src.diffface.main \
    third_party=diffface \
    third_party.function=$function \
    log.file_suffix=$suffix \
    log.record_level=$log_level
else
    echo "⚠️ Oops! That function doesn't exist."
fi