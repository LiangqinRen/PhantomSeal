#!/bin/bash

set -e

function=$1
function=`echo $function | tr '[:upper:]' '[:lower:]'`

if [ $# == 2 ] && [ "$2" == "debug" ]; then
    suffix="-debug"
    log_level="debug"
else
    suffix=""
    log_level="info"
fi

ROOT=$(dirname $(dirname $(realpath "$0")))

if [[ $function == 'metric' ]]
then
    PYTHONPATH="$ROOT" python -m src.hififace.main third_party=hififace third_party.function=$function log.file_suffix=$suffix log.record_level=$log_level
else
    echo "⚠️ Oops! That function doesn't exist."
fi