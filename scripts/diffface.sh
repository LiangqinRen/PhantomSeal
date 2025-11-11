#!/bin/bash

set -e

function=$1
function=`echo $function | tr '[:upper:]' '[:lower:]'`

ROOT=$(dirname $(dirname $(realpath "$0")))

if [[ $function == 'metric' || $function == 'validate' ]]
then
    PYTHONPATH="$ROOT" python -m src.diffface.main third_party=diffface third_party.function=$function
else
    echo "⚠️ Oops! That function doesn't exist"
fi