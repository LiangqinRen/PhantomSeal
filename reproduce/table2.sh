#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))

# Expected runtime: 1 min
PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="sample"