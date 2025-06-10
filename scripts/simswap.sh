#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))

PYTHONPATH="$ROOT/src:$ROOT/third_party/SimSwap" python "$ROOT/src/defense/simswap.py" third_party=simswap
