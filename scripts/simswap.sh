#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))

PYTHONPATH="$ROOT/src:$ROOT/third_party/SimSwap" python "$ROOT/src/simswap/main.py" third_party=simswap
