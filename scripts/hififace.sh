#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))

PYTHONPATH="$ROOT/src:$ROOT/third_party/HifiFace" python "$ROOT/src/defense/hififace.py" third_party=hififace