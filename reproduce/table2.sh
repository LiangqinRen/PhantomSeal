#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))
source "$ROOT/reproduce/common.sh"
check_experiment_args "$@"

EXPERIMENTS=(1)
init_experiments "${1:-}" "${EXPERIMENTS[@]}"

# Experiment 1
if should_run_experiment 1; then
    announce_experiment 1
    # Expected runtime: 1 min
    PYTHONPATH="$ROOT" python -m src.simswap.main \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="sample"
fi
