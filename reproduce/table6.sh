#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))
source "$ROOT/reproduce/common.sh"
check_experiment_args "$@"

EXPERIMENTS=(1 2)
init_experiments "${1:-}" "${EXPERIMENTS[@]}"

# Experiment 1
if should_run_experiment 1; then
    announce_experiment 1
    # Expected runtime: 9 minutes per batch, 2700 minutes in total.
    PYTHONPATH="$ROOT" python -m src.blackbox.main \
        third_party=blackbox \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=10 \
        evaluate.effectiveness.perturb=false \
        protection=phantomseal
fi

# Experiment 2
if should_run_experiment 2; then
    announce_experiment 2
    # Expected runtime: 8 minutes per batch, 2400 minutes in total.
    PYTHONPATH="$ROOT" python -m src.blackbox.main \
        third_party=blackbox \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=10 \
        evaluate.effectiveness.perturb=false \
        protection=nullswap
fi
