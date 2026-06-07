#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))
source "$ROOT/reproduce/common.sh"
check_experiment_args "$@"

EXPERIMENTS=(1 2 3 4)
init_experiments "${1:-}" "${EXPERIMENTS[@]}"

# Experiment 1
if should_run_experiment 1; then
    announce_experiment 1
    # Expected runtime per parameter setting: 2.5 minutes per batch, 125 minutes in total. This multirun evaluates 15 parameter settings, 1,875 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main -m \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        third_party.defense.weight.perturb=0,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000
fi

# Experiment 2
if should_run_experiment 2; then
    announce_experiment 2
    # Expected runtime per parameter setting: 2.5 minutes per batch, 125 minutes in total. This multirun evaluates 15 parameter settings, 1,875 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main -m \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        third_party.defense.weight.identity=0,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000
fi

# Experiment 3
if should_run_experiment 3; then
    announce_experiment 3
    # Expected runtime per parameter setting: 2.5 minutes per batch, 125 minutes in total. This multirun evaluates 15 parameter settings, 1,875 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main -m \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        third_party.defense.weight.cloak=0,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000
fi

# Experiment 4
if should_run_experiment 4; then
    announce_experiment 4
    # Expected runtime per parameter setting: 2.5 minutes per batch, 125 minutes in total. This multirun evaluates 15 parameter settings, 1,875 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main -m \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        third_party.defense.weight.context=0,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000
fi
