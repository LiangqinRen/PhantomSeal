#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))
source "$ROOT/reproduce/common.sh"
check_experiment_args "$@"

EXPERIMENTS=(1 2 3 4 5)
init_experiments "${1:-}" "${EXPERIMENTS[@]}"

# Experiment 1
if should_run_experiment 1; then
    announce_experiment 1
    # Expected runtime: 2.5 minutes per batch, 125 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60
fi

# Experiment 2
if should_run_experiment 2; then
    announce_experiment 2
    # Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        third_party.defense.limit.R=1 \
        third_party.defense.limit.G=1 \
        third_party.defense.limit.B=1
fi

# Experiment 3
if should_run_experiment 3; then
    announce_experiment 3
    # Expected runtime: 2.5 minutes per batch, 125 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        third_party.defense.limit.identity=1000000
fi

# Experiment 4
if should_run_experiment 4; then
    announce_experiment 4
    # Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        third_party.defense.limit.context=1000000
fi

# Experiment 5
if should_run_experiment 5; then
    announce_experiment 5
    # Expected runtime: 2.5 minutes per batch, 125 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        third_party.defense.limit.R=1 \
        third_party.defense.limit.G=1 \
        third_party.defense.limit.B=1 \
        third_party.defense.limit.identity=1000000 \
        third_party.defense.limit.context=1000000
fi
