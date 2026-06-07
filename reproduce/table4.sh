#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))
source "$ROOT/reproduce/common.sh"
check_experiment_args "$@"

EXPERIMENTS=(1 2 3 4 5 6 7 8 9 10)
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
    # Expected runtime: 2.5 minutes per batch, 125 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        third_party.dataset.cloak_dir=data/stylegan3_cloak
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
        third_party.dataset.cloak_mix=false
fi

# Experiment 4
if should_run_experiment 4; then
    announce_experiment 4
    # Expected runtime: 2.5 minutes per batch, 125 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        third_party.dataset.cloak_distance=1.35
fi

# Experiment 5
if should_run_experiment 5; then
    announce_experiment 5
    # Expected runtime: 2.5 minutes per batch, 125 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main -m \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        third_party.dataset.cloak_distance=1.04,1.25,1.50,1.75,2.00
fi

# Experiment 6
if should_run_experiment 6; then
    announce_experiment 6
    # Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
    PYTHONPATH="$ROOT" python -m src.diffface.main \
        third_party=diffface \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=30
fi

# Experiment 7
if should_run_experiment 7; then
    announce_experiment 7
    # Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
    PYTHONPATH="$ROOT" python -m src.diffface.main \
        third_party=diffface \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=30 \
        third_party.dataset.cloak_dir=data/stylegan3_cloak
fi

# Experiment 8
if should_run_experiment 8; then
    announce_experiment 8
    # Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
    PYTHONPATH="$ROOT" python -m src.diffface.main \
        third_party=diffface \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=30 \
        third_party.dataset.cloak_mix=false
fi

# Experiment 9
if should_run_experiment 9; then
    announce_experiment 9
    # Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
    PYTHONPATH="$ROOT" python -m src.diffface.main \
        third_party=diffface \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=30 \
        third_party.dataset.cloak_distance=1.55
fi

# Experiment 10
if should_run_experiment 10; then
    announce_experiment 10
    # Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
    PYTHONPATH="$ROOT" python -m src.diffface.main -m \
        third_party=diffface \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=30 \
        third_party.dataset.cloak_distance=1.04,1.25,1.50,1.75,2.00
fi
