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
    # Expected runtime: 3.5 minutes per batch, 175 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="forensics_robustness_metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        evaluate.effectiveness.perturb=false \
        evaluate.effectiveness.ASRo=false \
        evaluate.effectiveness.ASRp=false \
        third_party.defense.epochs=335
fi

# Experiment 2
if should_run_experiment 2; then
    announce_experiment 2
    # Expected runtime: 3.5 minutes per batch, 175 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="forensics_robustness_metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        evaluate.effectiveness.perturb=false \
        evaluate.effectiveness.ASRo=false \
        evaluate.effectiveness.ASRp=false \
        third_party.defense.epochs=335 \
        third_party.dataset.cloak_distance=1.04
fi

# Experiment 3
if should_run_experiment 3; then
    announce_experiment 3
    # Expected runtime: 1 minutes per batch, 30 minutes in total.
    PYTHONPATH="$ROOT" python -m src.artificialfingerprint.main \
        third_party=artificialfingerprint \
        third_party.function="forensics_robustness_metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=100
fi

# Experiment 4
if should_run_experiment 4; then
    announce_experiment 4
    # Expected runtime: 0.5 minutes per batch, 15 minutes in total.
    PYTHONPATH="$ROOT" python -m src.sepmark.main \
        third_party=sepmark \
        third_party.function="forensics_robustness_metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=100
fi
