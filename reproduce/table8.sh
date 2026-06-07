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
    # Expected runtime: 2.5 minutes per batch, 125 minutes in total.
    PYTHONPATH="$ROOT" python -m src.simswap.main \
        third_party=simswap \
        evaluate=evaluate_local \
        third_party.function="metric" \
        evaluate.facenet_512.enable=false \
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
        evaluate.facenet_512.enable=false \
        evaluate.face_recognition.enable=false \
        evaluate.aws.enable=true \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60
fi
