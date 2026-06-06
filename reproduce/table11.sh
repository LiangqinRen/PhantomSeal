#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))

# Experiment 1
# Expected runtime: 3.5 minutes per batch, 175 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="protection_robustness_metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    evaluate.effectiveness.TSR=false \
    third_party.defense.epochs=335

# Experiment 2
# Expected runtime: 3.5 minutes per batch, 175 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="protection_robustness_metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    evaluate.effectiveness.TSR=false \
    third_party.defense.epochs=335 \
    third_party.dataset.cloak_distance=1.35