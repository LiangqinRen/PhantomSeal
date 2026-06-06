#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))

# Experiment 1
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

# Experiment 2
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

# Experiment 3
# Expected runtime: 1 minutes per batch, 30 minutes in total.
PYTHONPATH="$ROOT" python -m src.artificialfingerprint.main \
    third_party=artificialfingerprint \
    third_party.function="forensics_robustness_metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=100

# Experiment 4
# Expected runtime: 0.5 minutes per batch, 15 minutes in total.
PYTHONPATH="$ROOT" python -m src.sepmark.main \
    third_party=sepmark \
    third_party.function="forensics_robustness_metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=100