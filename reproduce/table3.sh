#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))

# Experiment 1
# Expected runtime: 2.5 minutes per batch, 125 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60

# Experiment 2
# Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
PYTHONPATH="$ROOT" python -m src.diffface.main \
    third_party=diffface \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=30