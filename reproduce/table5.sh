#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))

# Experiment 1
# Expected runtime: 3 minutes per batch, 450 minutes in total.
PYTHONPATH="$ROOT" python -m src.unify.main \
    third_party=unify \
    evaluate=evaluate_local \
    third_party.function="metric" \
    evaluate.effectiveness.perturb=false \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=20

# Experiment 2
# Expected runtime: 10 minutes per batch, 1,500 minutes in total.
PYTHONPATH="$ROOT" python -m src.unify.main \
    third_party=unify \
    evaluate=evaluate_local \
    third_party.function="extend" \
    evaluate.effectiveness.perturb=false \
    third_party.extend.metric_pairs=3000 \
    third_party.defense.batch_size=20