#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))

# Experiment 1
# Expected runtime per parameter setting: 2.5 minutes per batch, 125 minutes in total. This multirun evaluates 15 parameter settings, 1,875 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main -m \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    third_party.defense.weight.perturb=0,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000

# Experiment 2
# Expected runtime per parameter setting: 2.5 minutes per batch, 125 minutes in total. This multirun evaluates 15 parameter settings, 1,875 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main -m \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    third_party.defense.weight.identity=0,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000

# Experiment 3
# Expected runtime per parameter setting: 2.5 minutes per batch, 125 minutes in total. This multirun evaluates 15 parameter settings, 1,875 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main -m \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    third_party.defense.weight.cloak=0,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000

# Experiment 4
# Expected runtime per parameter setting: 2.5 minutes per batch, 125 minutes in total. This multirun evaluates 15 parameter settings, 1,875 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main -m \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    third_party.defense.weight.context=0,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000