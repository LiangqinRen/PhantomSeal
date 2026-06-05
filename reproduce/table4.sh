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
# Expected runtime: 2.5 minutes per batch, 125 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    third_party.dataset.cloak_dir=data/stylegan3_cloak

# Experiment 3
# Expected runtime: 2.5 minutes per batch, 125 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    third_party.dataset.cloak_mix=false

# Experiment 4
# Expected runtime: 2.5 minutes per batch, 125 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    third_party.dataset.cloak_distance=1.35

# Experiment 5
# Expected runtime: 2.5 minutes per batch, 125 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main -m \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    third_party.dataset.cloak_distance=1.04,1.25,1.50,1.75,2.00

# Experiment 6
# Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
PYTHONPATH="$ROOT" python -m src.diffface.main \
    third_party=diffface \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=30

# Experiment 7
# Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
PYTHONPATH="$ROOT" python -m src.diffface.main \
    third_party=diffface \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=30 \
    third_party.dataset.cloak_dir=data/stylegan3_cloak

# Experiment 8
# Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
PYTHONPATH="$ROOT" python -m src.diffface.main \
    third_party=diffface \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=30 \
    third_party.dataset.cloak_mix=false

# Experiment 9
# Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
PYTHONPATH="$ROOT" python -m src.diffface.main \
    third_party=diffface \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=30 \
    third_party.dataset.cloak_distance=1.55

# Experiment 10
# Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
PYTHONPATH="$ROOT" python -m src.diffface.main -m \
    third_party=diffface \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=30 \
    third_party.dataset.cloak_distance=1.04,1.25,1.50,1.75,2.00