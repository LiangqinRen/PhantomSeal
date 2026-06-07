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
    third_party.defense.batch_size=60 \
    evaluate.effectiveness.perturb=false \
    third_party.defense.weight.context=0

# Experiment 2
# Expected runtime: 1.5 minutes per batch, 75 minutes in total.
PYTHONPATH="$ROOT" python -m src.faceshifter.main \
    third_party=faceshifter \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    evaluate.effectiveness.perturb=false

# Experiment 3
# Expected runtime: 3 minutes per batch, 150 minutes in total.
PYTHONPATH="$ROOT" python -m src.hififace.main \
    third_party=hififace \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    evaluate.effectiveness.perturb=false

# Experiment 4
# Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
PYTHONPATH="$ROOT" python -m src.diffface.main \
    third_party=diffface \
    evaluate=evaluate_local \
    third_party.function="metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=30 \
    evaluate.effectiveness.perturb=false \
    third_party.defense.weight.context=0

# Experiment 5
# Expected runtime: 5 minutes per batch, 250 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="lowkey" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    evaluate.effectiveness.TSR=false

# Experiment 6
# Expected runtime: 5 minutes per batch, 250 minutes in total.
PYTHONPATH="$ROOT" python -m src.faceshifter.main \
    third_party=faceshifter \
    evaluate=evaluate_local \
    third_party.function="lowkey" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    evaluate.effectiveness.TSR=false

# Experiment 7
# Expected runtime: 5 minutes per batch, 250 minutes in total.
PYTHONPATH="$ROOT" python -m src.hififace.main \
    third_party=hififace \
    evaluate=evaluate_local \
    third_party.function="lowkey" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    evaluate.effectiveness.TSR=false

# Experiment 8
# Expected runtime: 15 minutes per batch, 1500 minutes in total.
PYTHONPATH="$ROOT" python -m src.diffface.main \
    third_party=diffface \
    evaluate=evaluate_local \
    third_party.function="lowkey" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=30 \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    evaluate.effectiveness.TSR=false \
    third_party.defense.weight.context=0
