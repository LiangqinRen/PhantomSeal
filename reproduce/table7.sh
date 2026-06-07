#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))
source "$ROOT/reproduce/common.sh"
check_experiment_args "$@"

EXPERIMENTS=(1 2 3 4 5 6 7 8)
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
        third_party.defense.batch_size=60 \
        evaluate.effectiveness.perturb=false \
        third_party.defense.weight.context=0
fi

# Experiment 2
if should_run_experiment 2; then
    announce_experiment 2
    # Expected runtime: 1.5 minutes per batch, 75 minutes in total.
    PYTHONPATH="$ROOT" python -m src.faceshifter.main \
        third_party=faceshifter \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        evaluate.effectiveness.perturb=false
fi

# Experiment 3
if should_run_experiment 3; then
    announce_experiment 3
    # Expected runtime: 3 minutes per batch, 150 minutes in total.
    PYTHONPATH="$ROOT" python -m src.hififace.main \
        third_party=hififace \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=60 \
        evaluate.effectiveness.perturb=false
fi

# Experiment 4
if should_run_experiment 4; then
    announce_experiment 4
    # Expected runtime: 12.5 minutes per batch, 1250 minutes in total.
    PYTHONPATH="$ROOT" python -m src.diffface.main \
        third_party=diffface \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=30 \
        evaluate.effectiveness.perturb=false \
        third_party.defense.weight.context=0
fi

# Experiment 5
if should_run_experiment 5; then
    announce_experiment 5
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
fi

# Experiment 6
if should_run_experiment 6; then
    announce_experiment 6
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
fi

# Experiment 7
if should_run_experiment 7; then
    announce_experiment 7
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
fi

# Experiment 8
if should_run_experiment 8; then
    announce_experiment 8
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
fi
