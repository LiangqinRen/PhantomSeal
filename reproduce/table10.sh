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
    evaluate.effectiveness.ASRo=false

# Experiment 2
# Expected runtime: 2.5 minutes per batch, 125 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="ai_beauty" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    third_party.robustness.ai_beauty=true \
    third_party.robustness.ai_beauty_tool=ai_lab_tools \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false

# Experiment 3
# Expected runtime: 2.5 minutes per batch, 125 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="ai_beauty" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    third_party.robustness.ai_beauty=true \
    third_party.robustness.ai_beauty_tool=tencent_cloud \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false

# Experiment 4
# Expected runtime: 6 minutes per batch, 300 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="image_robustness_metric" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false

# Experiment 5
# Expected runtime: 4 minutes per batch, 200 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="adaptive_attack_with_self_image" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false

# Experiment 6
# Expected runtime: 4 minutes per batch, 200 minutes in total.
PYTHONPATH="$ROOT" python -m src.simswap.main \
    third_party=simswap \
    evaluate=evaluate_local \
    third_party.function="adaptive_attack_with_other_image" \
    third_party.dataset.metric_pairs=3000 \
    third_party.defense.batch_size=60 \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false

# Experiment 7
# Expected runtime per checkpoint: 1 minutes per batch, 60 minutes in total. This multirun evaluates 4 checkpoints, 240 minutes in total.
PYTHONPATH="$ROOT" python -m src.denoiser.main -m \
    third_party=denoiser \
    evaluate=evaluate_local \
    third_party.function="test" \
    evaluate.effectiveness.perturb=false \
    evaluate.effectiveness.ASRo=false \
    third_party.test.checkpoint_path=checkpoints/denoiser/10.pth,checkpoints/denoiser/50.pth,checkpoints/denoiser/100.pth,checkpoints/denoiser/500.pth \
    third_party.test.max_images=2000 \
    third_party.test.batch_size=32