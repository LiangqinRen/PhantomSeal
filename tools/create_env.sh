#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-phantomseal}"

cd "$ROOT"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[ERROR] Conda environment '$ENV_NAME' already exists."
    echo "[INFO] Remove it first with: conda env remove -n $ENV_NAME"
    exit 1
fi

PYTHONNOUSERSITE=1 conda env create -n "$ENV_NAME" -f environment.yml

conda run -n "$ENV_NAME" python -m pip install \
    --no-deps \
    face-recognition==1.3.0 \
    facenet-pytorch==2.6.0

conda run -n "$ENV_NAME" python - <<'PY_CHECK'
import dlib
import facenet_pytorch
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import mtcnn

print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('dlib', dlib.__version__)
print('facenet_pytorch', getattr(facenet_pytorch, '__version__', 'unknown'))
print('mtcnn_has_detect_faces', hasattr(mtcnn, 'detect_faces'))
print('facenet_classes', MTCNN.__name__, InceptionResnetV1.__name__)
PY_CHECK
