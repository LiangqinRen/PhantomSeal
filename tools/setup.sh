#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))

echo -e "\033[1;34m[1/6] Preparing dependency submodules...\033[0m"
(
    git submodule update --init --recursive
)

echo -e "\033[1;34m[2/6] Preparing HifiFace dependencies...\033[0m"
(
    cd "$ROOT/third_party/HifiFace" || exit

    if [ -d "model/Deep3DFaceRecon_pytorch" ] && [ -d "model/arcface_torch" ]; then
        echo "[INFO] HifiFace dependencies already prepared."
        exit
    fi

    git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch
    git clone https://github.com/deepinsight/insightface.git
    cp -r insightface/recognition/arcface_torch/ Deep3DFaceRecon_pytorch/models/
    cp -r insightface/recognition/arcface_torch/ ./model/
    rm -rf insightface
    cp -rf 3DMM/* Deep3DFaceRecon_pytorch
    mv Deep3DFaceRecon_pytorch model/
)

echo -e "\033[1;34m[3/6] Downloading datasets...\033[0m"
(
    gdown 1feXBOP0WGemPMpsjeNLKe1j0l1A3Cxjj -O data.tar.gz
    tar -xf data.tar.gz
    rm data.tar.gz
)

echo -e "\033[1;34m[4/6] Downloading pre-trained models...\033[0m"
(
    gdown 1ZpkwyuSipEaQj0JfxzvwcWWlyBlrkSvX -O checkpoints.tar.gz
    tar -xf checkpoints.tar.gz
    rm checkpoints.tar.gz
)

echo -e "\033[1;34m[5/6] Creating local evaluation file...\033[0m"
(
    cp config/evaluate/evaluate.yaml config/evaluate/evaluate_local.yaml
)

echo -e "\033[1;34m[6/6] Applying third-party patches...\033[0m"
(
    bash tools/apply_patches.sh
)

echo -e "\033[1;34mEnjoy! 🥳\033[0m"