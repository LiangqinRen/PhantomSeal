#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))
FAILED_STEPS=()
WARN_COLOR="\033[1;33m"
RESET_COLOR="\033[0m"
WARN_LINE="======================================================================"

download_and_extract() {
    local label=$1
    local file_id=$2
    local archive=$3

    if [ -f "$archive" ]; then
        echo "[INFO] Found existing ${archive}; validating before reuse..."
        if ! tar -tzf "$archive" >/dev/null; then
            echo "[WARN] Existing ${archive} is incomplete or corrupted; removing it."
            rm -f "$archive"
        fi
    fi

    if [ ! -f "$archive" ]; then
        echo "[INFO] Downloading ${label}..."
        if ! gdown --continue "$file_id" -O "$archive"; then
            echo "[ERROR] Failed to download ${label}. This is often caused by Google Drive quota limits. Continuing with the next setup step."
            return 1
        fi
    fi

    echo "[INFO] Extracting ${archive}..."
    if ! tar -xzf "$archive"; then
        echo "[ERROR] Failed to extract ${archive}. Continuing with the next setup step."
        return 1
    fi

    rm -f "$archive"
    return 0
}

record_failure() {
    FAILED_STEPS+=("$1")
}

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

REQUIRED_BFM_FILES=(
    "01_MorphableModel.mat"
    "Exp_Pca.bin"
    "BFM_front_idx.mat"
    "BFM_exp_idx.mat"
    "facemodel_info.mat"
    "similarity_Lm3D_all.mat"
    "std_exp.txt"
)
OPTIONAL_BFM_FILES=(
    "BFM_model_front.mat"
    "select_vertex_id.mat"
)
BFM_DIR="$ROOT/third_party/HifiFace/model/Deep3DFaceRecon_pytorch/BFM"
BFM_SOURCE_DIR="$ROOT/checkpoints/hififace"

bfm_missing_files() {
    local dir="$1"
    local missing=()
    local file
    for file in "${REQUIRED_BFM_FILES[@]}"; do
        if [ ! -f "$dir/$file" ]; then
            missing+=("$file")
        fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        printf "%s\n" "${missing[@]}"
    fi
}

restore_bfm_files() {
    local source_dir="$1"
    local file
    if [ ! -d "$source_dir" ]; then
        return
    fi

    mkdir -p "$BFM_DIR"
    for file in "${REQUIRED_BFM_FILES[@]}" "${OPTIONAL_BFM_FILES[@]}"; do
        if [ -f "$source_dir/$file" ]; then
            cp "$source_dir/$file" "$BFM_DIR/$file"
        fi
    done
}

mapfile -t MISSING_BFM_FILES < <(bfm_missing_files "$BFM_DIR")
if [ "${#MISSING_BFM_FILES[@]}" -gt 0 ]; then
    echo "[INFO] Restoring HifiFace BFM data from $BFM_SOURCE_DIR"
    restore_bfm_files "$BFM_SOURCE_DIR"
    mapfile -t MISSING_BFM_FILES < <(bfm_missing_files "$BFM_DIR")
fi

if [ "${#MISSING_BFM_FILES[@]}" -gt 0 ]; then
    echo "[ERROR] Missing HifiFace BFM data files in $BFM_DIR:" >&2
    for file in "${MISSING_BFM_FILES[@]}"; do
        echo "  - $file" >&2
    done
    echo "[ERROR] Put these files in $BFM_SOURCE_DIR, then rerun tools/setup.sh." >&2
    exit 1
fi

echo -e "\033[1;34m[3/6] Downloading datasets...\033[0m"
(
    cd "$ROOT" || exit
    download_and_extract "datasets" "1feXBOP0WGemPMpsjeNLKe1j0l1A3Cxjj" "data.tar.gz"
) || record_failure "datasets"

echo -e "\033[1;34m[4/6] Downloading pre-trained models...\033[0m"
(
    cd "$ROOT" || exit
    download_and_extract "pre-trained models" "1ykvmB4BIPM0Uix-aI6qxjbL056LKr4_d" "checkpoints.tar.gz"
) || record_failure "pre-trained models"

echo -e "\033[1;34m[5/6] Creating local evaluation file...\033[0m"
(
    cp config/evaluate/evaluate.yaml config/evaluate/evaluate_local.yaml
)

echo -e "\033[1;34m[6/6] Applying third-party patches...\033[0m"
(
    bash tools/apply_patches.sh
)

echo -e "\033[1;34mEnjoy! 🥳\033[0m"

if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
    echo -e "${WARN_COLOR}${WARN_LINE}${RESET_COLOR}"
    echo -e "${WARN_COLOR}[WARN] Manual download is required for the failed setup step(s).${RESET_COLOR}"
    echo -e "${WARN_COLOR}${WARN_LINE}${RESET_COLOR}"
    echo -e "${WARN_COLOR}[WARN] Failed non-blocking steps:${RESET_COLOR}"
    for step in "${FAILED_STEPS[@]}"; do
        echo "  - ${step}"
    done
    echo
    echo -e "${WARN_COLOR}[WARN] gdown failures are commonly caused by Google Drive quota limits.${RESET_COLOR}"
    echo -e "${WARN_COLOR}[WARN] Please download the failed archives manually from:${RESET_COLOR}"
    echo "  https://drive.google.com/drive/folders/1caHioBnA1478FR15W3JzNuHu36zxopJv?usp=sharing"
    echo
    echo -e "${WARN_COLOR}[WARN] Put the downloaded archive(s) in the repository root:${RESET_COLOR}"
    echo "  ${ROOT}"
    echo
    echo -e "${WARN_COLOR}[WARN] Then run the needed extraction command(s):${RESET_COLOR}"
    for step in "${FAILED_STEPS[@]}"; do
        case "$step" in
            datasets)
                echo "  tar -xzf data.tar.gz && rm -f data.tar.gz"
                ;;
            "pre-trained models")
                echo "  tar -xzf checkpoints.tar.gz && rm -f checkpoints.tar.gz"
                ;;
        esac
    done
    echo -e "${WARN_COLOR}${WARN_LINE}${RESET_COLOR}"
    exit 1
fi
