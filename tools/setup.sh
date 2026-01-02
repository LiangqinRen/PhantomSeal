#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))

clone_and_checkout () {
    local name=$1
    local url=$2
    local commit=$3

    local dir="$ROOT/third_party/$name"

    if [ -d "$dir" ]; then
        echo "[INFO] $name already exists, skipping clone."
        return
    fi

    echo "[INFO] Cloning $name ..."
    git clone "$url" "$dir"
    cd "$dir"

    echo "[INFO] Checking out $commit ..."
    git checkout "$commit"

    cd "$ROOT"
}

echo -e "\033[1;34m[1/6] Preparing dependency submodules...\033[0m"
(
    clone_and_checkout \
    "SimSwap" \
    "https://github.com/neuralchen/SimSwap.git" \
    "bd7b7686a17f41dd11cfcd5d82f7e4c5eb94b780"

    clone_and_checkout \
    "FaceShifter" \
    "https://github.com/richarduuz/Research_Project.git" \
    "d970296cf9d9fe059d729444c757334c9e13a798"

    clone_and_checkout \
    "HifiFace" \
    "https://github.com/maum-ai/hififace.git" \
    "c92db89c7bd0e4d1518f872827fdb69a98f16060"

    clone_and_checkout \
    "DiffFace" \
    "https://github.com/hxngiee/DiffFace.git" \
    "b7bef83719d97a19ead41e3d69cd2e3ce00b7ea3"

    clone_and_checkout \
    "ArtificialGANFingerprints" \
    "https://github.com/ningyu1991/ArtificialGANFingerprints.git" \
    "f4a4bcc9a5a268487786c6867fbdba043f749bdb"

    clone_and_checkout \
    "SepMark" \
    "https://github.com/sh1newu/SepMark.git" \
    "4bf060da5b31abc2d623b6edf8aafcdd2d2f8db1"

    cd $ROOT/third_party/HifiFace || exit
    git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch && git clone https://github.com/NVlabs/nvdiffrast && git clone https://github.com/deepinsight/insightface.git
    cp -r insightface/recognition/arcface_torch/ Deep3DFaceRecon_pytorch/models/
    cp -r insightface/recognition/arcface_torch/ ./model/
    rm -rf insightface
    cp -rf 3DMM/* Deep3DFaceRecon_pytorch
    mv Deep3DFaceRecon_pytorch model/
    rm -rf 3DMM
    rm -rf nvdiffrast
)

echo -e "\033[1;34m[2/6] Downloading datasets...\033[0m"
(
    gdown 1iZ1EOWVXZvGoQ7BursH8MqvxXfpkQwK8 -O data.tar
    tar -xf data.tar
    rm data.tar
)

echo -e "\033[1;34m[3/6] Downloading pre-trained models...\033[0m"
(
    declare -A FILES=(
        # SimSwap
        ["third_party/SimSwap/checkpoints/people/latest_net_D1.pth"]="13aFY8yYtpNnz2Qh7GLSeIpBHQTcVMB3i"
        ["third_party/SimSwap/checkpoints/people/latest_net_D2.pth"]="1nGqLigx4Lp6wKM4hfdwb-w_NlaQG8L9q"
        ["third_party/SimSwap/checkpoints/people/latest_net_G.pth"]="15_kHmSi1phbPS7ZLzOYeam-WPsJD7o-U"
        ["third_party/SimSwap/arcface_model/arcface_checkpoint.tar"]="1T1dYHvrDf65FusJzktZU2kMhNasspdtF"

        # FaceShifter
        ["third_party/FaceShifter/ModelC/saved_models/G_latest.pth"]="1SPMR9zq6OqHDyuOelNS900SDpWEmTaq8"
        ["third_party/FaceShifter/ModelC/face_modules/model_ir_se50.pth"]="1YX-4-ey9w4Xy7fAWMiMtgLq-RUbXUMjI"

        # HifiFace
        ["third_party/HifiFace/ms1mv3_arcface_r100_fp16_backbone.pth"]="1YICKrKUt1TvhTARFBJ3p2FWMfgkQU8SB"
        ["third_party/HifiFace/model/Deep3DFaceRecon_pytorch/checkpoints/epoch_20.pth"]="1mpPqwGnsqKaqbP6_V1xfZkQ2qgAT46Jw"
        ["third_party/HifiFace/model/Deep3DFaceRecon_pytorch/BFM/01_MorphableModel.mat"]="13ty-Cmgfl3AzQZD1ygMTJAlO8KpgxUP0"
        ["third_party/HifiFace/checkpoints/hififace_opensouce_299999.ckpt"]="1ETfgexjr3UafuF2ukbpbzMCkMhuZ0uwi"
        ["third_party/HifiFace/model/Deep3DFaceRecon_pytorch/BFM/Exp_Pca.bin"]="1DtMdU17HpO5-0PCuywVWBQCHh6K7Ijca"

        # DiffFace
        ["third_party/DiffFace/checkpoints/Model.pt"]="1P4ZgwOs29cJ0fFNrwaVxYH2PIWHrIZe-"
        ["third_party/DiffFace/checkpoints/GazeEstimator.pt"]="1LAqMIe1jpct0CAw-aWJCZPEmbXTXwjr5"
        ["third_party/DiffFace/checkpoints/FaceParser.pth"]="1D4oakP-rhjlsmHZ4U_-SGgPZjYVAtD1a"
        ["third_party/DiffFace/checkpoints/Arcface_model_only.tar"]="1OQNpEiF6aBZPfeJaoqsLLv8UZDh7bVpT"


        # ArtificialGANFingerprints
        ["checkpoints/artificialfingerprint/encoder.pth"]="1R-eIlCEZEyMsrcjukSQOAwfY-LE-Q7l7"
        ["checkpoints/artificialfingerprint/decoder.pth"]="14ISuaRWd3r4uD8w4j6Px8sxQgBN8QMTo"

        # SepMark
        ["checkpoints/sepmark/EC_90.pth"]="1rIXqbe4BdXBgbFFylpsVrVhmsE5Pxt2K"

        # FaceSwap
        ["checkpoints/faceswap/faceswap.pth"]="1x1QLqmTY7wMXhrBQgYANQAXw9mvNct6v"
    )

    for rel in "${!FILES[@]}"; do
        id="${FILES[$rel]}"
        dest="$ROOT/$rel"

        mkdir -p "$(dirname "$dest")"

        if [[ -f "$dest" ]]; then
            echo "[Skip] $rel exists"
            continue
        fi

        echo "[Download] $rel"
        gdown "https://drive.google.com/uc?id=$id" -O "$dest"
    done
)

echo -e "\033[1;34m[4/6] Creating local evaluation file...\033[0m"
(
    cp config/evaluate/evaluate.yaml config/evaluate/evaluate_local.yaml
)

echo -e "\033[1;34m[5/6] Updating DiffFace...\033[0m"
(
    cp src/diffface/patches/gaussian_diffusion.py third_party/DiffFace/models/guided_diffusion/gaussian_diffusion.py
)

echo -e "\033[1;34m[6/6] Updating SepMark...\033[0m"
(
    if [ -f third_party/SepMark/network/noise_layers/__init__.py ]; then
        : > third_party/SepMark/network/noise_layers/__init__.py
    fi
)

echo -e "\033[1;34mEnjoy! 🥳\033[0m"