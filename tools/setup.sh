#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))

echo -e "\033[1;34mPreparing dependencies for HifiFace...\033[0m"
(
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

echo -e "\033[1;34mDownloading pre-trained models and datasets...\033[0m"
(
    gdown 1sy2ya78ASyK5-CUrmsgnJy3wvNcVsJsf -O data.tar
    tar -xf data.tar
    rm data.tar
)

echo -e "\033[1;34mDownloading pre-trained models...\033[0m"
(
    declare -A FILES=(
        # SimSwap
        ["third_party/SimSwap/checkpoints/people/latest_net_D1.pth"]="1J4hiXljlqNtVC8bVqItT2Px1JoR2hI1M"
        ["third_party/SimSwap/checkpoints/people/latest_net_D2.pth"]="1_yGkDuL0ZcOg6aqcNTZUkLWg--48IbeU"
        ["third_party/SimSwap/checkpoints/people/latest_net_G.pth"]="1VlJCRwtUP0xR5PKimJ8wDsffwzTGsMVI"
        ["third_party/SimSwap/arcface_model/arcface_checkpoint.tar"]="1601KmNPWsULuGgyKnwoFKJxHF7vPcwMO"

        # FaceShifter
        ["third_party/FaceShifter/ModelC/saved_models/G_latest.pth"]="137YW1XEkpacUD35lpl2rhWkH-Qbasukn"
        ["third_party/FaceShifter/ModelC/face_modules/model_ir_se50.pth"]="1DIXjw81VQS2lce21HOHIcHqUC8g5rrXH"

        # HifiFace
        ["third_party/HifiFace/ms1mv3_arcface_r100_fp16_backbone.pth"]="15K-wd8-DNyz3Vc_VT_G0rS0_txK6IVCY"
        ["third_party/HifiFace/model/Deep3DFaceRecon_pytorch/checkpoints/epoch_20.pth"]="1A9CrnzqiS6xt6P-7NpPPm9B6Wv3JEU0k"
        ["third_party/HifiFace/model/Deep3DFaceRecon_pytorch/BFM/01_MorphableModel.mat"]="1vEeFfFNBJuM88aAKAt4gZleI8bIJUJlM"
        ["third_party/HifiFace/checkpoints/hififace_opensouce_299999.ckpt"]="1jc8LyfRPoDI6WTnJDH0jkpl9cShc9tMh"

        # DiffFace
        ["third_party/DiffFace/checkpoints/Model.pt"]="1kYnnctk4L4Z9TbvNCvCHEtJeO0WT_UHc"
        ["third_party/DiffFace/checkpoints/GazeEstimator.pt"]="12AH0_iT23hMUBQAiRU7oq0psnK7pvAUq"
        ["third_party/DiffFace/checkpoints/FaceParser.pth"]="1HZF2Z1aK-dxQ92oEhfuRbboGBOtE2mve"
        ["third_party/DiffFace/checkpoints/Arcface_model_only.tar"]="1qiAiF8g1lKD59VWB2vA50mGE2HDkflgZ"
        ["third_party/HifiFace/model/Deep3DFaceRecon_pytorch/BFM/Exp_Pca.bin"]="1Wwjp_3ZcvwUyJZLfdlzHcmB5DPZ03roI"

        # ArtificialGANFingerprints
        ["checkpoints/artificialfingerprint/encoder.pth"]="1icSfVjqAjorOYR2CRd58A4XAs47BF-iV"
        ["checkpoints/artificialfingerprint/decoder.pth"]="1wtyb4vj-uWqO2av6bdyT0FV5m0IeL0Sz"

        # SepMark
        ["checkpoints/sepmark/EC_90.pth"]="1TYFSA4VZJEAzsCcd039VcthOHpHebmgl"

        # FaceSwap
        ["checkpoints/faceswap/faceswap.pth"]="1d__sPXDHL4q2_rRIaL6fi4l3g-Iqa3i1"
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

echo -e "\033[1;34mCreating local evaluation file...\033[0m"
(
    cp config/evaluate/evaluate.yaml config/evaluate/evaluate_local.yaml
)

echo -e "\033[1;34mUpdating DiffFace...\033[0m"
(
    cp src/diffface/patches/gaussian_diffusion.py third_party/DiffFace/models/guided_diffusion/gaussian_diffusion.py
)

echo -e "\033[1;34mUpdating SepMark...\033[0m"
(
    if [ -f third_party/SepMark/network/noise_layers/__init__.py ]; then
        : > third_party/SepMark/network/noise_layers/__init__.py
    fi
)

echo -e "\033[1;34mEnjoy! 🥳\033[0m"