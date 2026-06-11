#!/bin/bash

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

apply_patches() {
  local submodule_path="$1"
  local patch_dir="$2"

  local tp="$ROOT/$submodule_path"
  local pd="$ROOT/$patch_dir"

  echo "[submodule] $submodule_path"
  git -C "$ROOT" submodule update --init --recursive "$submodule_path"

  shopt -s nullglob
  local patches=("$pd"/*.patch)
  shopt -u nullglob

  if [ ${#patches[@]} -eq 0 ]; then
    echo "[patch] no patches found in $pd"
    return 0
  fi

  IFS=$'\n' patches=($(printf '%s\n' "${patches[@]}" | sort))
  unset IFS

  for p in "${patches[@]}"; do
    echo "[patch] $(basename "$p")"
    if git -C "$tp" apply --reverse --check "$p" >/dev/null 2>&1; then
      echo "  already applied"
    else
      git -C "$tp" apply --whitespace=nowarn "$p"
    fi
  done
}

apply_patches "third_party/DiffFace" "patches/diffface"
apply_patches "third_party/DiffSwap" "patches/diffswap"
apply_patches "third_party/HifiFace" "patches/hififace"
apply_patches "third_party/SepMark" "patches/sepmark"
apply_patches "third_party/E4S" "patches/e4s"
apply_patches "third_party/InfoSwap" "patches/infoswap"
apply_patches "third_party/UniFace" "patches/uniface"