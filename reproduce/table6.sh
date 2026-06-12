#!/bin/bash

set -e

ROOT=$(dirname $(dirname $(realpath "$0")))
source "$ROOT/reproduce/common.sh"
check_experiment_args "$@"

EXPERIMENTS=(1 2)
init_experiments "${1:-}" "${EXPERIMENTS[@]}"

check_table6_build_deps() {
    if ! command -v ninja >/dev/null 2>&1; then
        echo "[ERROR] Table 6 requires ninja to build E4S/GPEN CUDA extensions." >&2
        echo "[INFO] Install it with: conda install -n phantomseal -c conda-forge ninja" >&2
        exit 1
    fi

    local cxx="${CXX:-c++}"
    if ! printf "int main() { return 0; }\n" | "$cxx" -x c++ - -fsyntax-only >/dev/null 2>&1; then
        echo "[ERROR] Table 6 requires a working C++ compiler for E4S/GPEN CUDA extensions." >&2
        echo "[INFO] On Ubuntu/Debian, install it with: sudo apt-get install -y build-essential" >&2
        echo "[INFO] If using Conda without sudo, try: conda install -n phantomseal -c conda-forge gxx_linux-64" >&2
        exit 1
    fi
}

check_table6_build_deps

# Experiment 1
if should_run_experiment 1; then
    announce_experiment 1
    # Expected runtime: 9 minutes per batch, 2700 minutes in total.
    PYTHONPATH="$ROOT" python -m src.blackbox.main \
        third_party=blackbox \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=10 \
        evaluate.effectiveness.perturb=false \
        protection=phantomseal
fi

# Experiment 2
if should_run_experiment 2; then
    announce_experiment 2
    # Expected runtime: 8 minutes per batch, 2400 minutes in total.
    PYTHONPATH="$ROOT" python -m src.blackbox.main \
        third_party=blackbox \
        evaluate=evaluate_local \
        third_party.function="metric" \
        third_party.dataset.metric_pairs=3000 \
        third_party.defense.batch_size=10 \
        evaluate.effectiveness.perturb=false \
        protection=nullswap
fi
