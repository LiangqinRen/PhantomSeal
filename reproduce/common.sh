#!/bin/bash

set -e

declare -Ag SELECTED_EXPERIMENTS

check_experiment_args() {
    if (( $# > 1 )); then
        echo "Usage: $0 [experiment_selector]" >&2
        echo "Examples: $0 1 | $0 1,3 | $0 1,3-5" >&2
        exit 1
    fi
}

_init_experiment_lookup() {
    VALID_EXPERIMENTS=()
    declare -gA VALID_EXPERIMENT_LOOKUP=()
    for experiment in "$@"; do
        VALID_EXPERIMENTS+=("$experiment")
        VALID_EXPERIMENT_LOOKUP["$experiment"]=1
    done
}

_add_selected_experiment() {
    local experiment="$1"
    if [[ -z "${VALID_EXPERIMENT_LOOKUP[$experiment]+x}" ]]; then
        echo "Unsupported experiment '$experiment'. Available experiments: ${VALID_EXPERIMENTS[*]}" >&2
        exit 1
    fi
    SELECTED_EXPERIMENTS["$experiment"]=1
}

init_experiments() {
    local selection="${1:-}"
    shift || true
    _init_experiment_lookup "$@"

    if [[ -z "$selection" ]]; then
        for experiment in "${VALID_EXPERIMENTS[@]}"; do
            _add_selected_experiment "$experiment"
        done
        return
    fi

    local token start end experiment
    IFS=',' read -ra tokens <<< "$selection"
    for token in "${tokens[@]}"; do
        token="${token//[[:space:]]/}"
        if [[ -z "$token" ]]; then
            continue
        elif [[ "$token" =~ ^[0-9]+$ ]]; then
            _add_selected_experiment "$token"
        elif [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start="${BASH_REMATCH[1]}"
            end="${BASH_REMATCH[2]}"
            if (( start > end )); then
                echo "Invalid experiment range '$token'." >&2
                exit 1
            fi
            for (( experiment=start; experiment<=end; experiment++ )); do
                _add_selected_experiment "$experiment"
            done
        else
            echo "Invalid experiment selector '$token'. Use forms like 1, 1-3, or 1,3-5." >&2
            exit 1
        fi
    done
}

should_run_experiment() {
    local experiment="$1"
    [[ -n "${SELECTED_EXPERIMENTS[$experiment]+x}" ]]
}

announce_experiment() {
    local script_name
    script_name=$(basename "$0" .sh)
    export PHANTOMSEAL_RUN_MARKER="${script_name}_experiment${1}"
    echo "Running experiment $1"
}
