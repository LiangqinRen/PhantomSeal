#!/bin/bash

LOGS_DIRS=(
  logs/run
  logs/multirun
)

for LOGS_DIR in "${LOGS_DIRS[@]}"; do
  [ -d "$LOGS_DIR" ] || continue

  echo "Processing log dir: $LOGS_DIR"

  for date_dir in "$LOGS_DIR"/*; do
    [ -d "$date_dir" ] || continue

    for time_dir in "$date_dir"/*; do
      if [[ -d "$time_dir" && "$time_dir" == *-debug ]]; then
        echo "Deleting debug folder: $time_dir"
        rm -rf "$time_dir"
      fi
    done

    if [ -z "$(ls -A "$date_dir")" ]; then
      echo "Deleting empty date folder: $date_dir"
      rmdir "$date_dir"
    fi
  done
done
