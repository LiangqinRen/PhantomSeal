#!/bin/bash

set -e

TARGET_DIR="${1:-.}"

echo "Removing .DS_Store files under: $TARGET_DIR"

find "$TARGET_DIR" -type f -name ".DS_Store" -print -delete

echo "Done."