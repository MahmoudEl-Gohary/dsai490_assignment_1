#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP_FILE="$SCRIPT_DIR/medical-mnist.zip"

curl -L -o "$ZIP_FILE" \
    "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/medical-mnist"

unzip -o "$ZIP_FILE" -d "$SCRIPT_DIR/Medical-MNIST"
rm -f "$ZIP_FILE"