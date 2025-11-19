#!/bin/bash

LOGS_DIR=$(cd "$(dirname "$0")" && pwd) # Resolve the absolute path of the directory
SCRIPT_NAME=$(basename "$0")

echo "Starting log organization in: $LOGS_DIR"

find "$LOGS_DIR" -maxdepth 1 -type f ! -name "$SCRIPT_NAME" | while read FILE_PATH; do
    DATE_STRING=$(stat -c %y "$FILE_PATH")
    TARGET_DIR_NAME=$(date -d "$DATE_STRING" +%a_%Y%m%d | tr '[:upper:]' '[:lower:]')
    TARGET_DIR="$LOGS_DIR/$TARGET_DIR_NAME"
    echo "  -> Moving $(basename "$FILE_PATH") to $TARGET_DIR_NAME"
    mkdir -p "$TARGET_DIR"
    mv "$FILE_PATH" "$TARGET_DIR/"
done

echo "Log organization complete."