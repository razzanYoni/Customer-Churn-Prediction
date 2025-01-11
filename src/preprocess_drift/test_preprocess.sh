#!/bin/bash

# Path to the checkpoint folder
CHECKPOINT_PATH="../../data/out/checkpoint/"

# Check if the checkpoint folder exists
if [ -d "$CHECKPOINT_PATH" ]; then
    echo "Deleting checkpoint folder: $CHECKPOINT_PATH"
    rm -rf "$CHECKPOINT_PATH"
    echo "Checkpoint folder deleted."
else
    echo "Checkpoint folder does not exist: $CHECKPOINT_PATH"
fi

# Path to the checkpoint folder
OUTPUT_PATH="../../data/out/preprocess"

# Check if the folder exists
if [ -d "$OUTPUT_PATH" ]; then
    echo "Deleting folder: $OUTPUT_PATH"
    rm -rf "$OUTPUT_PATH"
fi

# Recreate the folder
mkdir -p "$OUTPUT_PATH"
echo "Folder recreated: $OUTPUT_PATH"



python3 preprocess.py --input_path ../../data/out/drifted_data --output_path ../../data/out/preprocess --checkpoint_path ../../data/out/checkpoint --file_type parquet
