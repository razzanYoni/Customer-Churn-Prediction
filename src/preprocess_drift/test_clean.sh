#!/bin/bash

# Path to the checkpoint folder
OUTPUT_PATH="../../data/out/cleaned"

# Check if the folder exists
if [ -d "$OUTPUT_PATH" ]; then
    echo "Deleting folder: $OUTPUT_PATH"
    rm -rf "$OUTPUT_PATH"
fi

# Recreate the folder
mkdir -p "$OUTPUT_PATH"
echo "Folder recreated: $OUTPUT_PATH"



python3 cleaning.py --input_path ../../data/out/drifted_data --output_path ../../data/out/cleaned --file_type parquet
