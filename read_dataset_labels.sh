#!/bin/bash

# This script reads an MNIST dataset label and outputs the tensor file

if [ $# -ne 3 ]; then
    echo "Usage: $0 <label_dataset_input> <label_tensor_output> <label_index>"
    exit 1
fi

DATASET_INPUT="$1"
TENSOR_OUTPUT="$2"
LABEL_INDEX="$3"
BUILD_DIR="build"

# Ensure the executable exists
if [ ! -f "$BUILD_DIR/read_dataset_labels" ]; then
    echo "Error: read_dataset_labels executable not found. Run build.sh first." >&2
    exit 1
fi

# Run the label reader
$BUILD_DIR/read_dataset_labels "$DATASET_INPUT" "$TENSOR_OUTPUT" "$LABEL_INDEX"