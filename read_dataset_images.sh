#!/bin/bash

# This script reads an MNIST dataset image and outputs the tensor file

if [ $# -ne 2 ] && [ $# -ne 3 ]; then
    echo "Usage: $0 <image_dataset_input> <image_tensor_output>"
    echo "Usage: $0 <image_dataset_input> <image_tensor_output> <image_index>"
    exit 1
fi

# Get input parameters
DATASET_INPUT="$1"
TENSOR_OUTPUT="$2"
IMAGE_INDEX="$3"  # Will be empty if not provided

# Check if input file exists
if [ ! -f "$DATASET_INPUT" ]; then
    echo "Error: Input dataset file '$DATASET_INPUT' not found!"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$TENSOR_OUTPUT")
if [ ! -z "$OUTPUT_DIR" ] && [ "$OUTPUT_DIR" != "." ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Change to build directory
cd build || exit 1

# Execute the image reader program
if [ $# -eq 2 ]; then
    # Read all images
    ./read_dataset_images "../$DATASET_INPUT" "../$TENSOR_OUTPUT"
else
    # Read specific image by index
    ./read_dataset_images "../$DATASET_INPUT" "../$TENSOR_OUTPUT" "$IMAGE_INDEX"
fi

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo "Successfully read images from: $DATASET_INPUT"
    echo "Output written to: $TENSOR_OUTPUT"
else
    echo "Error: Failed to read dataset images!"
    exit 1
fi