#!/bin/bash

# Check if the configuration file argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_config>"
    exit 1
fi

# Configuration file path
CONFIG_FILE="$1"

# Check if configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Get the absolute path of the workspace directory
WORKSPACE_DIR=$(pwd)

# Initialize variables with empty values
TRAIN_IMAGES=""
TRAIN_LABELS=""
TEST_IMAGES=""
TEST_LABELS=""
LOG_FILE=""
EPOCHS=""
BATCH_SIZE=""
HIDDEN_SIZE=""
LEARNING_RATE=""

# Read and parse the configuration file
while IFS= read -r line || [ -n "$line" ]; do
    # Remove carriage return and % if present
    line="${line%$'\r'}"
    line="${line%\%}"
    
    # Skip empty lines and comments
    if [[ -z "$line" ]] || [[ "$line" == \#* ]]; then
        continue
    fi
    
    # Extract key and value
    if [[ "$line" =~ ^[[:space:]]*([^=]+)[[:space:]]*=[[:space:]]*(.+)[[:space:]]*$ ]]; then
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        
        # Trim whitespace from key and value
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        
        case "$key" in
            "rel_path_train_images") TRAIN_IMAGES="$value" ;;
            "rel_path_train_labels") TRAIN_LABELS="$value" ;;
            "rel_path_test_images") TEST_IMAGES="$value" ;;
            "rel_path_test_labels") TEST_LABELS="$value" ;;
            "rel_path_log_file") LOG_FILE="$(basename "$value")" ;; # Only use the filename
            "num_epochs") EPOCHS="$value" ;;
            "batch_size") BATCH_SIZE="$value" ;;
            "hidden_size") HIDDEN_SIZE="$value" ;;
            "learning_rate") LEARNING_RATE="$value" ;;
        esac
    fi
done < "$CONFIG_FILE"

# Debug output
echo "Parsed configuration:"
echo "TRAIN_IMAGES: $TRAIN_IMAGES"
echo "TRAIN_LABELS: $TRAIN_LABELS"
echo "TEST_IMAGES: $TEST_IMAGES"
echo "TEST_LABELS: $TEST_LABELS"
echo "LOG_FILE: $LOG_FILE"
echo "EPOCHS: $EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "HIDDEN_SIZE: $HIDDEN_SIZE"
echo "LEARNING_RATE: $LEARNING_RATE"

# Ensure all required parameters are set
if [ -z "$TRAIN_IMAGES" ] || [ -z "$TRAIN_LABELS" ] || [ -z "$TEST_IMAGES" ] || [ -z "$TEST_LABELS" ] || \
   [ -z "$LOG_FILE" ] || [ -z "$EPOCHS" ] || [ -z "$BATCH_SIZE" ] || [ -z "$HIDDEN_SIZE" ] || [ -z "$LEARNING_RATE" ]; then
    echo "Error: Missing parameters in configuration file!"
    exit 1
fi

# Create symbolic link for log file in build directory
cd build
ln -sf "../$LOG_FILE" "$LOG_FILE"

# Execute the C++ neural network program with relative paths
./mnist \
    "../$TRAIN_IMAGES" \
    "../$TRAIN_LABELS" \
    "../$TEST_IMAGES" \
    "../$TEST_LABELS" \
    "$LOG_FILE" \
    "$EPOCHS" \
    "$BATCH_SIZE" \
    "$HIDDEN_SIZE" \
    "$LEARNING_RATE"

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo "Training and testing completed successfully!"
    echo "Log file written to: $WORKSPACE_DIR/$LOG_FILE"
else
    echo "Error: Neural network execution failed!"
    exit 1
fi