#!/bin/bash

# Print usage information
print_usage() {
    echo "Usage: $0 <config_file>"
    echo "Example: $0 mnist-configs/input-ci.config"
    echo ""
    echo "The configuration file should contain the MNIST dataset paths and parameters"
}

# Check if the configuration file argument is provided
if [ "$#" -ne 1 ]; then
    print_usage
    exit 1
fi

# Configuration file path
CONFIG_FILE="$1"

# Check if configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    print_usage
    exit 1
fi

# Get the absolute path of the workspace directory
WORKSPACE_DIR=$(pwd)

# Define build directory
BUILD_DIR="build"

# Check if mnist executable exists
if [ ! -f "$BUILD_DIR/mnist" ]; then
    echo "Error: mnist executable not found!"
    echo "Please run ./build.sh first to build the project"
    exit 1
fi

echo "Starting neural network with config file: $CONFIG_FILE"

# Execute the C++ neural network program with the config file
$BUILD_DIR/mnist "$CONFIG_FILE"

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo "Neural network execution completed successfully!"
else
    echo "Error: Neural network execution failed!"
    echo "Please check the error messages above"
    exit 1
fi