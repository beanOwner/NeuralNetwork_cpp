# Handwriting Recognition (MNIST) Project

This project implements a neural network in C++ for recognizing handwritten digits from the MNIST dataset. The implementation includes three main applications:

1. `mnist`: The main neural network application for training and testing
2. `read_dataset_images`: Utility for reading and converting MNIST image data
3. `read_dataset_labels`: Utility for reading MNIST label data

## Project Structure

The project consists of the following key components:

* `build.sh`: Build script that compiles all three applications:
  - Creates a build directory
  - Runs CMake and Make
  - Compiles the project with parallel processing
  - Verifies successful build

* `read_dataset_images.sh`: Script to read MNIST image data:
  - Converts MNIST dataset images to tensor format
  - Supports reading all images or a specific image by index
  - Outputs tensor data to specified file

* `read_dataset_labels.sh`: Script to read MNIST label data:
  - Reads MNIST dataset labels
  - Converts labels to tensor format
  - Outputs label data to specified file

* `mnist.sh`: Main script for running the neural network:
  - Takes a configuration file as input
  - Validates configuration and build status
  - Executes the neural network training/testing
  - Handles error cases and provides feedback

## Source Code Structure

The project's source code is organized in the `src/` directory with the following key components:

### Core Neural Network Implementation
* `Neural_Network.hpp` and `Neural_Network.cpp`:
  - Implements a fully-connected neural network
  - Uses ADAM optimizer for training
  - Features ReLU activation and Softmax output layer
  - Implements cross-entropy loss function
  - Supports batch processing and matrix operations
  - Includes forward and backward propagation

### MNIST Data Handling
* `mnist_loader.hpp` and `mnist_loader.cpp`:
  - Handles reading and parsing MNIST dataset files
  - Converts binary MNIST data to Eigen matrices
  - Manages image and label data structures
  - Provides accessors for dataset properties

### Data Reading Utilities
* `read_dataset_images.cpp`:
  - Implements MNIST image file reading
  - Converts binary image data to tensor format
* `read_dataset_label.cpp`:
  - Implements MNIST label file reading
  - Converts binary label data to tensor format

### Configuration and Testing
* `parser.hpp`:
  - Handles configuration file parsing
  - Manages network parameters and settings
* `test.cpp`:
  - Contains testing functionality
  - Validates network performance
* `mnist.cpp`:
  - Main application entry point
  - Coordinates training and testing processes

### Core Data Structures
* `tensor.hpp`:
  - Custom tensor implementation for data handling
  - Supports multi-dimensional array operations
* `matvec.hpp`:
  - Matrix-vector multiplication implementation
  - Optimized for neural network operations

## Building and Running

1. Build the project:
```bash
./build.sh
```

2. Read dataset images:
```bash
./read_dataset_images.sh <image_dataset_input> <image_tensor_output> [image_index]
```

3. Read dataset labels:
```bash
./read_dataset_labels.sh <label_dataset_input> <label_tensor_output> <label_index>
```

4. Run the neural network:
```bash
./mnist.sh <config_file>
```

## Configuration

The neural network is configured using a configuration file that specifies:
- Dataset paths
- Network parameters
- Training/testing parameters

Example configuration files can be found in the `mnist-configs/` directory.

## Output

The neural network generates output that includes:
- Training progress
- Test results
- Accuracy metrics
- Performance statistics

## Dependencies

- C++ compiler with C++17 support
- CMake (version 3.10 or higher)
- Make
- MNIST dataset files
- Eigen library for matrix operations

## Notes

- The project uses a custom tensor implementation for data handling
- The build process is optimized for parallel compilation
- All scripts include error handling and validation
- The implementation focuses on both functionality and performance
- The neural network uses modern optimization techniques including ADAM optimizer
- Matrix operations are optimized using Eigen library

Besides the shell scripts, we also have the following directories:

* `pytorch/`:
  Reference code and playground in Python to demonstrate the program flow of a fully-connected neural network.
  The results (e.g. the development of the loss) obtained by the Python script **shall not be seen as a ground truth**
  but should provide intuition how a NN of our topology should behave, e.g. what order of magnitude for the prediction
  accuracy can be achieved.
* `expected-results/`:
  Set of reference solutions used by the CI pipeline.
  Inspect the `.gitlab-ci.yml` to see which `expected-results` file belongs to which test.
* `mnist-datasets/`:
  Binary files containing image and label data of the MNIST dataset for training/testing the neural network.
* `mnist-configs/`:
  Configuration files passed to the `mnist.sh` script that provide input arguments to steer the program flow (e.g.
  hyperparameters) of the neural network.
* `src/tensor.hpp`:
  Reference solution for the tensor assignment. We recommend you to use this implementation as central datastructure for
  image/label data, the weights and biases for your network, etc. Keep in mind that this implementation is **slow** and
  potentially needs improvements to overcome the time limits of the evaluation.
* `src/matvec.hpp`: Reference solution for a matrix-vector multiplication using the tensor class. Can be also a
  potential target for optimizations.

The file `.gitlab-ci.yml` triggers a continuous integration pipeline that clones, builds, and runs your project.
It does so using the datasets in `mnist-datasets/`. Note that **this is not the evaluation**.
We included this, so you can make sure that your code builds on our machines without having to wait for the evaluation.
The pipeline is triggered everytime you push a new commit to your repository.

**Please only trigger the pipeline when you actually want to test your code. Otherwise, we recommend adding `[skip ci]`
at the end of your commit messages.**

We suggest that you **do not modify** `.gitlab-ci.yml` unless you know what you are doing.
There should be no need to modify that file anyway.
Moving/renaming paths such as the `mnist-datasets/` directory or modifying its content might break the CI pipeline.

**If you abuse the CI resources for anything unrelated to the project we will disqualify your group.**

Obviously, you can easily revert to an earlier project state via `git revert` in case you break something by accident.

Good luck!