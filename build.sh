# Step1
# mkdir build

# Step2
# cd build 
# cmake ..
# make
#!/bin/bash

#!/bin/bash

# This script compiles and links the MNIST project

BUILD_DIR="build"

# Create build directory if not exists
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Run CMake and Make
cmake ..
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful."
else
    echo "Build failed!" >&2
    exit 1
fi