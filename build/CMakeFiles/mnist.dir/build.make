# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/build"

# Include any dependencies generated for this target.
include CMakeFiles/mnist.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mnist.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mnist.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mnist.dir/flags.make

CMakeFiles/mnist.dir/src/mnist.cpp.o: CMakeFiles/mnist.dir/flags.make
CMakeFiles/mnist.dir/src/mnist.cpp.o: /Users/mohammad/Desktop/untitled\ folder/ws2024-group-23-bean/src/mnist.cpp
CMakeFiles/mnist.dir/src/mnist.cpp.o: CMakeFiles/mnist.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mnist.dir/src/mnist.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mnist.dir/src/mnist.cpp.o -MF CMakeFiles/mnist.dir/src/mnist.cpp.o.d -o CMakeFiles/mnist.dir/src/mnist.cpp.o -c "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/src/mnist.cpp"

CMakeFiles/mnist.dir/src/mnist.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/mnist.dir/src/mnist.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/src/mnist.cpp" > CMakeFiles/mnist.dir/src/mnist.cpp.i

CMakeFiles/mnist.dir/src/mnist.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/mnist.dir/src/mnist.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/src/mnist.cpp" -o CMakeFiles/mnist.dir/src/mnist.cpp.s

CMakeFiles/mnist.dir/src/mnist_loader.cpp.o: CMakeFiles/mnist.dir/flags.make
CMakeFiles/mnist.dir/src/mnist_loader.cpp.o: /Users/mohammad/Desktop/untitled\ folder/ws2024-group-23-bean/src/mnist_loader.cpp
CMakeFiles/mnist.dir/src/mnist_loader.cpp.o: CMakeFiles/mnist.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/mnist.dir/src/mnist_loader.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mnist.dir/src/mnist_loader.cpp.o -MF CMakeFiles/mnist.dir/src/mnist_loader.cpp.o.d -o CMakeFiles/mnist.dir/src/mnist_loader.cpp.o -c "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/src/mnist_loader.cpp"

CMakeFiles/mnist.dir/src/mnist_loader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/mnist.dir/src/mnist_loader.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/src/mnist_loader.cpp" > CMakeFiles/mnist.dir/src/mnist_loader.cpp.i

CMakeFiles/mnist.dir/src/mnist_loader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/mnist.dir/src/mnist_loader.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/src/mnist_loader.cpp" -o CMakeFiles/mnist.dir/src/mnist_loader.cpp.s

CMakeFiles/mnist.dir/src/Neural_Network.cpp.o: CMakeFiles/mnist.dir/flags.make
CMakeFiles/mnist.dir/src/Neural_Network.cpp.o: /Users/mohammad/Desktop/untitled\ folder/ws2024-group-23-bean/src/Neural_Network.cpp
CMakeFiles/mnist.dir/src/Neural_Network.cpp.o: CMakeFiles/mnist.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/mnist.dir/src/Neural_Network.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mnist.dir/src/Neural_Network.cpp.o -MF CMakeFiles/mnist.dir/src/Neural_Network.cpp.o.d -o CMakeFiles/mnist.dir/src/Neural_Network.cpp.o -c "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/src/Neural_Network.cpp"

CMakeFiles/mnist.dir/src/Neural_Network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/mnist.dir/src/Neural_Network.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/src/Neural_Network.cpp" > CMakeFiles/mnist.dir/src/Neural_Network.cpp.i

CMakeFiles/mnist.dir/src/Neural_Network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/mnist.dir/src/Neural_Network.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/src/Neural_Network.cpp" -o CMakeFiles/mnist.dir/src/Neural_Network.cpp.s

# Object files for target mnist
mnist_OBJECTS = \
"CMakeFiles/mnist.dir/src/mnist.cpp.o" \
"CMakeFiles/mnist.dir/src/mnist_loader.cpp.o" \
"CMakeFiles/mnist.dir/src/Neural_Network.cpp.o"

# External object files for target mnist
mnist_EXTERNAL_OBJECTS =

mnist: CMakeFiles/mnist.dir/src/mnist.cpp.o
mnist: CMakeFiles/mnist.dir/src/mnist_loader.cpp.o
mnist: CMakeFiles/mnist.dir/src/Neural_Network.cpp.o
mnist: CMakeFiles/mnist.dir/build.make
mnist: /opt/homebrew/opt/libomp/lib/libomp.dylib
mnist: CMakeFiles/mnist.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable mnist"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mnist.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mnist.dir/build: mnist
.PHONY : CMakeFiles/mnist.dir/build

CMakeFiles/mnist.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mnist.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mnist.dir/clean

CMakeFiles/mnist.dir/depend:
	cd "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean" "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean" "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/build" "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/build" "/Users/mohammad/Desktop/untitled folder/ws2024-group-23-bean/build/CMakeFiles/mnist.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : CMakeFiles/mnist.dir/depend

