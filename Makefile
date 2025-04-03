# Compilers
NVCC = nvcc
CXX = g++

# OpenCV flags
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# Compiler flags
NVCCFLAGS = -Iinclude -Xcompiler -Wall -m64 $(OPENCV_CFLAGS)
CXXFLAGS = -Iinclude -Wall -m64 $(OPENCV_CFLAGS)

# Target executable
TARGET = Test

# Directories
SRC_DIR = src/morphology
TESTS_DIR = tests_cuda
OBJ_DIR = obj

# Source files
# CPP_SRCS = $(shell find $(SRC_DIR) $(TESTS_DIR) -type f -name '*.cpp')
# CU_SRCS = $(shell find $(SRC_DIR) $(TESTS_DIR) -type f -name '*.cu')
# ignore snakes folder -> it was causing compilation errors
CPP_SRCS = $(shell find $(SRC_DIR) $(TESTS_DIR) -type f -name '*.cpp' ! -path "$(SRC_DIR)/snakes/*")
CU_SRCS = $(shell find $(SRC_DIR) $(TESTS_DIR) -type f -name '*.cu' ! -path "$(SRC_DIR)/snakes/*")
SRCS = $(CPP_SRCS) $(CU_SRCS)

# Object files
CPP_OBJS = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(subst $(SRC_DIR)/,,$(subst $(TESTS_DIR)/,,$(CPP_SRCS))))
CU_OBJS = $(patsubst %.cu, $(OBJ_DIR)/%.o, $(subst $(SRC_DIR)/,,$(subst $(TESTS_DIR)/,,$(CU_SRCS))))
OBJS = $(CPP_OBJS) $(CU_OBJS)

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(CU_OBJS) $(CPP_OBJS) -o $(TARGET) $(OPENCV_LIBS)

# Compile C++ source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(TESTS_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(TESTS_DIR)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean