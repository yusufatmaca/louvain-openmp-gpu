CXX = clang++
CXXFLAGS = -std=c++17 -Wall -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_86 -g
LDFLAGS = -L/usr/local/cuda-12.9/lib64

SRC_DIR = src
INC_DIR = include
OBJ_DIR = build
BIN_DIR = bin

# List source files
SRC_FILES = main.cpp $(SRC_DIR)/csr_graph.cpp $(SRC_DIR)/louvain.cpp $(SRC_DIR)/utils.cpp

# Create matching object file paths in build/
OBJ_FILES = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(SRC_FILES)))

# Final binary
TARGET = $(BIN_DIR)/louvain_openmp

all: $(TARGET)

# Link the final binary
$(TARGET): $(OBJ_FILES) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# Rule to build object files in build/ from .cpp in root or src/
$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

# Ensure bin/ and build/ directories exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)

.PHONY: all clean

