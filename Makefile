CXX = clang++
CXXFLAGS = -std=c++17 -Wall -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_86
LDFLAGS = -L/usr/local/cuda-12.9/lib64

# Include directory
INC_DIR = inc

# Binary directory
BIN_DIR = bin

# Source files
SOURCES = main.cpp $(INC_DIR)/csr_graph.cpp $(INC_DIR)/modularity_optimization.cpp $(INC_DIR)/utils.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Target executable
TARGET = $(BIN_DIR)/louvain_openmp

all: $(TARGET)

# Rule to create the binary directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Link object files to create the executable in bin/
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# Rule for main.cpp
main.o: main.cpp $(INC_DIR)/csr_graph.h $(INC_DIR)/modularity_optimization.h
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

# Rule for csr_graph.cpp
$(INC_DIR)/csr_graph.o: $(INC_DIR)/csr_graph.cpp $(INC_DIR)/csr_graph.h
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

# Rule for modularity_optimization.cpp
$(INC_DIR)/modularity_optimization.o: $(INC_DIR)/modularity_optimization.cpp $(INC_DIR)/modularity_optimization.h $(INC_DIR)/utils.h
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS) *.csr
	rm -rf $(BIN_DIR)

.PHONY: all clean
