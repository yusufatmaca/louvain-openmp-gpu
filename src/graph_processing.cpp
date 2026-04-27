#include "csr_graph.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>

// Function to read a graph from a MTX file
Graph constructGraph(const std::string& filename) {
    Graph graph;
    std::ifstream file(filename);
    std::string line;
    
    auto start_time = std::chrono::high_resolution_clock::now();

    // Read the file header
    while (std::getline(file, line)) {
        if (line[0] != '%') break; // Skip comments
    }

    // Read dimensions and number of edges
    std::istringstream iss(line);
    int num_vertices, num_edges;
    iss >> num_vertices >> num_vertices >> num_edges; // Assuming square matrix

    graph.num_vertices = num_vertices;
    graph.num_edges = num_edges * 2; // For undirected graph, double the edges

    // Allocate memory for raw pointers
    graph.row_ptr = new int[num_vertices + 1](); // Initialize to 0
    graph.col_idx = new int[graph.num_edges]();
    graph.values = new double[graph.num_edges]();

    // Temporary counter for row_ptr
    std::vector<int> counter(num_vertices + 1, 0);

    // Read edges and count degrees
    while (std::getline(file, line)) {
        std::istringstream edge(line);
        int src, dst;
        double weight = 1.0; // Default weight
        edge >> src >> dst >> weight;

        // Adjust for 0-based indexing (if file is 1-based)
        src--;
        dst--;

        // Increment degree for src and dst (undirected graph)
        counter[src]++;
        counter[dst]++;
    }

    // Build row_ptr (cumulative sum of degrees)
    graph.row_ptr[0] = 0;
    for (int i = 1; i <= num_vertices; i++) {
        graph.row_ptr[i] = graph.row_ptr[i - 1] + counter[i - 1];
    }

    // Reset counter for filling col_idx and values
    std::fill(counter.begin(), counter.end(), 0);

    // Rewind file to read edges again
    file.clear();
    file.seekg(0);
    while (std::getline(file, line)) {
        if (line[0] == '%') continue;
        std::istringstream header(line);
        int temp;
        if (header >> temp) break; // Skip the header line
    }

    // Fill col_idx and values
    std::vector<int> offsets = std::vector<int>(graph.row_ptr, graph.row_ptr + num_vertices + 1);
    while (std::getline(file, line)) {
    std::istringstream edge(line);
    int src, dst;
    double weight = 1.0; // Default to 1.0 for unweighted graphs
    if (edge >> src >> dst) {
        weight = 1.0; // Explicitly set weight to 1.0
        // std::cout << "Edge (" << src << ", " << dst << ") weight: " << weight << std::endl;
    } else {
        std::cout << "Failed to parse edge: " << line << std::endl;
        continue;
    }

    src--;
    dst--;

    int idx = graph.row_ptr[src] + counter[src];
    graph.col_idx[idx] = dst;
    graph.values[idx] = weight;
    counter[src]++;

    idx = graph.row_ptr[dst] + counter[dst];
    graph.col_idx[idx] = src;
    graph.values[idx] = weight;
    counter[dst]++;
}

    file.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    printExecutionTime(start_time, end_time, "Graph construction");
    std::cout << "---------------------------------------------------" << std::endl;

    printGraphStats(graph);
    std::cout << "---------------------------------------------------" << std::endl;
    return graph;
}

// Function to print the graph (for debugging)
void printGraphStats(const Graph& graph) {
    std::cout << "Number of vertices: " << graph.num_vertices << "\n";
    std::cout << "Number of edges: " << graph.num_edges/2 << "\n";
}