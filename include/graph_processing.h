#ifndef CSR_GRAPH_H
#define CSR_GRAPH_H

#include <iostream>
#include <chrono>

// Structure to represent a graph in Compressed Sparse Row (CSR) format
struct Graph {
    int* row_ptr;           // Row pointers (offsets)
    int* col_idx;           // Column indices
    double* values;         // Edge weights/values
    
    int num_vertices;       // Number of vertices
    int num_edges;          // Number of edges
    int total_edge_weight;  // Total edge weight (sum of all weights)

    int max_degree;         // Maximum degree of the graph
    int min_degree;         // Minimum degree of the graph
};

Graph constructGraph(const std::string& filename);
void printGraphStats(const Graph& graph);
void printExecutionTime(
    std::chrono::high_resolution_clock::time_point start_time, 
    std::chrono::high_resolution_clock::time_point end_time,
    char* process_name);

#endif