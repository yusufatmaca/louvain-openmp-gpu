#ifndef CSR_GRAPH_H
#define CSR_GRAPH_H

#include <vector>
#include <string>

// Structure to represent an edge in the graph
struct Edge {
    int src;    // Source vertex (1-indexed in MTX)
    int dst;    // Destination vertex (1-indexed in MTX)
    int weight; // Edge weight
};

// Structure to represent a graph in CSR format
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

#endif