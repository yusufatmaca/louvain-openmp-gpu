#include "inc/csr_graph.h"
#include "inc/modularity_optimization.h"
#include "inc/utils.h"
#include <iostream>
#include <chrono>

/**
 * Implements the Louvain algorithm for community detection
 * 
 * @param graph         Pointer to the Graph structure
 * @return              Modularity value achieved

double louvain(Graph* graph) {
    int* communities = new int[graph->num_vertices];
    double threshold = 1e-6;
    
    for(;;) {
        modularity_optimization(*graph, communities, threshold);
    }
    
}
*/
int main(int argc, char* argv[]) {
    
    if (checkArgs(argc, argv) != 1 && checkDevice() != 1) {
        return -1;
    }

    const char* filename = argv[1];
    Graph graph = constructGraph(filename);


    auto start_time = std::chrono::high_resolution_clock::now();
    double modularity = modularity_optimization(graph);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Modularity optimization completed." << std::endl;
    std::cout << "Final modularity: " << modularity << std::endl;
    std::cout << "Execution time: " << duration.count() << " s" << std::endl;

    delete[] graph.row_ptr;
    delete[] graph.col_idx;
    delete[] graph.values;
    return 0;
}