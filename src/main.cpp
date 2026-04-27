#include "include/csr_graph.h"
#include "include/louvain.h"
#include "include/utils.h"

int main(int argc, char* argv[]) {
    
    // Check command line arguments
    if (checkArgs(argc, argv) != 1) {
        return -1;
    }
    
    // Check if OpenMP can handle offloaded to GPU
    if (checkDevice() != 1) {
        return -1;
    }

    const char* mtx_file = argv[1];
    Graph graph = constructGraph(mtx_file);

    //auto start_time = std::chrono::high_resolution_clock::now();
    double modularity = louvain(graph);
    //auto end_time = std::chrono::high_resolution_clock::now();

    
    //std::cout << "Final modularity: " << modularity << std::endl;
    //std::cout << "Execution time: " << duration.count() << " s" << std::endl;

    return 0;
}
