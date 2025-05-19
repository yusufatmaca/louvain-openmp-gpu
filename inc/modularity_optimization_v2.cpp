#include <omp.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include "csr_graph.h"
#include "utils.h"
#include "modularity_optimization.h"

double modularity_optimization(Graph& graph, int* community, double threshold) {
    // **Total Number of Vertices**: $\sum_{i \in V} 1$
    int num_vertices = graph.num_vertices;

    // **Vertex Strength**: $k_i = \sum_{j \in N[i]} w_{i, j}$
    double* vertex_strength = new double[num_vertices]();
    
    // **Community Strength**: $a_c = \sum_{i \in C} k_i$
    double* community_strength = new double[num_vertices]();

    // **Total Edge Weight**: $m = \sum_{e \in E} w_e$
    double m = 0.0;

    // **Intra-community Edge Weight**: $e_{i \to C(i)} = \sum_{j \in C(i)} w_{i, j}$
    double* e_i_c = new double[num_vertices]();
    
    bool changed = true;
    double final_modularity = 0.0;

    //int nteams = -1;
    //int nthreads = -1;


    // Mapping variables into the device data environment **without transferring execution to the device**
    #pragma omp target data \
            map(to: graph.row_ptr[0:(num_vertices + 1)], \
                    graph.col_idx[0:graph.num_edges], \
                    graph.values[0:graph.num_edges], \
                    e_i_c[0:num_vertices]) \
            map(tofrom: community[0:num_vertices], \
                        vertex_strength[0:num_vertices], \
                        community_strength[0:num_vertices], \
                        m, changed, final_modularity/*, nteams, nthreads*/)
    {

        /*
        #pragma omp target parallel for
        for (int i = 0; i < 128; i++) {
            if (omp_get_team_num() == 0 && omp_get_thread_num() == 0) {
                // Get the number of teams and threads
                nteams = omp_get_num_teams(); 
                nthreads = omp_get_num_threads();
                printf("Running on device with %d teams in total and %d threads in each team\n", nteams, nthreads);
            }
        }*/

        #pragma omp target teams distribute parallel for reduction(+:m)
        for (int i = 0; i < num_vertices; i++) {
            /*
            printf("Thread %d of %d is processing iteration %d\n", 
           omp_get_thread_num(), omp_get_num_threads(), i);*/


            double degree = 0.0;
            for (int j = graph.row_ptr[i]; j < graph.row_ptr[i + 1]; j++) {
                degree += graph.values[j];
            }
            vertex_strength[i] = degree;
            m += degree;
        }
        
        #pragma omp target update from(m /*, nteams, nthreads*/)
        m /= 2.0;
        std::cout << "Total edge weight (m): " << m << std::endl;
        //std::cout << "Number of teams: " << nteams << std::endl;
        //std::cout << "Number of threads: " << nthreads << std::endl;

        // Initially, each vertex starts in its own community
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < num_vertices; i++) {
            community[i] = i;
            community_strength[i] = vertex_strength[i];
        }

        int iteration = 0;
        while (changed) {
            changed = false;
            double modularity_gain = 0.0;

            #pragma omp target teams distribute parallel for schedule(dynamic) reduction(+:modularity_gain) reduction(||:changed)
            for (int i = 0; i < num_vertices; i++) {
                int current_comm = community[i];

                // Reset e_i_c for this vertex (sequential reset)
                for (int j = 0; j < num_vertices; j++) {
                    e_i_c[j] = 0.0;
                }

                for (int j = graph.row_ptr[i]; j < graph.row_ptr[i + 1]; j++) {
                    int neighbor = graph.col_idx[j];
                    int comm = community[neighbor];
                    e_i_c[comm] += graph.values[j];
                }

                int best_comm = current_comm;
                double max_gain = 0.0;

                for (int comm = 0; comm < num_vertices; comm++) {
                    if (comm == current_comm) continue;
                    double k = e_i_c[comm];
                    double gain = (k / m) - (vertex_strength[i] * community_strength[comm]) / (m * m);
                    if (gain > max_gain) {
                        max_gain = gain;
                        best_comm = comm;
                    }
                }

                if (best_comm != current_comm && max_gain > 0) {
                    #pragma omp critical
                    {
                        community[i] = best_comm;
                        changed = true;
                        modularity_gain += max_gain;
                        community_strength[current_comm] -= vertex_strength[i];
                        community_strength[best_comm] += vertex_strength[i];
                    }
                }
            }
            iteration++;
            std::cout << "Iteration " << iteration << ": modularity_gain = " << modularity_gain << std::endl;
            final_modularity += modularity_gain;
            if (modularity_gain < threshold) changed = false;
        }
    }

    delete[] vertex_strength;
    delete[] community_strength;
    delete[] e_i_c;

    return final_modularity;
}