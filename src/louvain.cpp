#include <omp.h>
#include <iostream>
#include "csr_graph.h"
#include "utils.h"
#include "louvain.h"
#include <limits.h>
#include <cmath>

#define MAX_HASH_SIZE 4096

void parallel_prefix_sum(int* arr, int n) {
    // Simple sequential prefix sum on GPU
    // For better performance, use a more sophisticated parallel algorithm
    #pragma omp target teams num_teams(1) thread_limit(1) \
        map(tofrom: arr[0:n])
    {
        for (int i = 1; i < n; i++) {
            arr[i] += arr[i-1];
        }
    }
}



VertexSet* partition(double* vertex_strength, int num_vertices,
                     int bucket_k, int* buckDSize) {
    int lower_bound = (bucket_k > 0) ? buckDSize[bucket_k - 1] : 0;
    int upper_bound = buckDSize[bucket_k];

    int count = 0;

    #pragma omp target teams distribute parallel for reduction(+:count) \
        map(to: vertex_strength[0:num_vertices], lower_bound, upper_bound, bucket_k)
    for (int i = 0; i < num_vertices; i++) {
        double degree = vertex_strength[i];
        // Special case for first bucket to include degree 0 vertices
        if (bucket_k == 0) {
            if (degree >= lower_bound && degree <= upper_bound) {
                count++;
            }
        } else {
            if (degree > lower_bound && degree <= upper_bound) {
                count++;
            }
        }
    }

    VertexSet* vSet = new VertexSet;
    vSet->size = count;

    if (count == 0) {
        vSet->vertex_ids = nullptr;
        return vSet;
    }

    vSet->vertex_ids = new int[count];

    char* in_bucket = new char[num_vertices]();     // 0 or 1 for each vertex
    int* positions = new int[num_vertices]();       // Position in output array

    #pragma omp target data \
        map(to: vertex_strength[0:num_vertices], lower_bound, upper_bound) \
        map(tofrom: in_bucket[0:num_vertices], positions[0:num_vertices]) \
        map(from: vSet->vertex_ids[0:count])
    {
        // Mark vertices that belong to this bucket
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < num_vertices; i++) {
            int degree = vertex_strength[i];
            // Special case for first bucket to include degree 0 vertices
            if (bucket_k == 0) {
                in_bucket[i] = (degree >= lower_bound && degree <= upper_bound) ? 1 : 0;
            } else {
                in_bucket[i] = (degree > lower_bound && degree <= upper_bound) ? 1 : 0;
            }
        }

        // Compute exclusive prefix sum to get positions
        // Note: For large-scale parallel prefix sum, you might want to use a more
        // sophisticated algorithm, but for now we'll use a simple sequential scan
        #pragma omp target teams num_teams(1) thread_limit(1)
        {
            int running_sum = 0;
            for (int i = 0; i < num_vertices; i++) {
                positions[i] = running_sum;
                running_sum += in_bucket[i];
            }
        }

        // Collect vertices into the output array
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < num_vertices; i++) {
            if (in_bucket[i]) {
                vSet->vertex_ids[positions[i]] = i;
            }
        }
    }

    delete[] in_bucket;
    delete[] positions;

    return vSet;
}

void calculateBucketBoundaries(Graph& graph, int numDBuckets, int* bucDSize) {
    int max_degree = graph.max_degree;
    int min_degree = graph.min_degree;

    // First bucket always starts at 0
    bucDSize[0] = 0;

    // Choose distribution strategy based on graph characteristics
    double degree_ratio = (double)max_degree / min_degree;

    if (degree_ratio > 100) {
        // For highly skewed graphs, using logarithmic distribution may be preferable
        for (int i = 1; i < numDBuckets; i++) {
            bucDSize[i] = (int)(pow(max_degree, (double)i/numDBuckets));
        }
        bucDSize[numDBuckets] = max_degree + 1;
    } else if (degree_ratio > 10) {
        // For moderately skewed graphs, using square root scaling may be prefarable
        double sqrt_min = sqrt(min_degree);
        double sqrt_max = sqrt(max_degree);
        double sqrt_range = sqrt_max - sqrt_min;

        for (int i = 1; i < numDBuckets; i++) {
            double sqrt_val = sqrt_min + (i * sqrt_range) / numDBuckets;
            bucDSize[i] = (int)round(sqrt_val * sqrt_val);
        }
    }
    else {
        // For more uniform degree distributions, using linear distribution may be sufficient
        double linear_range = max_degree - min_degree;
        for (int i = 1; i < numDBuckets; i++) {
            double linear_val = min_degree + (i * linear_range) / numDBuckets;
            bucDSize[i] = (int)round(linear_val);
        }
    }
    // Last bucket should include the maximum degree
    bucDSize[numDBuckets] = max_degree + 1;
    // Ensure bucket boundaries are strictly increasing
    for (int i = 1; i <= numDBuckets; i++) {
        if (bucDSize[i] <= bucDSize[i - 1]) {
            bucDSize[i] = bucDSize[i - 1] + 1;
        }
    }
    std::cout << "Bucket boundaries: ";
    for (int i = 0; i <= numDBuckets; i++) {
        std::cout << bucDSize[i];
        if (i < numDBuckets) std::cout << ", ";
    }
    std::cout << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;


}

int nextPrime(int n) {
    if (n <= 1) return 2;

    bool found = false;
    while (!found) {
        found = true;
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                found = false;
                break;
            }
        }
        if (!found) n++;
    }
    return n;
}

// Compare-and-swap operation for GPU (using OpenMP atomics)
inline bool atomicCAS(int* address, int expected, int desired) {
    int old_val;
    #pragma omp critical
    {
        old_val = *address;
        if (*address == expected) {
            *address = desired;
        }
    }
    return (old_val == expected);
}


// Double hashing function for collision resolution
inline int doubleHash(int community_id, int iteration, int table_size) {
    // First hash function
    int h1 = community_id % table_size;
    // Second hash function (must be coprime with table_size)
    int h2 = 1 + (community_id % (table_size - 1));
    // Combined hash with iteration
    return (h1 + iteration * h2) % table_size;
}

// computeMove function needs to be inlined or its contents moved to computeMoveParallel
// to avoid host allocations on the device.
// We are moving the content of computeMove into computeMoveParallel directly.

void computeMoveParallel(int* vertices, int num_vertices_in_bucket,
                        Graph& graph,
                        int* communities,
                        double* community_strength,
                        double* vertex_strength,
                        ComputeMove* compute_moves, double m) {
    
    int* all_hashComm = new int[num_vertices_in_bucket * MAX_HASH_SIZE];
    double* all_hashWeight = new double[num_vertices_in_bucket * MAX_HASH_SIZE];
    
    // Initialize to -1 and 0.0 on host (optional, can be done on device)
    std::fill(all_hashComm, all_hashComm + num_vertices_in_bucket * MAX_HASH_SIZE, -1);
    std::fill(all_hashWeight, all_hashWeight + num_vertices_in_bucket * MAX_HASH_SIZE, 0.0);

    #pragma omp target teams distribute parallel for \
        map(to: \
            vertices[0:num_vertices_in_bucket], \
            num_vertices_in_bucket, \
            graph.row_ptr[0:(graph.num_vertices + 1)], \
            graph.col_idx[0:graph.num_edges], \
            graph.values[0:graph.num_edges]) \
            communities[0:graph.num_vertices], \
            community_strength[0:graph.num_vertices], \
            vertex_strength[0:graph.num_vertices], m, \
            all_hashComm[0:num_vertices_in_bucket * MAX_HASH_SIZE], \
            all_hashWeight[0:num_vertices_in_bucket * MAX_HASH_SIZE]) \
        map(tofrom: compute_moves[0:num_vertices_in_bucket])
    for (int idx = 0; idx < num_vertices_in_bucket; idx++) {
        int vertex = vertices[idx];

        int* hashComm = &all_hashComm[idx * MAX_HASH_SIZE];
        double* hashWeight = &all_hashWeight[idx * MAX_HASH_SIZE];

        // --- Start of computeMove logic (now inlined) ---
        // Get current community and vertex strength
        int current_community = communities[vertex];
        double k_i = (double)vertex_strength[vertex];  // Cast to double for calculations

        // Initialize outputs (local to each thread/iteration)
        int local_best_community = current_community;  // Default: stay in current community
        double local_best_gain = 0.0;

        compute_moves[idx].vertex_id = vertex;
        compute_moves[idx].old_community = current_community;
        compute_moves[idx].new_community = current_community;
        compute_moves[idx].gain = 0.0;
        
        // If isolated vertex, no move needed
        if (k_i == 0) {
            continue; // Move to the next vertex
        }

        int degree = (int)(k_i / 2.0);  // Approximate degree
        int desired_hash_size = nextPrime((int)(1.5 * degree) + 1);
        int hash_size = (desired_hash_size < MAX_HASH_SIZE) ? desired_hash_size : MAX_HASH_SIZE;

        // Initialize hash tables (-1 represents empty slot)
        for (int i = 0; i < hash_size; i++) {
            hashComm[i] = -1;
            hashWeight[i] = 0.0;
        }

        // Step 1: Build hash table of neighboring communities and their edge weights
        for (int edge_idx = graph.row_ptr[vertex]; edge_idx < graph.row_ptr[vertex + 1]; edge_idx++) {
            int neighbor = graph.col_idx[edge_idx];
            double edge_weight = (double)graph.values[edge_idx];  // Ensure double precision
            int neighbor_community = communities[neighbor];

            // Find position in hash table using double hashing
            int iteration = 0;
            int pos;
            bool found = false;

            while (iteration < hash_size) {
                pos = doubleHash(neighbor_community, iteration, hash_size);

                if (hashComm[pos] == neighbor_community) {
                    // Community already in hash table - add weight
                    hashWeight[pos] += edge_weight;
                    found = true;
                    break;
                } else if (hashComm[pos] == -1) {
                    // Empty slot - try to claim it
                    // No need for atomicCAS here, as hashComm is private to this thread
                    hashComm[pos] = neighbor_community;
                    hashWeight[pos] += edge_weight;
                    found = true;
                    break;
                }
                iteration++;
            }
            if (!found) {
                // This means the hash table is effectively "full" or has a bad collision pattern.
                // Could indicate nextPrime isn't picking good primes, or hash_size is too small.
                // For simplicity, we'll continue, but a robust solution might resize or use a different strategy.
                printf("Warning: Hash table full for vertex %d, neighbor_comm %d\n", vertex, neighbor_community);
            }
        }

        // Step 2: Calculate modularity gain for each neighboring community

        // Pre-calculate terms used in Equation 2
        double a_c_i = community_strength[current_community] - k_i;  // a_{C(i)\{i}}

        // Track best move
        double max_gain = 0.0;
        int target_community = current_community;

        // Find edge weight to current community (for e_{i→C(i)\{i}})
        double e_i_to_current = 0.0;
        for (int i = 0; i < hash_size; i++) {
            if (hashComm[i] == current_community) {
                e_i_to_current = hashWeight[i];
                break;
            }
        }

        // Evaluate each neighboring community
        for (int i = 0; i < hash_size; i++) {
            if (hashComm[i] != -1) { // Only process filled slots
                int comm = hashComm[i];
                double e_i_to_c = hashWeight[i];  // Edge weight from vertex to this community

                // Calculate modularity gain using Equation 2 from the paper
                // ΔQ = [e_{i→C} - e_{i→C(i)\{i}}] / m - k_i * [a_{C(i)\{i}} - a_C] / (2m²)
                double gain;
                if (comm == current_community) {
                    gain = 0.0; // Moving to same community has no gain
                } else {
                    double first_term = (e_i_to_c - e_i_to_current) / m;

                    double a_c = community_strength[comm];
                    double second_term = (k_i * (a_c_i - a_c)) / (2.0 * m * m); // Use 2.0 * m * m from global scope

                    gain = first_term + second_term;
                }

                // Update best move if this is better
                // Tie-breaking: prefer lower community ID (as per paper)
                if (gain > max_gain) {
                    max_gain = gain;
                    target_community = comm;
                } else if (gain == max_gain && comm < target_community) {
                    target_community = comm;
                }
            }
        }

        // Step 3: Store the best move for this vertex
        compute_moves[idx].new_community = target_community;
        compute_moves[idx].gain = max_gain;
        // --- End of computeMove logic ---

        // Clean up per-thread hash tables
    delete[] all_hashComm;
    delete[] all_hashWeight;


    }
}


double modularity_optimization(Graph& graph) {
    /*
        The *modularity optimization* phase of the *Louvain* method is designed to maximize the *modularity* of the graph.
        This is done by iteratively moving vertices between communities to find a configuration that maximizes the modularity score.
    */

    int num_vertices            =   graph.num_vertices;             // Total Number of Vertices := $|V|$
    int num_edges               =   graph.num_edges;                // Total Number of Edges:= $|E|$

    int max_degree              =   0;                              // Maximum Degree
    int min_degree              =   INT_MAX;                        // Minimum Degree

    int* communities            =   new int[num_vertices]();        // Community ID for each vertex
    double* community_strength  =   new double[num_vertices]();     // Community Strength := $a_c = \sum_{i \in C} k_i$

    /*
    #### From the paper
        Initially the algorithm computes the values of total edge weight, which is $m$,
        and in parallel for each $i \in V the value of $k_i$

    #### My approach
        Since I *assume* the implementation will only work on *unweighted graphs*, the *total edge weight will be equal to the total number of edges*.
    */

    graph.total_edge_weight     = 0;                                // Total Edge Weight (for *unweighted* graph) := $m = |E|$
    double m                    = 0.0;                                // Total Edge Weight := $m = \sum_{e \in E} w_e$

    double threshold_value      =   1e-6;                           // Threshold for convergence (fixed)
    double total_modularity_gain = 0.0;                             // Total Modularity Gain
    double modularity_gain      =   threshold_value + 1;            // Ensure we enter the loop
    // double constant_term        =   1.0 / (2.0 * m); // This term depends on m, so it should be calculated after m is finalized

    double* vertex_strength        = new double[num_vertices]();          // `vertex_strength` = $k_i$

    // Offload related datas to GPU
    #pragma omp target data \
        map(to: graph.row_ptr[0:(num_vertices + 1)], \
                graph.col_idx[0:graph.num_edges], \
                graph.values[0:graph.num_edges]) \
        map(tofrom: vertex_strength[0:num_vertices], m, max_degree, min_degree, \
                    communities[0:num_vertices], community_strength[0:num_vertices])
    {

        // Calculate vertex strengths
        #pragma omp target teams distribute parallel for reduction(+: m) reduction(max: max_degree) reduction(min: min_degree)
        for (int i = 0; i < num_vertices; i++) {
            // Calculate strength of *a* vertex
            int thread_local_strength = 0;
            for (int j = graph.row_ptr[i]; j < graph.row_ptr[i + 1]; j++) {
                thread_local_strength += graph.values[j];
            }
            vertex_strength[i] = thread_local_strength;

            // Update maximum through reduction - this happens automatically
            max_degree = (thread_local_strength > max_degree) ? thread_local_strength : max_degree;
            min_degree = (thread_local_strength < min_degree) ? thread_local_strength : min_degree;
            m += (double)thread_local_strength; // Update total edge weight
        }

        // Update host data
        #pragma omp target update from(m, max_degree, min_degree)
        graph.max_degree = max_degree;
        graph.min_degree = min_degree;
        graph.total_edge_weight = m / 2.0;                  // For unweighted graphs, total edge weight is half the sum of vertex strengths
        m /= 2.0; // Correct m for modularity gain calculation

        std::cout << "Total edge weight (graph.total_edge_weight): " << graph.total_edge_weight << std::endl;
        std::cout << "Total edge weight (m): " << m << std::endl;

        std::cout << "Maximum degree: " << max_degree << std::endl;
        std::cout << "Minimum degree: " << min_degree << std::endl;

        /*
            #### From the paper:
                Note that initially $a_{C(i)} = k_i$ as each vertex starts out as a community by itself.
        */
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < num_vertices; i++) {
            communities[i] = i;
        }

        #pragma omp target teams distribute parallel for
        for (int i = 0; i < num_vertices; i++) {
            community_strength[i] = (double)vertex_strength[i];
        }

        const int numDBuckets = 10; // Statically assigned number of buckets
        /*
        The array `buckDSize` stores the *boundary points* that separate our degree ranges into buckets.
        For `numDBuckets = 10`, we're dividing the range of vertex_strength into 10 distinct buckets,
        which requires 11 boundary points (10 + 1).
        */
        int* buckDSize = new int[numDBuckets + 1];
        calculateBucketBoundaries(graph, numDBuckets, buckDSize);

        int iter = 0;
        while (modularity_gain >= threshold_value && iter < 100) { // Limit iterations to prevent infinite loops

            std::cout << "Iteration " << (iter + 1) << " started" << std::endl;
            // The number of communities might change, so it's not simply graph.num_vertices
            // You'd need to count distinct communities after each iteration.
            // std::cout << "There are " << graph.num_vertices << " communities at the beginning of this iteration." << std::endl;

            modularity_gain = 0.0;
            int moves_made = 0;

            for (int k = 0; k <= numDBuckets; k++) { // Loop through buckets, starting from 0
                VertexSet* vSet = partition(vertex_strength, num_vertices, k, buckDSize);

                if (vSet->size == 0) {
                    delete[] vSet->vertex_ids;
                    delete vSet;
                    continue;
                }

                ComputeMove* computeMove = new ComputeMove[vSet->size];

                std::cout << "Processing bucket " << k << " with " << vSet->size << " vertices" << std::endl;

                // Call the modified computeMoveParallel
                computeMoveParallel(vSet->vertex_ids, vSet->size, graph,
                                   communities, community_strength, vertex_strength,
                                   computeMove, m);

                #pragma omp target update from(computeMove[0:vSet->size])
                // Process the computed moves to find the best community for each vertex
                for (int idx = 0; idx < vSet->size; idx++) {
                    std::cout << "Vertex " << computeMove[idx].vertex_id
                              << " old community: " << computeMove[idx].old_community
                              << ", new community: " << computeMove[idx].new_community
                              << ", gain: " << computeMove[idx].gain << std::endl;
                }
                delete[] computeMove;
            }

/*
                double bucket_gain = 0.0;
                int bucket_moves = 0;

                // Accumulate modularity gain and apply moves
                for (int idx = 0; idx < vSet->size; idx++) {
                    int vertex = vSet->vertex_ids[idx];

                    // Check if vertex is actually moving
                    if (communities[vertex] != best_communities[vertex]) {
                        bucket_moves++;
                        bucket_gain += best_gains[vertex];

                        // Debug output for first few moves
                        if (moves_made < 5) {
                            std::cout << "  Vertex " << vertex << " moves from community "
                                     << communities[vertex] << " to " << best_communities[vertex]
                                     << " with gain " << best_gains[vertex] << std::endl;
                        }
                    }
                }

                modularity_gain += bucket_gain;
                moves_made += bucket_moves;

                std::cout << "  Bucket " << k << ": " << bucket_moves << " moves, gain = "
                         << bucket_gain << std::endl;

                // Line 8-9 of Algorithm 1: Update communities
                // This update must happen *after* processing all vertices in the bucket
                // and *before* the next bucket or re-calculating community strengths.
                // Since best_communities is already updated from device,
                // now transfer it back to the 'communities' array on the device.

                #pragma omp target teams distribute parallel for \
                    map(to: vSet->vertex_ids[0:vSet->size], \
                            best_communities[0:num_vertices]) \
                    map(tofrom: communities[0:num_vertices])
                for (int idx = 0; idx < vSet->size; idx++) {
                    int vertex = vSet->vertex_ids[idx];
                    communities[vertex] = best_communities[vertex];
                }
                delete[] vSet->vertex_ids;
                delete vSet;
            }
            
            // Implementation of Algorithm 1, lines 10-11:
            // Re-compute community_strength for all communities
            // First, reset all community strengths to 0 on the device
            #pragma omp target teams distribute parallel for
            for (int i = 0; i < num_vertices; i++) {
                community_strength[i] = 0.0;
            }
            
            // Then, aggregate vertex strengths into their respective communities on the device
            #pragma omp target teams distribute parallel for
            for (int v = 0; v < num_vertices; v++) {
                int comm = communities[v];
                // Using atomic update since multiple vertices might belong to the same community
                // and try to update its strength concurrently.
                #pragma omp atomic update
                community_strength[comm] += (double)vertex_strength[v];
            }
            
            
            total_modularity_gain += modularity_gain;
            
            std::cout << "Iteration " << (iter + 1) << " completed" << std::endl;
            std::cout << "Total moves in this iteration: " << moves_made << std::endl;
            std::cout << "Modularity gain for this iteration: " << modularity_gain << std::endl;
            std::cout << "Total cumulative gain: " << total_modularity_gain << std::endl;
            */

            iter++;
        }
        delete[] buckDSize;

    }

    delete[] communities;
    delete[] community_strength;
    delete[] vertex_strength;

    return total_modularity_gain;
}

/**
 * Perform modularity optimization on the graph
 *
 * @param graph         Pointer to the Graph structure
 */
double louvain(Graph& graph) {
    /*
        The *Louvain* method is a *multi-level* algorithm for *community detection* in large networks.
        It consists of *two main phases*:
            - Modularity Optimization, and
            - Community Aggregation
    */
    double final_modularity = modularity_optimization(graph);
    return final_modularity;
}