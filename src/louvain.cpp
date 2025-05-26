#include <omp.h>
#include <iostream>
#include "csr_graph.h"
#include "utils.h"
#include "louvain.h"
#include <limits.h>
#include <cmath>

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



VertexSet* partition(int* vertex_strength, int num_vertices,
                     int bucket_k, int* buckDSize) {
    int lower_bound = (bucket_k > 0) ? buckDSize[bucket_k - 1] : 0;
    int upper_bound = buckDSize[bucket_k];

    int count = 0;

    #pragma omp target teams distribute parallel for reduction(+:count) \
        map(to: vertex_strength[0:num_vertices], lower_bound, upper_bound, bucket_k)
    for (int i = 0; i < num_vertices; i++) {
        int degree = vertex_strength[i];
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

void computeMoveParallel(int* vertices,         int num_vertices /*# num of vertices in the current bucket*/,
                         Graph& graph,
                         int* communities,      double* community_strength,
                         int* vertex_strength,  double m,
                         int* best_communities, double* best_gains,
                         DebugInfo* debug_info) {

    int count = 0;

    #pragma omp target parallel for \
            map(tofrom: \
                vertices[0:num_vertices], \
                num_vertices, \
                communities[0:graph.num_vertices], \
                community_strength[0:graph.num_vertices], \
                vertex_strength[0:graph.num_vertices], \
                graph.row_ptr[0:(graph.num_vertices + 1)], \
                graph.col_idx[0:graph.num_edges], \
                graph.values[0:graph.num_edges], m, \
                best_communities[0:graph.num_vertices], \
                best_gains[0:graph.num_vertices], \
                debug_info[0:num_vertices])
            /*map(to: vertices[0:num_vertices], \
                    vertex_strength[0: num_vertices]) \ */
        for (int idx = 0; idx < num_vertices; idx++) {
            int vertex = vertices[idx]; // Get the vertex ID from the current bucket
            int vertex_degree = vertex_strength[vertex];

            int best_community = 0;
            double best_gain = 0.0;

            if (vertex_degree == 0) {
                // Isolated vertex stays in its own community
                best_community = communities[vertex];
                best_gain = 0.0;
            } else {
            /*
                #### From the paper:
                     The size of the hash tables for `i` 
                     is drawn from a list of precomputed prime numbers 
                     as the smallest value larger than 1.5 times the degree of `i`.
            */
                int hash_size = nextPrime((int)(2 * vertex_degree) + 1);

                // Allocate hash tables
                int* hashComm = new int[hash_size];
                double* hashWeight = new double[hash_size];

                // Initialize hash tables (-1 represents `null`/`empty`)
                for (int i = 0; i < hash_size; i++) {
                    hashComm[i] = -1;
                    hashWeight[i] = 0.0;
                }

                // Current community of the vertex
                int current_comm = communities[vertex];
                int k_i = vertex_strength[vertex]; // Vertex Strength

                // Variables to track the best move
                double max_gain = 0.0;
                int target_community = current_comm;

                // Terms that are constant for all community comparisons
                double a_c_i = community_strength[current_comm] - k_i;  // a_{C(i)\{i}}
                double second_denominator = (2.0 * m * m);

                // Process all neighbors
                for (int edge_idx = graph.row_ptr[vertex]; edge_idx < graph.row_ptr[vertex + 1]; edge_idx++) {
                    int neighbor = graph.col_idx[edge_idx];
                    double edge_weight = graph.values[edge_idx];
                    int neighbor_comm = communities[neighbor];

                    // Find position in hash table using double hashing
                    int iteration = 0;
                    int pos;
                    bool found = false;

                    while (iteration < hash_size) {
                        pos = doubleHash(neighbor_comm, iteration, hash_size);

                        if (hashComm[pos] == neighbor_comm) {
                            // Community already in hash table - add weight
                            #pragma omp atomic
                            hashWeight[pos] += edge_weight;
                            found = true;
                            break;
                        } else if (hashComm[pos] == -1) {
                            // Empty slot - try to claim it
                            if (atomicCAS(&hashComm[pos], -1, neighbor_comm)) {
                                // Successfully claimed the slot
                                #pragma omp atomic
                                hashWeight[pos] += edge_weight;
                                found = true;
                                break;
                            }
                            // If CAS failed, another thread claimed it - continue searching
                        }
                        iteration++;
                    }

                    if (!found) {
                        // Hash table full (shouldn't happen with proper sizing)
                        printf("Warning: Hash table full for vertex %d\n", vertex);
                    }
                }


                // Now evaluate modularity gain for each neighboring community
                for (int i = 0; i < hash_size; i++) {
                    if (hashComm[i] != -1) {
                        int comm = hashComm[i];
                        // Intra-community edge weight to new community (e_{i→C(j)})
                        double e_i_to_c = hashWeight[i];

                        // Calculate modularity gain using Equation (2)
                        double gain;
                        if (comm == current_comm) {
                            // Moving to same community - no gain
                            gain = 0.0;
                        } else {
                            // First term: (e_{i→C(j)} - e_{i→C(i)\{i}}) / m
                            double e_i_to_current = 0.0;
                            if (current_comm != vertex) {  // If not in singleton community
                                // Find edges to current community (excluding self)
                                for (int j = 0; j < hash_size; j++) {
                                    if (hashComm[j] == current_comm) {
                                        e_i_to_current = hashWeight[j];
                                        break;
                                    }
                                }
                            }

                            double first_term = (e_i_to_c - e_i_to_current) / m;

                            // Second term: k_i * (a_{C(i)\{i}} - a_{C(j)}) / (2m²)
                            double a_c_j = community_strength[comm];
                            /*`k_i` and `a_c_i` were pre-calculated outside the current loop (line 199-200)*/
                            double second_term = k_i * (a_c_i - a_c_j) / second_denominator;

                            gain = first_term + second_term;
                        }

                        // Track best move
                        if (gain > max_gain || (gain == max_gain && comm < target_community)) {
                            max_gain = gain;
                            target_community = comm;
                        }
                    }
                }

                // Clean up hash tables
                delete[] hashComm;
                delete[] hashWeight;

                // Return results
                best_community = target_community;
                best_gain = max_gain;
            }
            
            // Assign outputs (unchanged)
            best_communities[vertex] = best_community;
            best_gains[vertex] = best_gain;
            /*
            communities[vertex] = best_community;
            community_strength[vertex] = vertex_strength[vertex];
            */
           
           debug_info[idx].vertex_id = vertex;
           debug_info[idx].old_community = communities[vertex];
           debug_info[idx].new_community = best_community;
           debug_info[idx].gain = best_gain;
        }
}

/*
    This function calculates the bucket boundaries for the given graph based on the degree distribution.
    The degree distribution is divided into `numDBuckets` buckets, and the boundaries are stored in `bucDSize`.
    */
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
    double m                    = 0;                                // Total Edge Weight := $m = \sum_{e \in E} w_e$

    double threshold_value      =   1e-6;                           // Threshold for convergence (fixed)
    double total_modularity_gain = 0.0;                             // Total Modularity Gain
    double modularity_gain      =   threshold_value + 1;            // Ensure we enter the loop
    double constant_term        =   1.0 / (2.0 * m);

    int* vertex_strength        = new int[num_vertices]();          // `vertex_strength` = $k_i$

    int* best_communities       = new int[num_vertices]();
    double* best_gains          = new double[num_vertices]();


    // Offload related datas to GPU
    #pragma omp target data                                                                             \
            map(to:                                                                                     \
                    graph.row_ptr[0:(num_vertices + 1)],                                                \
                    graph.col_idx[0:graph.num_edges],                                                   \
                    graph.values[0:graph.num_edges])                                                    \
            map(tofrom:                                                                                 \
                    vertex_strength[0:num_vertices], m, max_degree, min_degree, \
                    communities[0:num_vertices], community_strength[0:num_vertices], modularity_gain,   \
                    best_communities[0:num_vertices], best_gains[0:num_vertices])
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
            m += thread_local_strength; // Update total edge weight
        }

        // Update host data
        #pragma omp target update from(m, max_degree, min_degree)
        graph.max_degree = max_degree;
        graph.min_degree = min_degree;
        graph.total_edge_weight = m/2;                  // For unweighted graphs, total edge weight is half the sum of vertex strengths
        
        std::cout << "Total edge weight (m): " << graph.total_edge_weight << std::endl;
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
            community_strength[i] = vertex_strength[i];
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
        while (modularity_gain >= threshold_value) {
            
            std::cout << "Iteration " << (iter + 1) << " started" << std::endl;
            std::cout << "There are " << graph.num_vertices << " communities at the beginning of this iteration." << std::endl;
            
            modularity_gain = 0.0;
            
            for (int k = 1; k <= numDBuckets; k++) {
                VertexSet* vSet = partition(vertex_strength, num_vertices, k, buckDSize);
                std::cout << "There are " << vSet->size << " vertices in bucket: " << k << std::endl;
                
                if (vSet->size == 0) {
                    delete[] vSet->vertex_ids;
                    delete vSet;
                    continue;
                }
                
                DebugInfo* debug_info = new DebugInfo[vSet->size];

                computeMoveParallel(vSet->vertex_ids, vSet->size, graph,
                    communities, community_strength, vertex_strength,
                    m, best_communities, best_gains, debug_info);
                   
                // Update communities in parallel (lines 8-9 of Algorithm 1)
                #pragma omp target teams distribute parallel for \
                    map(to: vSet->vertex_ids[0:vSet->size], best_communities[0:num_vertices]) \
                    map(tofrom: communities[0:num_vertices]) \
                    map(tofrom: modularity_gain)
                for (int idx = 0; idx < vSet->size; idx++) {
                    int vertex = vSet->vertex_ids[idx];
                    communities[vertex] = best_communities[vertex];
                    #pragma omp atomic
                    modularity_gain += best_gains[vertex];
                }

                delete[] debug_info;
                delete[] vSet->vertex_ids;
                delete vSet;
            }

            #pragma omp target teams distribute parallel for
            for (int c = 0; c < num_vertices; c++) {
                community_strength[c] = 0.0;
            }

            // Then sum up vertex strengths for each community
            #pragma omp target teams distribute parallel for
            for (int v = 0; v < num_vertices; v++) {
                int comm = communities[v];
                #pragma omp atomic
                community_strength[comm] += vertex_strength[v];
            }

            #pragma omp target update from(modularity_gain)
            total_modularity_gain += modularity_gain;
            
            std::cout << "Iteration " << (iter + 1) << " completed." << std::endl;
            std::cout << "Modularity gain in this iteration: " << modularity_gain << std::endl;
            std::cout << "Total modularity gain so far: " << total_modularity_gain << std::endl;
            
            iter++;
        }
        delete[] buckDSize;
        
    }

    delete[] communities;
    delete[] community_strength;
    delete[] vertex_strength;
    delete[] best_communities;
    delete[] best_gains;
    return total_modularity_gain;
}

CondensedGraph* contract(Graph& graph, int* communities, int* vertex_strength) {
    int num_vertices = graph.num_vertices;
    
    // Step 1: Initialize community size and degree arrays
    int* comSize = new int[num_vertices]();
    int* comDegree = new int[num_vertices]();
    
    #pragma omp target data \
        map(to: communities[0:num_vertices], vertex_strength[0:num_vertices]) \
        map(tofrom: comSize[0:num_vertices], comDegree[0:num_vertices])
    {
        // Initialize to zero
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < num_vertices; i++) {
            comSize[i] = 0;
            comDegree[i] = 0;
        }
        
        // Step 2: Count vertices and degrees per community (lines 4-6)
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < num_vertices; i++) {
            int c = communities[i];
            #pragma omp atomic
            comSize[c] += 1;
            #pragma omp atomic
            comDegree[c] += vertex_strength[i];
        }
    }
    
    // Step 3: Create new community IDs (lines 7-11)
    int* newID = new int[num_vertices]();
    
    #pragma omp target data \
        map(to: comSize[0:num_vertices]) \
        map(tofrom: newID[0:num_vertices])
    {
        #pragma omp target teams distribute parallel for
        for (int c = 0; c < num_vertices; c++) {
            newID[c] = (comSize[c] == 0) ? 0 : 1;
        }
    }
    
    // Prefix sum to get new community IDs (line 12)
    parallel_prefix_sum(newID, num_vertices);
    
    // Get the total number of communities
    int num_communities;
    #pragma omp target update from(newID[num_vertices-1])
    num_communities = newID[num_vertices-1];
    
    // Step 4: Prepare for edge aggregation (lines 13-16)
    int* edgePos = new int[num_vertices + 1]();
    int* vertexStart = new int[num_vertices + 1]();
    
    #pragma omp target data \
        map(to: comDegree[0:num_vertices], comSize[0:num_vertices]) \
        map(from: edgePos[0:num_vertices+1], vertexStart[0:num_vertices+1])
    {
        // Copy community degrees to edgePos
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < num_vertices; i++) {
            edgePos[i] = comDegree[i];
            vertexStart[i] = comSize[i];
        }
        edgePos[num_vertices] = 0;
        vertexStart[num_vertices] = 0;
    }
    
    // Prefix sum for edge positions and vertex starts
    parallel_prefix_sum(edgePos, num_vertices + 1);
    parallel_prefix_sum(vertexStart, num_vertices + 1);
    
    // Step 5: Assign vertices to communities (lines 17-19)
    int* com = new int[num_vertices]();
    
    #pragma omp target data \
        map(to: communities[0:num_vertices], vertexStart[0:num_vertices]) \
        map(from: com[0:num_vertices])
    {
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < num_vertices; i++) {
            int c = communities[i];
            int res;
            #pragma omp atomic capture
            {
                res = vertexStart[c];
                vertexStart[c]++;
            }
            com[res] = i;
        }
    }
    
    // Step 6: Create the condensed graph structure
    CondensedGraph* condensed = new CondensedGraph;
    condensed->num_communities = num_communities;
    condensed->community_map = newID;
    
    // Now we need to merge communities and create the new graph
    // This part involves the bucket-based merging (lines 20-23)
    
    // Calculate bucket boundaries for communities
    int* bucCSize = new int[11];  // Assuming numCBuckets = 10
    
    // Find max community degree
    int max_com_degree = 0;
    #pragma omp target teams distribute parallel for reduction(max: max_com_degree) \
        map(to: comDegree[0:num_vertices])
    for (int c = 0; c < num_vertices; c++) {
        if (comDegree[c] > max_com_degree) {
            max_com_degree = comDegree[c];
        }
    }
    
    // Simple linear bucketing for communities
    for (int i = 0; i <= 10; i++) {
        bucCSize[i] = (max_com_degree * i) / 10;
    }
    bucCSize[10] = max_com_degree + 1;
    
    // Step 7: Merge communities
    // Note: The actual mergeCommunity function is complex and involves
    // hash table operations similar to computeMoveParallel
    // For brevity, I'll provide the structure here
    
    // Allocate space for new graph (this is an estimate)
    int estimated_edges = graph.num_edges;
    condensed->row_ptr = new int[num_communities + 1]();
    condensed->col_idx = new int[estimated_edges];
    condensed->values = new double[estimated_edges];
    
    // The actual merging would happen here using the bucket approach
    // This involves complex hash table operations similar to computeMoveParallel
    
    // Clean up temporary arrays
    delete[] comSize;
    delete[] comDegree;
    delete[] edgePos;
    delete[] vertexStart;
    delete[] com;
    delete[] bucCSize;
    
    return condensed;
}

void mergeCommunity(int community_id, Graph& graph, int* communities,
                   int* com, int com_start, int com_size,
                   CondensedGraph* condensed, int* edge_offset) {
    
    // Estimate hash table size based on community degree
    int estimated_neighbors = com_size * 10;  // Rough estimate
    int hash_size = nextPrime((int)(1.5 * estimated_neighbors) + 1);
    
    // Allocate hash tables for neighbor communities and weights
    int* hashComm = new int[hash_size];
    double* hashWeight = new double[hash_size];
    
    // Initialize hash tables
    for (int i = 0; i < hash_size; i++) {
        hashComm[i] = -1;
        hashWeight[i] = 0.0;
    }
    
    // Process all vertices in this community
    for (int idx = com_start; idx < com_start + com_size; idx++) {
        int vertex = com[idx];
        
        // Process all edges of this vertex
        for (int e = graph.row_ptr[vertex]; e < graph.row_ptr[vertex + 1]; e++) {
            int neighbor = graph.col_idx[e];
            double weight = graph.values[e];
            int neighbor_comm = communities[neighbor];
            
            // Skip self-loops at community level
            if (neighbor_comm == community_id) continue;
            
            // Insert into hash table using double hashing
            int iteration = 0;
            bool inserted = false;
            
            while (iteration < hash_size && !inserted) {
                int pos = doubleHash(neighbor_comm, iteration, hash_size);
                
                if (hashComm[pos] == neighbor_comm) {
                    #pragma omp atomic
                    hashWeight[pos] += weight;
                    inserted = true;
                } else if (hashComm[pos] == -1) {
                    if (atomicCAS(&hashComm[pos], -1, neighbor_comm)) {
                        #pragma omp atomic
                        hashWeight[pos] += weight;
                        inserted = true;
                    }
                }
                iteration++;
            }
        }
    }
    
    // Extract edges from hash table and add to condensed graph
    int edge_count = 0;
    for (int i = 0; i < hash_size; i++) {
        if (hashComm[i] != -1) {
            int pos = (*edge_offset) + edge_count;
            condensed->col_idx[pos] = hashComm[i];
            condensed->values[pos] = hashWeight[i];
            edge_count++;
        }
    }
    
    // Update edge offset
    #pragma omp atomic
    *edge_offset += edge_count;
    
    // Clean up
    delete[] hashComm;
    delete[] hashWeight;
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
    modularity_optimization(graph);
    return 0.0;
}
