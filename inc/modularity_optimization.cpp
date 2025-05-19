#include <omp.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include "csr_graph.h"
#include "utils.h"
#include "modularity_optimization.h"
#include <limits.h>
#include <cmath>

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

/**
 * Calculate boundaries for degree buckets using pre-computed min/max degrees
 * 
 * @param graph        Reference to the Graph structure containing min/max degree information
 * @param numDBuckets  Number of buckets to create
 * @param bucDSize     Array to store bucket boundaries (size numDBuckets+1)
 */
void calculateBucketBoundaries(
    Graph& graph,
    int numDBuckets,
    int* bucDSize) {

    int maxDegree = graph.max_degree;
    int minDegree = graph.min_degree;

    // First bucket always starts at 0
    bucDSize[0] = 0;

    // Choose distribution strategy based on graph characteristics
    double degree_ratio = (double)maxDegree / minDegree;
    
    if (degree_ratio > 100) {
        // For highly skewed graphs, using logarithmic distribution may be preferable
        double log_min = log(minDegree);
        double log_max = log(maxDegree);
        double log_range = log_max - log_min;
        
        for (int i = 1; i < numDBuckets; i++) {
            double log_val = log_min + (i * log_range) / numDBuckets;
            bucDSize[i] = (int)round(exp(log_val));
        }
    } else if (degree_ratio > 10) {
        // For moderately skewed graphs, using square root scaling may be prefarable
        double sqrt_min = sqrt(minDegree);
        double sqrt_max = sqrt(maxDegree);
        double sqrt_range = sqrt_max - sqrt_min;
        
        for (int i = 1; i < numDBuckets; i++) {
            double sqrt_val = sqrt_min + (i * sqrt_range) / numDBuckets;
            bucDSize[i] = (int)round(sqrt_val * sqrt_val);
        }
    }
    else {
        // For more uniform degree distributions, using linear distribution may be sufficient
        double linear_range = maxDegree - minDegree;
        for (int i = 1; i < numDBuckets; i++) {
            double linear_val = minDegree + (i * linear_range) / numDBuckets;
            bucDSize[i] = (int)round(linear_val);
        }
    }
    
    // Last bucket should include the maximum degree
    bucDSize[numDBuckets] = maxDegree + 1;

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
}

/**
 * Perform modularity optimization on the graph
 * 
 * @param graph         Pointer to the Graph structure
 */
double modularity_optimization(Graph& graph) {
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
    
    double final_modularity = 0.0;

    int max_degree = INT_MIN;
    int min_degree = INT_MAX;

    int* communities = new int[num_vertices];
    int* newComm = new int[num_vertices];

    // GPU OFFLOADING
    // Mapping variables into the device data environment **without transferring execution to the device**
    #pragma omp target data \
            map(to: graph.row_ptr[0:(num_vertices + 1)], \
                    graph.col_idx[0:graph.num_edges], \
                    graph.values[0:graph.num_edges]/*, \
                    e_i_c[0:num_vertices]*/) \
            map(tofrom: communities[0:num_vertices], \
                        vertex_strength[0:num_vertices], \
                        community_strength[0:num_vertices], \
                        newComm[0:num_vertices], \
                        m, final_modularity/*, nteams, nthreads*/, max_degree, min_degree)
    {
        /*
            Calculate: 
            - **vertex strength** (or **degree** in unweighted graph), which is $k_i$, and 
            - **total edge weight**, which is $m$

            Since we do not know the strengths of the vertices yet, we assign **one thread to each vertex**.
            Load balancing is very poor...

            _From the paper_:
                Initially the algorithm computes the values of $m$ and **in parallel** for each $i ∈ V$ the value of $k_i$.
        */
        #pragma omp target teams distribute parallel for reduction(+:m) reduction(max: max_degree) reduction(min: min_degree)
        for (int i = 0; i < num_vertices; i++) {
            double degree = 0.0;
            for (int j = graph.row_ptr[i]; j < graph.row_ptr[i + 1]; j++) {
                degree += graph.values[j];
            }
            vertex_strength[i] = degree;
            if (degree > max_degree) max_degree = degree;
            if (degree < min_degree) min_degree = degree;
            m += degree;
        }

        /* 
            _From the paper_:
                Note that initially $a_{C(i)} = k_i$ as **each vertex starts out as a community by itself**.
        */
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < num_vertices; i++) {
            communities[i] = i;
            community_strength[i] = vertex_strength[i];
        }

        /*
            _From the paper_:
                These values for which:
                    - total edge weight: $m$,
                    - vertex strength: $k_i$,
                    - community strength: $a_c{i}$ 
                are needed in subsequent evaluations of **Modularity Gain** equation. 
        */
        
        // Back to host memory to print out the related values
        #pragma omp target update from(m /*, nteams, nthreads*/, max_degree, min_degree)
        m /= 2.0;

        graph.max_degree = max_degree;
        graph.min_degree = min_degree;
        graph.total_edge_weight = m;

        std::cout << "Total edge weight (m): " << graph.total_edge_weight << std::endl;
        std::cout << "Max degree (max(k_i)): " << graph.max_degree << std::endl;
        std::cout << "Min degree (min(k_i)): " << graph.min_degree << std::endl;

        //std::cout << "Number of teams: " << nteams << std::endl;
        //std::cout << "Number of threads: " << nthreads << std::endl;

        
        /*
            _From the paper_:
                For each iteration the vertices are **divided into buckets** depending on their degrees.
                
                The variable `numDBuckets` holds the number of buckets,
                while the $k$th bucket contains vertices of degree ranging from
                `bucDSize[k − 1]` up to `bucDSize[k]`.

                To extract the vertices within a certain degree range
                we use the **Thrust** method `partition()`. (but _I implemented manually_) 

                This **reorders the elements of an array**
                so that those elements satisfying the given boolean condition
                **can easily be extracted to an array `vSet`**.

        */

        const int numDBuckets = 10; // Statically assigned number of buckets
        /*
        The array `buckDSize` stores the **boundary points** that separate our degree ranges into buckets.
        For `numDBuckets = 10`, we're dividing the range of degrees into 10 distinct buckets,
        which requires 11 boundary points (10+1).
        */
        int* buckDSize = new int[numDBuckets + 1]();
        calculateBucketBoundaries(graph, numDBuckets, buckDSize); // Calculate bucket boundaries

        #pragma omp target enter data map(to: buckDSize[0:(numDBuckets + 1)])
        
        double threshold = 1e-6;  // Threshold for convergence
        int max_iterations = 20;  // Maximum iterations **to prevent infinite loops**
        int iteration = 0;
        double total_gain = 0.0;
        
        int hash_table_size = nextPrime((int)(1.5 * max_degree) + 1);
        int* hashComm = new int[hash_table_size];
        double* hashWeight = new double[hash_table_size];
        
        bool changed = true;

        #pragma omp target enter data \
            map(alloc: hashComm[0:hash_table_size],\
                    hashWeight [0:hash_table_size])

        /*
            The outermost loop then iterates over the vertices 
            until the accumulated change in modularity
            during the iteration falls below a given threshold
        */            
        while (changed && iteration < max_iterations) {
            changed = false;
            double iteration_gain = 0.0;

            std::cout << "Iteration " << (iteration + 1) << " started" << std::endl;
            std::cout << "There are " << graph.num_vertices << " communities in this iteration." << std::endl;

            #pragma omp target teams distribute parallel for
            for (int i = 0; i < num_vertices; i++) {
                newComm[i] = communities[i];
            }

            for (int bucket = 1; bucket <= numDBuckets; bucket++) {
                int low_degree = buckDSize[bucket-1];
                int high_degree = buckDSize[bucket];
                int bucket_size = 0;
                #pragma omp target teams distribute parallel for reduction(+:bucket_size)
                for (int i = 0; i < num_vertices; i++) {
                    int degree = vertex_strength[i];
                    if (degree > low_degree && degree <= high_degree) {
                        bucket_size++;
                    }
                }

                if (bucket_size == 0) continue;

                std::cout << "  Processing bucket " << bucket << " with " << bucket_size 
                          << " vertices (degrees " << low_degree << " to " << high_degree << ")" << std::endl;

                /*
                    Manual partition instead of `thrust::partition`
                */
                int* vSet = new int[bucket_size];
                int count = 0;
                #pragma omp target update from(vertex_strength[0:num_vertices])
                for (int i = 0; i < num_vertices; i++) {
                    if (vertex_strength[i] > low_degree && vertex_strength[i] <= high_degree) {
                        vSet[count++] = i;
                    }
                }

                #pragma omp target enter data map(to: vSet[0:bucket_size])
                #pragma omp target teams distribute parallel for
                for (int idx = 0; idx < bucket_size; idx++) {
                    int vertex = vSet[idx];

                    for (int h = 0; h < hash_table_size; h++) {
                        hashComm[h] = -1;  // -1 indicates empty slot
                        hashWeight[h] = 0.0;
                    }

                    for (int j = graph.row_ptr[vertex]; j < graph.row_ptr[vertex+1]; j++) {
                        int neighbor = graph.col_idx[j];
                        int neighborComm = communities[neighbor];
                        double weight = graph.values[j];
                        bool found = false;
                        int it = 0;
                        while (!found) {
                            // Simple hash function (with linear probing)
                            int curPos = (neighborComm + it) % hash_table_size;
                            // Check if we found the community in the hash table
                            if (hashComm[curPos] == neighborComm) {
                                // Add weight (line 7 in Algorithm 2)
                                #pragma omp atomic
                                hashWeight[curPos] += weight;
                                found = true;
                            }
                            // If we found an empty slot
                            else if (hashComm[curPos] == -1) {
                                // Try to claim this slot (line 9 in Algorithm 2)
                                // Using atomic compare-and-swap
                                bool success = false;
                                #pragma omp critical
                                {
                                    if (hashComm[curPos] == -1) {
                                        hashComm[curPos] = neighborComm;
                                        success = true;
                                    }
                                }
                                
                                if (success) {
                                    hashWeight[curPos] = weight;
                                    found = true;
                                }
                                else if (hashComm[curPos] == neighborComm) {
                                    // Another thread claimed this slot for the same community
                                    #pragma omp atomic
                                    hashWeight[curPos] += weight;
                                    found = true;
                                }
                            }
                            
                            it++;
                            // Prevent infinite loops
                            if (it >= hash_table_size) {
                                // Fall back to linear search
                                for (int h = 0; h < hash_table_size; h++) {
                                    if (hashComm[h] == neighborComm) {
                                        #pragma omp atomic
                                        hashWeight[h] += weight;
                                        found = true;
                                        break;
                                    }
                                    else if (hashComm[h] == -1) {
                                        bool innerSuccess = false;
                                        #pragma omp critical
                                        {
                                            if (hashComm[h] == -1) {
                                                hashComm[h] = neighborComm;
                                                innerSuccess = true;
                                            }
                                        }
                                        if (innerSuccess) {
                                            hashWeight[h] = weight;
                                            found = true;
                                            break;
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                    
                    // Find best community (line 14-18 in Algorithm 2)
                    double bestGain = 0.0;
                    int bestComm = communities[vertex];
                    int currentComm = communities[vertex];
                    
                    // Calculate constants for Equation 2
                    double k_i = vertex_strength[vertex];
                    
                    // For each community in hash table
                    for (int h = 0; h < hash_table_size; h++) {
                        int targetComm = hashComm[h];
                        if (targetComm != -1) {
                            // Apply movement constraints from the paper
                            bool valid_move = (targetComm != currentComm);
                            
                            // Constraint for singleton communities: only move to lower index
                            if (currentComm == vertex && targetComm != vertex) {
                                valid_move = valid_move && (targetComm < currentComm);
                            }
                            
                            if (valid_move) {
                                // Calculate weights to current and target communities
                                double e_to_target = hashWeight[h];
                                double e_to_current = 0.0;
                                
                                for (int j = graph.row_ptr[vertex]; j < graph.row_ptr[vertex+1]; j++) {
                                    int n = graph.col_idx[j];
                                    if (communities[n] == currentComm && n != vertex) {
                                        e_to_current += graph.values[j];
                                    }
                                }
                                
                                // Calculate modularity gain using Equation 2
                                double a_current = community_strength[currentComm];
                                double a_target = community_strength[targetComm];
                                
                                double gain = (e_to_target - e_to_current) / m + 
                                              (k_i * (a_current - a_target - k_i)) / (2 * m * m);
                                
                                // Update best community if this gives better gain
                                if (gain > bestGain || (gain == bestGain && targetComm < bestComm)) {
                                    bestGain = gain;
                                    bestComm = targetComm;
                                }
                            }
                        }
                    }
                    
                    // Store result in newComm array
                    newComm[vertex] = (bestGain > 0) ? bestComm : currentComm;
                }


                
                // Update communities based on calculated moves
                double bucket_gain = 0.0;
                #pragma omp target teams distribute parallel for reduction(+:bucket_gain) reduction(||:changed)
                for (int idx = 0; idx < bucket_size; idx++) {
                    int vertex = vSet[idx];
                    int oldComm = communities[vertex];
                    int newCommunity = newComm[vertex];
                    
                    if (oldComm != newCommunity) {
                        // Calculate modularity gain from this move
                        double move_gain = 0.0;
                        for (int j = graph.row_ptr[vertex]; j < graph.row_ptr[vertex+1]; j++) {
                            int neighbor = graph.col_idx[j];
                            if (communities[neighbor] == newCommunity) {
                                move_gain += graph.values[j];
                            }
                            if (communities[neighbor] == oldComm && neighbor != vertex) {
                                move_gain -= graph.values[j];
                            }
                        }
                        
                        // Update community assignments
                        communities[vertex] = newCommunity;
                        
                        // Update community strengths atomically
                        #pragma omp atomic
                        community_strength[oldComm] -= vertex_strength[vertex];
                        #pragma omp atomic
                        community_strength[newCommunity] += vertex_strength[vertex];
                        
                        bucket_gain += move_gain;
                        changed = true;
                    }
                }
                #pragma omp target update from(changed)
                iteration_gain += bucket_gain;
                
                // Clean up vSet
                #pragma omp target exit data map(delete: vSet[0:bucket_size])
                delete[] vSet;
            }

            // Update total gain and check convergence
            total_gain += iteration_gain;
            if (iteration_gain <= threshold) {
                changed = false;
            }
            
            std::cout << "  Iteration " << iteration + 1 
                      << " completed with gain: " << iteration_gain << std::endl;

            if (changed && iteration_gain > threshold) {
                std::cout << "  Starting community aggregation phase..." << std::endl;
                
                int* comSize = new int[num_vertices]();
                double* comDegree = new double[num_vertices]();
        
                #pragma omp target enter data map(to: comSize[0:num_vertices], comDegree[0:num_vertices])
        
                // Initialize arrays on device
                #pragma omp target teams distribute parallel for
                for (int i = 0; i < num_vertices; i++) {
                    comSize[i] = 0;
                    comDegree[i] = 0.0;
                }

                #pragma omp target teams distribute parallel for
                for (int i = 0; i < num_vertices; i++) {
                    int comm = communities[i];
                    #pragma omp atomic
                    comSize[comm]++;
                    
                    #pragma omp atomic
                    comDegree[comm] += vertex_strength[i];
                }

                int* newID = new int[num_vertices]();
                #pragma omp target enter data map(to: newID[0:num_vertices])

                // Mark non-empty communities
                #pragma omp target teams distribute parallel for
                for (int c = 0; c < num_vertices; c++) {
                    if (comSize[c] == 0) {
                        newID[c] = 0;
                    } else {
                        newID[c] = 1;
                    }
                }

                // Calculate new community IDs using prefix sum
                // Since parallel prefix sum is complex on GPU, we'll do this on the CPU
                #pragma omp target update from(newID[0:num_vertices])
                
                int new_vertex_count = 0;
                for (int i = 0; i < num_vertices; i++) {
                    int temp = newID[i];
                    newID[i] = new_vertex_count;
                    new_vertex_count += temp;
                }

                #pragma omp target update to(newID[0:num_vertices])
                std::cout << "    Compressed " << num_vertices << " vertices into " 
              << new_vertex_count << " communities" << std::endl;

              // Step 3: Set up data structures for the new graph (lines 13-16 in Algorithm 3)
    
                // Copy comDegree to edgePos for prefix sum
                int* edgePos = new int[num_vertices + 1]();
                int* vertexStart = new int[num_vertices + 1]();

                #pragma omp target update from(comSize[0:num_vertices], comDegree[0:num_vertices])
                for (int i = 0; i < num_vertices; i++) {
                    edgePos[i] = comDegree[i];
                }

                int total_edges = 0;
                for (int i = 0; i < num_vertices; i++) {
                    int pos = total_edges;
                    total_edges += (int)edgePos[i];
                    edgePos[i] = pos;
                }
                edgePos[num_vertices] = total_edges;

                // Calculate starting positions for vertex arrays
                vertexStart[0] = 0;
                for (int i = 0; i < num_vertices; i++) {
                    vertexStart[i+1] = vertexStart[i] + comSize[i];
                }
                #pragma omp target enter data map(to: edgePos[0:num_vertices+1], vertexStart[0:num_vertices+1])
    
    // Step 4: Create temporary array for vertex assignments (lines 17-19 in Algorithm 3)
    int* com = new int[num_vertices]();
    #pragma omp target enter data map(alloc: com[0:num_vertices])
    
    // Create local copy of vertexStart for atomic updates
    int* local_vertex_start = new int[num_vertices]();
    for (int i = 0; i < num_vertices; i++) {
        local_vertex_start[i] = vertexStart[i];
    }
    
    #pragma omp target enter data map(to: local_vertex_start[0:num_vertices])
    
    // Assign vertices to ordered positions
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < num_vertices; i++) {
        int comm = communities[i];
        if (comSize[comm] > 0) {
            int pos;
            #pragma omp atomic capture
            {
                pos = local_vertex_start[comm];
                local_vertex_start[comm]++;
            }
            com[pos] = i;
        }
    }
    
    // Step 5: Create data structures for the new graph
    int new_num_vertices = new_vertex_count;
    int new_max_edges = total_edges;  // Upper bound on number of edges
    
    // Allocate memory for the new graph
    int* new_row_ptr = new int[new_num_vertices + 1]();
    int* new_col_idx = new int[new_max_edges]();
    double* new_values = new double[new_max_edges]();
    
    #pragma omp target enter data map(alloc: new_row_ptr[0:new_num_vertices+1], \
                                      new_col_idx[0:new_max_edges], \
                                      new_values[0:new_max_edges])
    
    // Initialize new row pointers
    #pragma omp target teams distribute parallel for
    for (int i = 0; i <= new_num_vertices; i++) {
        new_row_ptr[i] = 0;
    }
    
    // Step 6: Process communities to create new graph edges (mergeCommunity operation)
    // Define bucket sizes for community processing based on community sizes
    const int numCBuckets = 5;  // Fewer buckets than vertex processing since we have fewer communities
    int* bucCSize = new int[numCBuckets + 1]();
    
    // Calculate community bucket boundaries based on degree distribution
    bucCSize[0] = 0;
    if (new_num_vertices > 0) {
        double log_range = log(comDegree[num_vertices-1] + 1);
        for (int i = 1; i < numCBuckets; i++) {
            bucCSize[i] = exp(i * log_range / numCBuckets);
        }
        bucCSize[numCBuckets] = INT_MAX;
    }
    
    #pragma omp target enter data map(to: bucCSize[0:numCBuckets+1])
    
    // Process each bucket of communities
    for (int bucket = 1; bucket <= numCBuckets; bucket++) {
        int low_degree = bucCSize[bucket-1];
        int high_degree = bucCSize[bucket];
        
        // Count communities in this bucket
        int bucket_size = 0;
        for (int c = 0; c < num_vertices; c++) {
            if (comSize[c] > 0 && comDegree[c] > low_degree && comDegree[c] <= high_degree) {
                bucket_size++;
            }
        }
        
        if (bucket_size == 0) continue;
        
        // Collect communities in this bucket
        int* comSet = new int[bucket_size];
        int count = 0;
        for (int c = 0; c < num_vertices; c++) {
            if (comSize[c] > 0 && comDegree[c] > low_degree && comDegree[c] <= high_degree) {
                comSet[count++] = c;
            }
        }
        
        #pragma omp target enter data map(to: comSet[0:bucket_size])
        
        // Process each community in the bucket in parallel (mergeCommunity)
        #pragma omp target teams distribute parallel for
        for (int idx = 0; idx < bucket_size; idx++) {
            int comm = comSet[idx];
            int new_vertex = newID[comm];
            
            // Create temporary hash table for collecting edges
            int* edge_hash_comm = new int[new_num_vertices];
            double* edge_hash_weight = new double[new_num_vertices];
            
            // Initialize hash table
            for (int i = 0; i < new_num_vertices; i++) {
                edge_hash_comm[i] = -1;
                edge_hash_weight[i] = 0.0;
            }
            
            // Process all vertices in this community
            for (int v_idx = vertexStart[comm]; v_idx < vertexStart[comm] + comSize[comm]; v_idx++) {
                int vertex = com[v_idx];
                
                // For each neighbor of this vertex
                for (int e = graph.row_ptr[vertex]; e < graph.row_ptr[vertex+1]; e++) {
                    int neighbor = graph.col_idx[e];
                    int neighbor_comm = communities[neighbor];
                    double weight = graph.values[e];
                    
                    // Get new community ID
                    int target_new_id = newID[neighbor_comm];
                    
                    // Add weight to hash table
                    edge_hash_weight[target_new_id] += weight;
                    edge_hash_comm[target_new_id] = target_new_id;
                }
            }
            
            // Count edges for this community
            int edge_count = 0;
            for (int i = 0; i < new_num_vertices; i++) {
                if (edge_hash_comm[i] != -1) {
                    edge_count++;
                }
            }
            
            // Store row pointer for this vertex
            new_row_ptr[new_vertex + 1] = edge_count;
            
            // Store edges in new graph
            int edge_index = 0;
            for (int i = 0; i < new_num_vertices; i++) {
                if (edge_hash_comm[i] != -1) {
                    new_col_idx[edgePos[comm] + edge_index] = i;
                    new_values[edgePos[comm] + edge_index] = edge_hash_weight[i];
                    edge_index++;
                }
            }
            
            // Clean up temporary arrays
            delete[] edge_hash_comm;
            delete[] edge_hash_weight;
        }
        
        #pragma omp target exit data map(delete: comSet[0:bucket_size])
        delete[] comSet;
    }
    
    // Convert new_row_ptr from counts to offsets
    #pragma omp target update from(new_row_ptr[0:new_num_vertices+1])
    
    int offset = 0;
    for (int i = 0; i <= new_num_vertices; i++) {
        int count = new_row_ptr[i];
        new_row_ptr[i] = offset;
        offset += count;
    }
    
    #pragma omp target update to(new_row_ptr[0:new_num_vertices+1])
    
    #pragma omp target update from(new_row_ptr[0:new_num_vertices+1], \
                                  new_col_idx[0:new_max_edges], \
                                  new_values[0:new_max_edges])
    
    // Step 7: Replace old graph with new contracted graph
    int actual_edges = new_row_ptr[new_num_vertices];
    std::cout << "    New graph has " << new_num_vertices << " vertices and " 
              << actual_edges << " edges" << std::endl;
    
    // Clean up old graph data
    delete[] graph.row_ptr;
    delete[] graph.col_idx;
    delete[] graph.values;
    
    // Replace with new graph
    graph.num_vertices = new_num_vertices;
    graph.num_edges = actual_edges;
    graph.row_ptr = new int[new_num_vertices + 1];
    graph.col_idx = new int[actual_edges];
    graph.values = new double[actual_edges];
    
    // Copy data to new graph
    for (int i = 0; i <= new_num_vertices; i++) {
        graph.row_ptr[i] = new_row_ptr[i];
    }
    
    for (int i = 0; i < actual_edges; i++) {
        graph.col_idx[i] = new_col_idx[i];
        graph.values[i] = new_values[i];
    }
    
    // Clean up arrays for new graph
    delete[] new_row_ptr;
    delete[] new_col_idx;
    delete[] new_values;
    
    // Step 8: Reset for next iteration
    delete[] communities;
    delete[] newComm;
    delete[] vertex_strength;
    delete[] community_strength;
    
    // Reallocate arrays for the new graph size
    num_vertices = new_num_vertices;
    communities = new int[num_vertices];
    newComm = new int[num_vertices];
    vertex_strength = new double[num_vertices];
    community_strength = new double[num_vertices];
    
    #pragma omp target enter data map(to: communities[0:num_vertices], \
                                    newComm[0:num_vertices], \
                                    vertex_strength[0:num_vertices], \
                                    community_strength[0:num_vertices])
    
    // Initialize for next iteration - each vertex starts in its own community
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < num_vertices; i++) {
        communities[i] = i;
        
        // Calculate vertex strength for the new graph
        double degree = 0.0;
        for (int j = graph.row_ptr[i]; j < graph.row_ptr[i+1]; j++) {
            degree += graph.values[j];
        }
        vertex_strength[i] = degree;
        community_strength[i] = degree;
    }
    
    // Clean up temporary arrays
    #pragma omp target exit data map(delete: comSize[0:num_vertices], \
                                  comDegree[0:num_vertices], \
                                  newID[0:num_vertices], \
                                  edgePos[0:num_vertices+1], \
                                  vertexStart[0:num_vertices+1], \
                                  local_vertex_start[0:num_vertices], \
                                  com[0:num_vertices], \
                                  bucCSize[0:numCBuckets+1])
    
    delete[] comSize;
    delete[] comDegree;
    delete[] newID;
    delete[] edgePos;
    delete[] vertexStart;
    delete[] local_vertex_start;
    delete[] com;
    delete[] bucCSize;
    
    std::cout << "  Community aggregation phase completed" << std::endl;
    
    // Reset changed flag to ensure we continue with at least one more iteration
    changed = true;
}


            iteration++;
        }
        
        // Calculate final modularity
        final_modularity = 0.0;
        #pragma omp target teams distribute parallel for reduction(+:final_modularity)
        for (int i = 0; i < num_vertices; i++) {
            int comm_i = communities[i];
            double e_in_comm = 0.0;
            
            for (int j = graph.row_ptr[i]; j < graph.row_ptr[i+1]; j++) {
                int neighbor = graph.col_idx[j];
                if (communities[neighbor] == comm_i) {
                    e_in_comm += graph.values[j];
                }
            }
            
            final_modularity += e_in_comm / (2 * m) - 
                               (vertex_strength[i] * community_strength[comm_i]) / (4 * m * m);
        }
        
        std::cout << "Modularity optimization completed in " << iteration << " iterations" << std::endl;
        //std::cout << "Final modularity: " << final_modularity << std::endl;
        
        // Count communities
        int community_count = 0;
        #pragma omp target teams distribute parallel for reduction(+:community_count)
        for (int i = 0; i < num_vertices; i++) {
            if (community_strength[i] > 0) {
                community_count++;
            }
        }
        
        std::cout << "Number of communities: " << community_count << std::endl;
        
        // Clean up device memory
        #pragma omp target exit data map(delete: hashComm[0:hash_table_size], hashWeight[0:hash_table_size])
        #pragma omp target exit data map(delete: buckDSize[0:(numDBuckets + 1)])
    }
    
    // Clean up host memory
    delete[] communities;
    delete[] newComm;
    delete[] vertex_strength;
    delete[] community_strength;
    delete[] e_i_c;
    
    return final_modularity;
}