    Graph* current_graph = &graph;
    Graph* condensed_graph = nullptr;

    double total_modularity = 0.0;
    int level = 0;
    bool improvement = true;

    int* final_communities = new int[graph.num_vertices];
    for (int i = 0; i < graph.num_vertices; i++) {
        final_communities[i] = i;
    }

    while (improvement && level < 10) {  // Limit levels to prevent infinite loops
        std::cout << "\n=== LEVEL " << level << " ===" << std::endl;
        std::cout << "Graph has " << current_graph->num_vertices << " vertices and " 
                  << current_graph->num_edges << " edges" << std::endl;

        double level_gain = modularity_optimization(*current_graph);
        total_modularity += level_gain;

        if (level_gain < 1e-6) {
            improvement = false;
            std::cout << "No significant improvement at level " << level << std::endl;
            break;
        }

        int* communities = new int[current_graph->num_vertices];

        std::cout << "\nStarting aggregation phase..." << std::endl;

        int* vertex_strength = new int[current_graph->num_vertices];
        #pragma omp target teams distribute parallel for \
            map(to: current_graph->row_ptr[0:(current_graph->num_vertices + 1)], \
                    current_graph->values[0:current_graph->num_edges]) \
            map(from: vertex_strength[0:current_graph->num_vertices])
        for (int i = 0; i < current_graph->num_vertices; i++) {
            int strength = 0;
            for (int j = current_graph->row_ptr[i]; j < current_graph->row_ptr[i + 1]; j++) {
                strength += current_graph->values[j];
            }
            vertex_strength[i] = strength;
        }

        CondensedGraph* condensed = contract(*current_graph, communities, vertex_strength);
        
        // Update final community assignments
        if (level > 0) {
            // Map through multiple levels of community assignments
            int* temp_communities = new int[graph.num_vertices];
            for (int i = 0; i < graph.num_vertices; i++) {
                // Follow the chain of community assignments
                int comm = final_communities[i];
                temp_communities[i] = condensed->community_map[communities[comm]];
            }
            delete[] final_communities;
            final_communities = temp_communities;
        } else {
            // First level - direct mapping
            for (int i = 0; i < graph.num_vertices; i++) {
                final_communities[i] = condensed->community_map[communities[i]];
            }
        }
        
        // Convert CondensedGraph to Graph structure for next iteration
        if (level > 0 && condensed_graph != nullptr) {
            // Clean up previous condensed graph
            delete[] condensed_graph->row_ptr;
            delete[] condensed_graph->col_idx;
            delete[] condensed_graph->values;
            delete condensed_graph;
        }
        
        // Create new Graph from CondensedGraph
        condensed_graph = new Graph;
        condensed_graph->num_vertices = condensed->num_communities;
        condensed_graph->num_edges = condensed->num_edges;
        condensed_graph->row_ptr = condensed->row_ptr;
        condensed_graph->col_idx = condensed->col_idx;
        condensed_graph->values = condensed->values;
        
        // Update current graph pointer
        current_graph = condensed_graph;
        
        // Cleanup
        delete[] communities;
        delete[] vertex_strength;
        delete[] condensed->community_map;
        delete condensed;
        
        level++;
        
        std::cout << "Level " << level << " completed. New graph has " 
                  << current_graph->num_vertices << " communities" << std::endl;
    }
    
    // Output final results
    std::cout << "\n=== FINAL RESULTS ===" << std::endl;
    std::cout << "Total levels: " << level << std::endl;
    std::cout << "Total modularity: " << total_modularity << std::endl;
    
    // Count final number of communities
    int num_communities = 0;
    for (int i = 0; i < graph.num_vertices; i++) {
        if (final_communities[i] > num_communities) {
            num_communities = final_communities[i];
        }
    }
    num_communities++;  // Since community IDs start from 0
    
    std::cout << "Number of communities found: " << num_communities << std::endl;
    
    // You might want to save or return the final community assignments
    // For now, we'll just clean up
    delete[] final_communities;
    
    // Clean up the last condensed graph if it exists
    if (condensed_graph != nullptr) {
        delete[] condensed_graph->row_ptr;
        delete[] condensed_graph->col_idx;
        delete[] condensed_graph->values;
        delete condensed_graph;
    }