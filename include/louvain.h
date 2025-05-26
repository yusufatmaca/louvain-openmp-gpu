struct DebugInfo {
    int vertex_id;
    int old_community;
    int new_community;
    double gain;
};

// Structure to hold community information
struct CommunityInfo {
    int size;          // Number of vertices in this community
    int degree;        // Total degree of vertices in this community
    int new_id;        // New ID after renumbering
    int vertex_start;  // Starting position in vertex array
    int edge_start;    // Starting position in edge array
};

// Structure for the condensed graph
struct CondensedGraph {
    int num_communities;
    int num_edges;
    int* row_ptr;
    int* col_idx;
    double* values;
    int* community_map;  // Maps old community IDs to new vertex IDs
};

struct VertexSet {
    int* vertex_ids;    // Array of vertex IDs in this set
    int size;           // Number of vertices in this set
};


double louvain(Graph& graph);
double calculateTotalWeight(Graph* graph);
void calculateDegrees(Graph& graph, int* degrees);
void calculateCommunityWeight(Graph* graph, int* community, double* community_weight, int num_communities);
void calculateMaxDegree(int* row_ptr, int num_vertices, int* result);
int calculateMinDegree(Graph& graph);
void calculateBucketBoundaries(Graph& graph, int numDBuckets, int* bucDSize);
int nextPrime(int n);
void computeMoveParallel(int* vertices,         int num_vertices,   Graph& graph,
                         int* communities,      double* community_strength,
                         int* vertex_strength,  double m,
                         int* best_communities, double* best_gains);