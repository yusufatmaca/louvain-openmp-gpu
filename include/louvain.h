struct ComputeMove {
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
void computeMoveParallel(int* vertices, int num_vertices_in_bucket,
                        Graph& graph,
                        int* communities,
                        double* community_strength,
                        double* vertex_strength,
                        ComputeMove& compute_moves, double m);