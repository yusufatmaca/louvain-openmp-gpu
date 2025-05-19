double modularity_optimization(Graph& graph);
double calculateTotalWeight(Graph* graph);
int nextPrime(int n);
void calculateDegrees(Graph& graph, int* degrees);
void calculateCommunityWeight(Graph* graph, int* community, double* community_weight, int num_communities);
int calculateMaxDegree(Graph* graph);
int calculateMinDegree(Graph* graph);
void calculateBucketBoundaries(int* degrees, int num_vertices, int* bucDSize, int numDBuckets);