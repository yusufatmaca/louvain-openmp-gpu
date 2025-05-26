import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import coo_matrix

input_file = "wiki-Talk.txt.gz"
output_file = "wiki-Talk.mtx"

edges = pd.read_csv(
    input_file,
    delimiter="\t",
    comment="#",
    names=["FromNodeId", "ToNodeId"],
    dtype={"FromNodeId": int, "ToNodeId": int},
)


num_nodes = edges[["FromNodeId", "ToNodeId"]].max().max() + 1
row = edges["FromNodeId"].values
col = edges["ToNodeId"].values
data = [1] * len(edges)
adj_matrix = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

mmwrite(output_file, adj_matrix)

print(f"Converted {input_file} to {output_file}")

