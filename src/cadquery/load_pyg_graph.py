import numpy as np
import torch
from torch_geometric.data import Data

def load_graph(path):
    arrays = np.load(path)
    x = torch.tensor(arrays["face_features"], dtype=torch.float32)
    edge_index = torch.tensor(arrays["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(arrays["edge_features"], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

if __name__ == "__main__":
    graph = load_graph("graphs/plate_two_holes.npz")
    print(graph)
