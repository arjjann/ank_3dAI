
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from pathlib import Path


def load_graph(path):
    arrays = np.load(path)

    x = torch.tensor(arrays["face_features"], dtype=torch.float32)
    edge_index = torch.tensor(arrays["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(arrays["edge_features"], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def visualize_graph(graph):
    G = nx.Graph()

    # Add each face as a graph node
    num_faces = graph.x.shape[0]
    G.add_nodes_from(range(num_faces))

    # Add connections between adjacent faces
    edges = graph.edge_index.numpy().T
    G.add_edges_from(edges)

    # Generate node positions
    pos = nx.spring_layout(G, seed=42)

    # Draw graph
    plt.figure(figsize=(10, 8))

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=1000,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        width=2
    )

    plt.title("B-Rep Face Adjacency Graph")
    plt.show()


if __name__ == "__main__":

    graph_path = Path("../../data/graphs/plate_two_holes.npz")

    graph = load_graph(graph_path)

    print(graph)
    print("Number of faces:", graph.x.shape[0])
    print("Number of graph connections:", graph.edge_index.shape[1])

    visualize_graph(graph)
