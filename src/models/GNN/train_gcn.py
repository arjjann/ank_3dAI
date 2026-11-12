
from pathlib import Path
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# 1. Project paths and settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]

GRAPH_DIR = PROJECT_ROOT / "data" / "graphs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

NUM_CLASSES = 7
INPUT_DIM = 10
HIDDEN_DIM = 32
# BATCH_SIZE=16
EPOCHS = 200
LEARNING_RATE = 0.01

torch.manual_seed(42)


# 2. Load all CAD graphs

def load_dataset(graph_dir):

    files = sorted(graph_dir.glob("*.npz"))

    if not files:
        raise RuntimeError(f"No NPZ files found in {graph_dir}")

    all_features = []
    all_labels = []
    all_edges = []

    offset = 0

    for file in files:

        arrays = np.load(file)

        face_features = arrays["face_features"]
        edge_index = arrays["edge_index"]

        # First 7 columns contain surface-type labels.
        labels = np.argmax(face_features[:, :7], axis=1)

        # Remaining 10 columns are model input features.
        features = face_features[:, 7:]

        all_features.append(features)
        all_labels.append(labels)

        # Shift node indices to combine separate graphs.
        all_edges.append(edge_index + offset)

        offset += face_features.shape[0]

        print(
            f"Loaded {file.name}: "
            f"{face_features.shape[0]} faces"
        )

    x = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    edge_index = np.concatenate(all_edges, axis=1)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return x, edge_index, y


# 3. Define the GCN model


class GCN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):

        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)

        return x


# 4. Prepare training, validation and test sets


def prepare_data(x, y):

    num_nodes = x.shape[0]

    if num_nodes < 3:
        raise RuntimeError("Not enough nodes for training.")

    indices = torch.randperm(num_nodes)

    train_end = int(0.70 * num_nodes)
    val_end = int(0.85 * num_nodes)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Normalize using training data only.
    mean = x[train_idx].mean(dim=0)
    std = x[train_idx].std(dim=0).clamp(min=1e-6)

    x = (x - mean) / std

    return x, train_idx, val_idx, test_idx, mean, std

# 5. Training and evaluation

def accuracy(logits, labels, indices):

    predictions = logits[indices].argmax(dim=1)

    return (
        predictions == labels[indices]
    ).float().mean().item()


def main():

    x, edge_index, y = load_dataset(GRAPH_DIR)

    print("\nDataset summary")
    print("-------------------------")
    print("Total faces:", x.shape[0])
    print("Total graph connections:", edge_index.shape[1])
    print("Node feature dimension:", x.shape[1])
    print("Number of classes:", NUM_CLASSES)

    x, train_idx, val_idx, test_idx, mean, std = prepare_data(x, y)

    print("\nTraining nodes:", len(train_idx))
    print("Validation nodes:", len(val_idx))
    print("Test nodes:", len(test_idx))

    model = GCN(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=5e-4
    )

    best_val_acc = -1.0
    best_state = None

    print("\nStarting GCN training...\n")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        optimizer.zero_grad()

        logits = model(x, edge_index)

        loss = F.cross_entropy(
            logits[train_idx],
            y[train_idx]
        )

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():

            logits = model(x, edge_index)

            train_acc = accuracy(logits, y, train_idx)
            val_acc = accuracy(logits, y, val_idx)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Accuracy: {train_acc:.4f} | "
                f"Validation Accuracy: {val_acc:.4f}"
            )

    # 
    # Test the best model

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():

        logits = model(x, edge_index)
        test_acc = accuracy(logits, y, test_idx)

    print("\nTraining completed!")
    print("-------------------------")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

      # Save the trained model


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "gcn_surface_classifier.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_classes": NUM_CLASSES,
            "feature_mean": mean,
            "feature_std": std
        },
        model_path
    )

    print("\nModel saved at:", model_path)


if __name__ == "__main__":
    main()
