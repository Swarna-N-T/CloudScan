"""
Dynamic Inference Module
Builds a graph from raw Terraform files and runs RGCN inference on-the-fly,
without requiring the full CloudDataset pipeline.
"""

import os
import pickle
import torch
import numpy as np
from torch_geometric.data import Data

from build_sample_graphs import build_graph
from rgcn_model import RGCN

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "rgcn_model.pth")
DEFAULT_NODE_TYPE_MAP_PATH = os.path.join(BASE_DIR, "processed", "node_type_map.pkl")

# ── Risk labels (same as llm_remediation.py) ──────────────────────────────────
RISK_LABELS = {0: "Safe", 1: "Low", 2: "Medium", 3: "High/Critical"}


def load_node_type_map(path=None):
    """Load the node-type-to-index mapping used during training."""
    path = path or DEFAULT_NODE_TYPE_MAP_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"node_type_map.pkl not found at {path}. "
            "Run the training pipeline first to generate this file."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def nx_graph_to_pyg_data(G, node_type_map):
    """
    Convert a NetworkX DiGraph (from build_graph) into a PyG Data object
    using the training-time node_type_map for consistent feature indices.
    """
    if G.number_of_nodes() == 0:
        return None

    # Map node IDs to 0…N-1
    node_list = list(G.nodes())
    node_id_map = {n: i for i, n in enumerate(node_list)}

    # Node features: node type index from the training vocabulary
    x_indices = []
    for n in node_list:
        node_data = G.nodes[n]
        t = node_data.get("type", "unknown")
        # Fall back to 0 for unseen types, the model will still work
        x_indices.append(node_type_map.get(t, 0))

    x = torch.tensor(x_indices, dtype=torch.long)

    # Edges
    edge_src, edge_dst, edge_types = [], [], []
    for u, v, data in G.edges(data=True):
        if u in node_id_map and v in node_id_map:
            edge_src.append(node_id_map[u])
            edge_dst.append(node_id_map[v])
            etype_str = data.get("edge_type", "dependency")
            edge_types.append(1 if etype_str == "permission" else 0)

    # Handle graphs with no edges — add self-loop so the model doesn't crash
    if len(edge_src) == 0:
        edge_src = [0]
        edge_dst = [0]
        edge_types = [0]

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.num_nodes = len(node_list)
    return data


def run_dynamic_inference(
    tf_folder_path,
    model_path=None,
    node_type_map_path=None,
    risk_threshold=1,
    enable_remediation=True,
):
    """
    End-to-end dynamic inference on a folder of Terraform files.

    Parameters
    ----------
    tf_folder_path : str
        Path to a folder containing .tf files.
    model_path : str, optional
        Path to the trained RGCN .pth file.
    node_type_map_path : str, optional
        Path to node_type_map.pkl from training.
    risk_threshold : int
        Minimum risk level to flag (1=Low, 2=Med, 3=High).
    enable_remediation : bool
        Whether to call the LLM for remediation advice.

    Returns
    -------
    dict with keys:
        - graph_stats: {nodes, edges, node_types}
        - all_predictions: [{node_id, resource_type, predicted_risk, risk_label}]
        - flagged_resources: [{node_id, resource_type, predicted_risk, risk_label, config}]
        - remediation: str or None
        - terraform_source: str (concatenated .tf content)
    """
    model_path = model_path or DEFAULT_MODEL_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Build NetworkX graph from raw .tf files
    G = build_graph(tf_folder_path)
    if G.number_of_nodes() == 0:
        return {
            "graph_stats": {"nodes": 0, "edges": 0, "node_types": []},
            "all_predictions": [],
            "flagged_resources": [],
            "remediation": None,
            "terraform_source": None,
        }

    # 2. Load node type map from training
    node_type_map = load_node_type_map(node_type_map_path)

    # 3. Convert to PyG Data
    pyg_data = nx_graph_to_pyg_data(G, node_type_map)
    pyg_data = pyg_data.to(device)

    # 4. Load model
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    num_node_types = state_dict["node_embedding.weight"].shape[0]

    model = RGCN(
        num_node_types=num_node_types,
        hidden_channels=64,
        num_classes=4,
        num_relations=2,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Run inference
    with torch.no_grad():
        out = model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_type)
        predictions = out.argmax(dim=1).cpu().numpy()

    # 6. Map predictions back to resources
    node_list = list(G.nodes())
    unique_types = set()
    all_predictions = []
    flagged = []

    for idx, node_id in enumerate(node_list):
        pred_risk = int(predictions[idx])
        node_data = G.nodes[node_id]
        res_type = node_data.get("type", "unknown")
        unique_types.add(res_type)

        entry = {
            "node_id": node_id,
            "resource_type": res_type,
            "predicted_risk": pred_risk,
            "risk_label": RISK_LABELS.get(pred_risk, "Unknown"),
        }
        all_predictions.append(entry)

        if pred_risk >= risk_threshold:
            entry_flagged = {
                **entry,
                "config": node_data.get("config", {}),
            }
            flagged.append(entry_flagged)

    # 7. Read Terraform source for LLM context
    tf_source = _read_tf_source(tf_folder_path)

    # 8. LLM remediation (optional)
    remediation_text = None
    if enable_remediation and flagged:
        try:
            from llm_remediation import generate_remediation
            remediation_text = generate_remediation(flagged, terraform_source=tf_source)
        except Exception as e:
            remediation_text = f"LLM remediation failed: {e}"

    return {
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "node_types": sorted(list(unique_types)),
        },
        "all_predictions": all_predictions,
        "flagged_resources": flagged,
        "remediation": remediation_text,
        "terraform_source": tf_source,
    }


def _read_tf_source(folder_path):
    """Read and concatenate all .tf files for LLM context."""
    tf_parts = []
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if file.endswith(".tf"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        tf_parts.append(f"# ── {file} ──\n{f.read()}")
                except:
                    pass
    return "\n\n".join(tf_parts) if tf_parts else None


# ── CLI for standalone testing ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python dynamic_inference.py <path_to_terraform_folder>")
        sys.exit(1)

    folder = sys.argv[1]
    print(f"Running dynamic inference on: {folder}")
    result = run_dynamic_inference(folder, enable_remediation=False)

    print(f"\nGraph Stats: {json.dumps(result['graph_stats'], indent=2)}")
    print(f"\nFlagged Resources ({len(result['flagged_resources'])}):")
    for r in result["flagged_resources"]:
        print(f"  [{r['risk_label']}] {r['resource_type']} → {r['node_id']}")
