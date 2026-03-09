"""
CloudScan Inference Pipeline
Runs the RGCN model on Terraform infrastructure graphs, flags risky resources,
and calls the Gemini LLM for remediation advice.
"""

import os
import sys
import json
import argparse
import pickle
import torch
from torch_geometric.loader import DataLoader

# Local imports (cloud_dataset.py and rgcn_model.py are in the same directory)
from cloud_dataset import CloudDataset
from rgcn_model import RGCN
from llm_remediation import generate_remediation, RISK_LABELS

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gpickle_graphs(processed_graphs_dir):
    """
    Load all .gpickle files from the Processed_Graphs directory.
    Returns an ordered list of (filename, NetworkX DiGraph) tuples,
    sorted in the same order the CloudDataset processes them.
    """
    files = sorted([f for f in os.listdir(processed_graphs_dir) if f.endswith('.gpickle')])
    graphs = []
    for file in files:
        path = os.path.join(processed_graphs_dir, file)
        try:
            with open(path, 'rb') as f:
                G = pickle.load(f)
            if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                graphs.append((file, G))
        except Exception as e:
            print(f"  ⚠ Could not load {file}: {e}")
    return graphs


def extract_flagged_resources(G, predictions, threshold=1):
    """
    Map RGCN per-node predictions back to the original NetworkX graph nodes.
    Returns a list of dicts for every node whose predicted risk >= threshold.
    """
    nodes = list(G.nodes())
    flagged = []

    for idx, node_id in enumerate(nodes):
        pred_risk = int(predictions[idx])
        if pred_risk >= threshold:
            node_data = G.nodes[node_id]
            flagged.append({
                "node_id": node_id,
                "resource_type": node_data.get("type", "unknown"),
                "predicted_risk": pred_risk,
                "config": node_data.get("config", {}),
            })

    return flagged


def read_terraform_source(sample_data_dir, repo_name):
    """
    Read and concatenate all .tf files from the sample_data repo directory
    to provide full context to the LLM.
    """
    repo_path = os.path.join(sample_data_dir, repo_name)
    if not os.path.isdir(repo_path):
        return None

    tf_contents = []
    for root, _, files in os.walk(repo_path):
        for file in sorted(files):
            if file.endswith(".tf"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        tf_contents.append(f"# ── {file} ──\n{f.read()}")
                except:
                    pass

    return "\n\n".join(tf_contents) if tf_contents else None


# ── Main Inference ────────────────────────────────────────────────────────────

def run_inference(pth_file_path, dataset_path, enable_remediation=True, risk_threshold=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model Weights (to infer architecture size)
    print(f"Loading model weights from: {pth_file_path}")
    state_dict = torch.load(pth_file_path, map_location=device, weights_only=True)
    num_node_types = state_dict['node_embedding.weight'].shape[0]

    # 2. Load Dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = CloudDataset(root=dataset_path)

    # 3. Initialize Model
    HIDDEN_CHANNELS = 64
    NUM_CLASSES = 4
    NUM_RELATIONS = 2

    model = RGCN(
        num_node_types=num_node_types,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=NUM_CLASSES,
        num_relations=NUM_RELATIONS
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!\n")

    # 4. Load original .gpickle graphs to recover Terraform context
    processed_graphs_dir = os.path.join(
        os.path.dirname(os.path.abspath(dataset_path)),
        os.path.basename(dataset_path),
        "Processed_Graphs"
    )
    # Try the inference-local Processed_Graphs first, then dataset root
    if not os.path.isdir(processed_graphs_dir):
        processed_graphs_dir = os.path.join(dataset_path, "Processed_Graphs")
    if not os.path.isdir(processed_graphs_dir):
        processed_graphs_dir = os.path.join(os.path.dirname(__file__), "Processed_Graphs")

    gpickle_graphs = load_gpickle_graphs(processed_graphs_dir)
    print(f"Loaded {len(gpickle_graphs)} original graphs from: {processed_graphs_dir}")

    # Sample data directory (for reading raw .tf source)
    sample_data_dir = os.path.join(os.path.dirname(__file__), "sample_data")

    # 5. Run Inference
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_flagged = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_type)
            predictions = out.argmax(dim=1).cpu().numpy()

            # Count risk levels
            risk_counts = {label: 0 for label in RISK_LABELS.values()}
            for p in predictions:
                risk_counts[RISK_LABELS.get(int(p), "Unknown")] += 1

            print(f"{'='*60}")
            print(f"  Graph {batch_idx + 1}")
            print(f"  Total Nodes: {data.num_nodes}")
            print(f"  Risk Distribution: {json.dumps(risk_counts)}")

            # Map predictions back to Terraform resources
            if batch_idx < len(gpickle_graphs):
                gpickle_name, G = gpickle_graphs[batch_idx]
                repo_name = gpickle_name.replace(".gpickle", "")
                print(f"  Source: {repo_name}")

                flagged = extract_flagged_resources(G, predictions, threshold=risk_threshold)

                if flagged:
                    print(f"\n  🚩 Flagged Resources ({len(flagged)}):")
                    for res in flagged:
                        risk_label = RISK_LABELS.get(res["predicted_risk"], "?")
                        print(f"     [{risk_label}] {res['resource_type']}  →  {res['node_id']}")
                    all_flagged.append({
                        "graph_idx": batch_idx,
                        "repo_name": repo_name,
                        "flagged": flagged,
                    })
                else:
                    print(f"\n  ✅ No resources flagged above threshold.")
            else:
                print(f"\n  ⚠ No matching .gpickle graph for context mapping.")

            print()

    # 6. LLM Remediation
    if enable_remediation and all_flagged:
        print(f"\n{'='*60}")
        print(f"  🤖 LLM REMEDIATION")
        print(f"{'='*60}\n")

        for entry in all_flagged:
            repo_name = entry["repo_name"]
            flagged = entry["flagged"]

            print(f"── Remediation for: {repo_name} ({len(flagged)} flagged resources) ──\n")

            # Try to read the original .tf source for extra LLM context
            tf_source = read_terraform_source(sample_data_dir, repo_name)

            try:
                remediation = generate_remediation(flagged, terraform_source=tf_source)
                print(remediation)
            except Exception as e:
                print(f"  ⚠ LLM call failed: {e}")

            print(f"\n{'─'*60}\n")

    elif not all_flagged:
        print("\n✅ No resources were flagged across all graphs. No remediation needed.")

    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RGCN inference + LLM remediation on Terraform infrastructure graphs."
    )

    default_model = os.path.join(os.path.dirname(__file__), "rgcn_model.pth")
    default_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset_risk"))

    parser.add_argument("--model",     type=str, default=default_model,    help="Path to trained .pth file")
    parser.add_argument("--dataset",   type=str, default=default_dataset,  help="Path to dataset root")
    parser.add_argument("--threshold", type=int, default=1, choices=[1,2,3], help="Minimum risk level to flag (1=Low, 2=Medium, 3=High)")
    parser.add_argument("--no-llm",    action="store_true",                help="Skip LLM remediation (only show flags)")

    args = parser.parse_args()

    run_inference(
        args.model,
        args.dataset,
        enable_remediation=not args.no_llm,
        risk_threshold=args.threshold,
    )
