
import torch
import pickle
import os
from rgcn_model import RGCN
from torch_geometric.data import Data

def verify_propagation():
    # 1. Load Mappings
    try:
        map_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../dataset_risk/processed/node_type_map.pkl"))
        with open(map_path, 'rb') as f:
            node_map = pickle.load(f)
    except:
        print("Mappings not found. Run training first.")
        return

    # Identify IDs for common resources
    # We'll use 'aws_s3_bucket' (assume distinct type) and 'aws_iam_role'
    s3_idx = node_map.get('aws_s3_bucket', 0)
    iam_idx = node_map.get('aws_iam_role', 1) 
    
    print(f"S3 Bucket Index: {s3_idx}")
    print(f"IAM Role Index: {iam_idx}")

    # 2. Load Model
    HIDDEN_CHANNELS = 64
    model = RGCN(num_node_types=len(node_map) + 1, 
                 hidden_channels=HIDDEN_CHANNELS, 
                 num_classes=4, 
                 num_relations=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load('rgcn_model.pth', weights_only=True))
    model.eval()
    

    # Search for a "Safe" node type
    print("Searching for a naturally 'Safe' node type...")
    safe_idx = -1
    for i in range(len(node_map)):
        x_test = torch.tensor([i], dtype=torch.long).to(device)
        edge_index_test = torch.tensor([[], []], dtype=torch.long).to(device)
        edge_type_test = torch.tensor([], dtype=torch.long).to(device)
        
        with torch.no_grad():
            out = model(x_test, edge_index_test, edge_type_test)
            pred = out.argmax().item()
            
        if pred == 0:
            safe_idx = i
            print(f"Found Safe Node Type Index: {i} (Class 0)")
            break
            
    if safe_idx == -1:
        print("Could not find any Safe node type in isolation. Model might be biased.")
        return

    # Use the previously identified S3 bucket as the "Risky Neighbor" (Class 3)
    risky_idx = s3_idx
    
    # 3. Scenario A: Isolated Safe Node
    x_iso = torch.tensor([safe_idx], dtype=torch.long).to(device)
    edge_index_iso = torch.tensor([[], []], dtype=torch.long).to(device)
    edge_type_iso = torch.tensor([], dtype=torch.long).to(device)
    
    with torch.no_grad():
        out_iso = model(x_iso, edge_index_iso, edge_type_iso)
        probs_iso = torch.softmax(out_iso, dim=1)
        pred_iso = out_iso.argmax().item()
        
    print("-" * 30)
    print(f"Scenario A: Isolated Node {safe_idx}")
    print(f"Predicted Class: {pred_iso}")
    print(f"Probabilities: {probs_iso.cpu().numpy()}")
    
    # 4. Scenario B: Safe Node connected to Risky Neighbor
    # Node 0: Safe Node (Target)
    # Node 1: Risky Node (Neighbor)
    x_conn = torch.tensor([safe_idx, risky_idx], dtype=torch.long).to(device)
    
    # Edge: Risky -> Safe (Permission/Dependency)
    # We test both edge directions and types to be sure
    # Let's try Edge Type 1 (Permission) from Risky to Safe
    edge_index_conn = torch.tensor([[1], [0]], dtype=torch.long).to(device) # 1 -> 0
    edge_type_conn = torch.tensor([1], dtype=torch.long).to(device) 
    
    with torch.no_grad():
        out_conn = model(x_conn, edge_index_conn, edge_type_conn)
        probs_conn = torch.softmax(out_conn, dim=1)
        pred_conn = out_conn[0].argmax().item()
        
    print("-" * 30)
    print(f"Scenario B: Node {safe_idx} connected to Risky Node {risky_idx}")
    print(f"Target Node Prediction: {pred_conn}")
    print(f"Target Probabilities: {probs_conn[0].cpu().numpy()}")
    
    # Compare Risk (Sum of probs for classes 1, 2, 3)
    risk_iso = 1.0 - probs_iso[0, 0].item()
    risk_conn = 1.0 - probs_conn[0, 0].item()
    
    print("-" * 30)
    print(f"Risk Probability Change: {risk_iso:.4f} -> {risk_conn:.4f}")
    
    if risk_conn > risk_iso:
        print("✅ SUCCESS: Risk increased due to connection!")
        diff = risk_conn - risk_iso
        print(f"Magnitude of increase: {diff:.4f}")
    else:
        print("⚠️ WARNING: Risk did not increase.")


if __name__ == "__main__":
    verify_propagation()
