
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Graphbuilding')))
import torch
import random
from cloud_dataset import CloudDataset
from rgcn_model import RGCN
import numpy as np

def verify():
    # 1. Load Data
    dataset = CloudDataset(root=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../dataset_risk")))
    
    # 2. Load Model
    # Note: Architecture params must match training
    HIDDEN_CHANNELS = 64
    model = RGCN(num_node_types=len(dataset.node_type_map) + 1, 
                 hidden_channels=HIDDEN_CHANNELS, 
                 num_classes=4, 
                 num_relations=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    try:
        model.load_state_dict(torch.load('rgcn_model.pth', weights_only=True))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    # 3. Pick random graphs to verify
    print("-" * 40)
    print("Running Inference on Random Graphs")
    print("-" * 40)
    
    indices = random.sample(range(len(dataset)), 5)
    
    for idx in indices:
        data = dataset[idx].to(device)
        
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_type)
            pred = out.argmax(dim=1)
            
        correct = (pred == data.y).sum().item()
        acc = correct / data.num_nodes
        
        print(f"Graph Index: {idx}")
        print(f"Nodes: {data.num_nodes}")
        print(f"Accuracy: {acc:.2%}")
        
        # Show a few sample nodes
        print("Sample Predictions:")
        for i in range(min(5, data.num_nodes)):
            print(f"  Node {i}: Pred={pred[i].item()}, True={data.y[i].item()}")
        print("-" * 20)

if __name__ == "__main__":
    verify()
