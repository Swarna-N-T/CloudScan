
import torch
from torch_geometric.loader import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Graphbuilding')))
from cloud_dataset import CloudDataset
from rgcn_model import RGCN
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

def evaluate():
    # 1. Load Data
    print("Loading Dataset...")
    dataset = CloudDataset(root=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../dataset_risk")))
    
    # 2. Replicate Split (Best Effort)
    # Note: If the original training didn't save the split or seed, this might overlap with training data.
    # We set a seed here for reproducibility of the *evaluation*.
    torch.manual_seed(42) 
    dataset = dataset.shuffle()
    
    n = len(dataset)
    train_idx = int(n * 0.8)
    val_idx = int(n * 0.9)
    
    test_dataset = dataset[val_idx:]
    
    print(f"Total Graphs: {n}")
    print(f"Test Set Size: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Load Model
    HIDDEN_CHANNELS = 64
    model = RGCN(num_node_types=len(dataset.node_type_map) + 1, 
                 hidden_channels=HIDDEN_CHANNELS, 
                 num_classes=4, 
                 num_relations=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    try:
        model.load_state_dict(torch.load('rgcn_model.pth', map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_type)
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            
    # 4. Metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, digits=4)
    
    print("\n" + "="*40)
    print(f"Test Accuracy: {acc:.4%}")
    print("="*40)
    
    print("\nConfusion Matrix:")
    class_names = ["Safe", "Low", "Medium", "High"]
    
    # Calculate column widths
    col_width = 10
    header_row = "".join([f"{name:>{col_width}}" for name in class_names])
    print(f"{'':<{col_width}}{header_row}")
    
    for i, row in enumerate(cm):
        row_str = "".join([f"{val:>{col_width}}" for val in row])
        print(f"{class_names[i]:<{col_width}}{row_str}")
    
    print("\nClassification Report:")
    print(cr)
    
if __name__ == "__main__":
    evaluate()
