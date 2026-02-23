
import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Graphbuilding')))
from cloud_dataset import CloudDataset
from rgcn_model import RGCN
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

# Configuration
BATCH_SIZE = 32
HIDDEN_CHANNELS = 64
EPOCHS = 50
LEARNING_RATE = 0.01

def train():
    # 1. Load Dataset
    dataset = CloudDataset(root=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../dataset_risk")))
    
    # Shuffle and Split
    dataset = dataset.shuffle()
    n = len(dataset)
    train_idx = int(n * 0.8)
    val_idx = int(n * 0.9)
    
    train_dataset = dataset[:train_idx]
    val_dataset = dataset[train_idx:val_idx]
    test_dataset = dataset[val_idx:]
    
    print(f"Dataset Size: {n}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Number of Node Types: {len(dataset.node_type_map)}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    # num_node_types is dynamic based on dataset vocabulary
    # num_classes = 4 (0, 1, 2, 3)
    # num_relations = 2 (dependency, permission)
    model = RGCN(num_node_types=len(dataset.node_type_map) + 1, 
                 hidden_channels=HIDDEN_CHANNELS, 
                 num_classes=4, 
                 num_relations=2)
    
    # Calculate Class Weights for Imbalance
    # Count labels in train_dataset
    y_train_all = []
    for data in train_dataset:
        y_train_all.extend(data.y.tolist())
    
    y_train_all = np.array(y_train_all)
    class_counts = np.bincount(y_train_all, minlength=4)
    total_samples = len(y_train_all)
    
    print(f"Class Counts: {class_counts}")
    
    # Inverse Class Frequency Weights
    # weight[c] = total / (num_classes * count[c])
    class_weights = total_samples / (4 * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print(f"Class Weights: {class_weights}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights) 
    
    print(f"Training on {device}...")
    
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index, data.edge_type)
            
            # Loss is calculated on all nodes
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # VALIDATE
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_type)
                pred = out.argmax(dim=1)
                
                correct += (pred == data.y).sum().item()
                total += data.num_nodes # Total nodes in batch
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
                
        val_acc = correct / total
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Epoch {epoch+1:03d}: Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'rgcn_model.pth')
            
    print("Training Complete.")
    
    # TEST
    print("Evaluating on Test Set...")
    model.load_state_dict(torch.load('rgcn_model.pth'))
    model.eval()
    
    correct = 0
    total = 0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_type)
            pred = out.argmax(dim=1)
            
            correct += (pred == data.y).sum().item()
            total += data.num_nodes
            
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(data.y.cpu().numpy())
            
    test_acc = correct / total
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

if __name__ == "__main__":
    train()
