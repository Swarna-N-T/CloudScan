
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Graphbuilding')))
from cloud_dataset import CloudDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train_baseline():
    print("Loading Dataset...")
    dataset = CloudDataset(root=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../dataset_risk")))
    
    # Flatten Data for standard ML
    X_all = []
    y_all = []
    
    # We use the same split logic implicitly by index, but let's just do a random split on nodes
    # actually, to be fair, we should split by GRAPH, not by NODE, to avoid data leakage
    # The RGCN split was by graph index.
    
    # Shuffle same way as RGCN (we hope the seed is consistent or we just take the same ratio)
    # dataset = dataset.shuffle() # Logic in train_rgcn was shuffle then split
    # To be perfectly fair we should have saved the indices. 
    # But for a quick baseline, a random split of graphs is fine.
    
    dataset = dataset.shuffle()
    n = len(dataset)
    train_idx = int(n * 0.8)
    test_idx = int(n * 0.9) # Validation + Test in RGCN was 20%, here let's just verify on 20%
    
    train_data = dataset[:train_idx]
    test_data = dataset[train_idx:]
    
    print(f"Train Graphs: {len(train_data)}")
    print(f"Test Graphs: {len(test_data)}")

    # Extract Features (Node Type ID) and Labels
    def extract_features(data_list):
        X = []
        y = []
        for data in data_list:
            # x is a tensor of shape [num_nodes] containing node type indices
            # We reshape to [num_nodes, 1] for sklearn
            features = data.x.numpy().reshape(-1, 1)
            labels = data.y.numpy()
            
            X.append(features)
            y.append(labels)
        
        X_stacked = np.vstack(X)
        y_stacked = np.concatenate(y)
        return X_stacked, y_stacked

    print("Extracting features...")
    X_train_raw, y_train = extract_features(train_data)
    X_test_raw, y_test = extract_features(test_data)
    
    # One-Hot Encode
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=True) # Use sparse to save memory
    X_train = enc.fit_transform(X_train_raw)
    X_test = enc.transform(X_test_raw)

    
    print(f"Train Nodes: {X_train.shape[0]}")
    print(f"Test Nodes: {X_test.shape[0]}")
    
    # Initialize Random Forest
    # n_estimators=100, standard
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    
    print("Training Random Forest...")
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print("-" * 30)
    print(f"Baseline (Random Forest on Node Types Only)")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 30)

if __name__ == "__main__":
    train_baseline()
