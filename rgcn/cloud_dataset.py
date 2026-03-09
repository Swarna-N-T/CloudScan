
import os
import pickle
import torch
from torch_geometric.data import Data, InMemoryDataset
import networkx as nx
import numpy as np

class CloudDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CloudDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
        # Load mappings
        if os.path.exists(self.processed_paths[1]):
            with open(self.processed_paths[1], 'rb') as f:
                self.node_type_map = pickle.load(f)
        else:
            self.node_type_map = {}

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt', 'node_type_map.pkl']

    def download(self):
        pass

    def process(self):
        # Path to raw pickles
        raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Processed_Graphs'))
        files = [f for f in os.listdir(raw_dir) if f.endswith('.gpickle')]
        
        data_list = []
        node_types = set()
        
        # First pass: Collect all node types for vocabulary
        print("Building vocabulary...")
        for file in files:
            path = os.path.join(raw_dir, file)
            try:
                with open(path, 'rb') as f:
                    G = pickle.load(f)
                    for _, data in G.nodes(data=True):
                        t = data.get('type', 'unknown')
                        node_types.add(t)
            except: pass
            
        # Create mapping
        node_type_map = {t: i for i, t in enumerate(sorted(list(node_types)))}
        print(f"Found {len(node_type_map)} unique node types.")
        
        # Second pass: Create Data objects
        print("Processing graphs...")
        for file in files:
            path = os.path.join(raw_dir, file)
            try:
                with open(path, 'rb') as f:
                    G = pickle.load(f)
                    
                if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                    continue
                    
                # Node Features (x)
                # We use the index of the node type as the feature
                x_indices = []
                y_labels = []
                
                # Map node IDs to 0...N-1 for edge_index
                node_id_map = {n: i for i, n in enumerate(G.nodes())}
                
                for n in G.nodes():
                    data = G.nodes[n]
                    t = data.get('type', 'unknown')
                    x_indices.append(node_type_map.get(t, 0))
                    
                    # Target Label (Risk Score)
                    risk = data.get('risk_score', 0)
                    y_labels.append(risk)
                    
                x = torch.tensor(x_indices, dtype=torch.long)
                y = torch.tensor(y_labels, dtype=torch.long)
                
                # Edges
                edge_src = []
                edge_dst = []
                edge_types = []
                
                for u, v, data in G.edges(data=True):
                    if u in node_id_map and v in node_id_map:
                        edge_src.append(node_id_map[u])
                        edge_dst.append(node_id_map[v])
                        
                        # Edge Type: 0 for dependency, 1 for permission
                        etype_str = data.get('edge_type', 'dependency')
                        etype = 1 if etype_str == 'permission' else 0
                        edge_types.append(etype)
                        
                edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
                edge_type = torch.tensor(edge_types, dtype=torch.long)
                
                data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
                data.num_nodes = G.number_of_nodes()
                
                data_list.append(data)
                
            except Exception as e:
                print(f"Failed {file}: {e}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        with open(self.processed_paths[1], 'wb') as f:
            pickle.dump(node_type_map, f)

if __name__ == "__main__":
    # Test run
    dataset = CloudDataset(root=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../dataset_risk")))
    print(f"Dataset created with {len(dataset)} graphs.")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of node features: {dataset.num_features}")
