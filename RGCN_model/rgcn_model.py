
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCN(torch.nn.Module):
    def __init__(self, num_node_types, hidden_channels, num_classes, num_relations):
        super(RGCN, self).__init__()
        
        # Embedding layer for node types
        # Maps integer node types to continuous vectors
        self.node_embedding = torch.nn.Embedding(num_node_types, hidden_channels)
        
        # First RGCN Layer
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_bases=30)
        
        # Second RGCN Layer
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_bases=30)
        
        # Decoder (MLP) to predict Risk Score
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_type):
        # x is (num_nodes, ) indices of node types
        x = self.node_embedding(x)
        
        # Graph Convolution 1
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Graph Convolution 2
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Dense Layers
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        
        # Output is (num_nodes, num_classes) logits
        return x
