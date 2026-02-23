
import os
import pickle
import networkx as nx
from collections import Counter
import numpy as np

pickle_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Processed_Graphs'))
files = [f for f in os.listdir(pickle_dir) if f.endswith('.gpickle')]

total_graphs = 0
valid_graphs = 0
graphs_no_edges = 0
risk_counts = Counter()
edge_counts = []
node_counts = []

print(f"Scanning {len(files)} files...")

for i, f_name in enumerate(files):
    try:
        with open(os.path.join(pickle_dir, f_name), 'rb') as f:
            G = pickle.load(f)
            
            n = G.number_of_nodes()
            e = G.number_of_edges()
            
            if n == 0:
                continue
                
            total_graphs += 1
            node_counts.append(n)
            edge_counts.append(e)
            
            if e == 0:
                graphs_no_edges += 1
            else:
                valid_graphs += 1
                
            # Collect labels
            for _, data in G.nodes(data=True):
                risk = data.get('risk_score', 0)
                risk_counts[risk] += 1
                
    except Exception as e:
        print(f"Error reading {f_name}: {e}")

print("-" * 30)
print(f"Total Graphs Checked: {total_graphs}")
print(f"Graphs with Edges: {valid_graphs}")
print(f"Graphs with NO Edges: {graphs_no_edges} ({graphs_no_edges/total_graphs*100:.1f}%)")
print("-" * 30)
print(f"Avg Nodes: {np.mean(node_counts):.1f}")
print(f"Avg Edges: {np.mean(edge_counts):.1f}")
print("-" * 30)
print("Risk Score Distribution (Nodes):")
for k, v in sorted(risk_counts.items()):
    print(f"  Score {k}: {v} nodes")
