
import os
import pickle
import networkx as nx

pickle_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Processed_Graphs'))
files = [f for f in os.listdir(pickle_dir) if f.endswith('.gpickle')]
if files:
    path = os.path.join(pickle_dir, files[0])
    with open(path, 'rb') as f:
        G = pickle.load(f)
        print(f"Graph loaded from {path}")
        if len(G.nodes) > 0:
            print(f"Sample Node: {list(G.nodes(data=True))[0]}")
        else:
            print("Graph has no nodes")
else:
    print("No files found")
