
import os
import pickle
import networkx as nx

# Path to the directory containing pickle files
pickle_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Processed_Graphs'))

if not os.path.exists(pickle_dir):
    print(f"Directory not found: {pickle_dir}")
    exit()

# Get the first pickle file
pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.gpickle') or f.endswith('.pickle') or f.endswith('.pkl')]

if not pickle_files:
    print("No pickle files found.")
else:
    first_file = os.path.join(pickle_dir, pickle_files[0])
    print(f"Loading {first_file}...")
    
    try:
        with open(first_file, 'rb') as f:
            data = pickle.load(f)
            print(f"Type of data: {type(data)}")
            
            if isinstance(data, nx.Graph):
                print(f"Number of nodes: {data.number_of_nodes()}")
                print(f"Number of edges: {data.number_of_edges()}")
                if data.nodes():
                    print(f"Node attributes (first node): {list(data.nodes(data=True))[0]}")
                if data.edges():
                    print(f"Edge attributes (first edge): {list(data.edges(data=True))[0]}")
            else:
                print(f"Data type: {type(data)}")
    except Exception as e:
        print(f"Error loading pickle: {e}")
