import os
import networkx as nx
import pickle
import matplotlib.pyplot as plt

# Configuration
INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "Processed_Graphs"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "Visualizations"))

def visualize_graph(g_path, out_path, repo_name):
    try:
        with open(g_path, 'rb') as f:
            G = pickle.load(f)
            
        if G.number_of_nodes() == 0:
            print(f"Skipping {repo_name} - Graph is empty.")
            return

        plt.figure(figsize=(16, 12))
        
        # Determine node colors based on risk_score if available
        # Default to safe (blue) if no risk score is found
        node_colors = []
        for n, data in G.nodes(data=True):
            risk = data.get('risk_score', 0)
            if risk == 3: # High Risk
                node_colors.append('#ff4d4d') # Red
            elif risk == 2: # Medium Risk
                node_colors.append('#ffb84d') # Orange
            elif risk == 1: # Low Risk
                node_colors.append('#ffff66') # Yellow
            else: # Safe
                node_colors.append('#66b3ff') # Blue

        # Draw the graph 
        # spring_layout works well for small to medium graphs
        pos = nx.spring_layout(G, k=0.15, iterations=20) 
        
        # Edge styling based on type
        edges = G.edges(data=True)
        dep_edges = [(u, v) for u, v, d in edges if d.get('edge_type') == 'dependency']
        perm_edges = [(u, v) for u, v, d in edges if d.get('edge_type') == 'permission']

        # Draw Nodes
        nx.draw_networkx_nodes(G, pos, node_size=150, node_color=node_colors, alpha=0.9, edgecolors='black')
        
        # Draw Edges
        nx.draw_networkx_edges(G, pos, edgelist=dep_edges, edge_color='gray', width=1.0, alpha=0.5, arrows=True)
        nx.draw_networkx_edges(G, pos, edgelist=perm_edges, edge_color='red', width=1.5, alpha=0.7, arrows=True, style='dashed')
        
        # Draw Labels (Cleaned up Types)
        labels = {}
        for n, data in G.nodes(data=True):
            clean_type = data.get('type', 'unknown').replace('aws_', '')
            labels[n] = clean_type
            
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black', font_family='sans-serif')

        # Title and styling
        plt.title(f"Infrastructure Graph: {repo_name}\nNodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}", size=15)
        plt.axis('off')
        
        # Save exact graph to PNG
        plt.tight_layout()
        plt.savefig(out_path, format='png', dpi=300)
        plt.close()
        
        print(f"Saved visualization: {out_path}")
        
    except Exception as e:
        print(f"Failed to visualize {repo_name}: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    graphs = [f for f in os.listdir(INPUT_DIR) if f.endswith('.gpickle')]
    
    print(f"Generating visualizations for {len(graphs)} graphs...")
    for file in graphs:
        repo_name = file.replace('.gpickle', '')
        in_path = os.path.join(INPUT_DIR, file)
        out_path = os.path.join(OUTPUT_DIR, f"{repo_name}.png")
        
        visualize_graph(in_path, out_path, repo_name)
        
    print("Done generating graph visualizations.")

if __name__ == "__main__":
    main()
