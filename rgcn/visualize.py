import os
import networkx as nx
import matplotlib.pyplot as plt

GRAPH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Processed_Graphs"))
FILE_NAME = "700998331.gpickle"   # <-- your file
OUTPUT_FILE = "700998331graphrisk.png"         # <-- output image

def visualize_single_graph():
    path = os.path.join(GRAPH_DIR, FILE_NAME)

    if not os.path.exists(path):
        print("❌ File not found:", path)
        print("Files in folder:", os.listdir(GRAPH_DIR))
        return

    G = nx.read_gpickle(path)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    plt.figure(figsize=(24, 18))

    # Layout that works without extra libs
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=300, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=0.7, alpha=0.4)

    # Labels only if graph is small
    if G.number_of_nodes() <= 40:
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(FILE_NAME)
    plt.axis("off")

    plt.savefig(OUTPUT_FILE, dpi=400, bbox_inches="tight")
    plt.close()

    print("✅ Graph saved as:", OUTPUT_FILE)

if __name__ == "__main__":
    visualize_single_graph()

