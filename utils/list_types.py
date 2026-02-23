
import pickle
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../dataset_risk/processed/node_type_map.pkl"))
if os.path.exists(path):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)
        print(f"Total Types: {len(mapping)}")
        print("First 20 types:")
        for k, v in list(mapping.items())[:20]:
            print(f"  {k}: {v}")
else:
    print("Map not found")
