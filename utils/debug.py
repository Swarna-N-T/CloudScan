import os
import json
import hcl2

# --- CONFIGURATION ---
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../AWSonly_graph_Dataset"))
# ---------------------

def debug_repo(repo_path):
    print(f"\n--- Debugging: {os.path.basename(repo_path)} ---")
    
    # 1. READ CHECKOV REPORT
    report_path = os.path.join(repo_path, "checkov_report.json")
    checkov_ids = set()
    
    if not os.path.exists(report_path):
        print("❌ Checkov Report Missing!")
        return
    
    try:
        with open(report_path, 'r') as f:
            content = f.read().strip()
            if not content:
                print("❌ Checkov Report is EMPTY.")
                return
            data = json.loads(content)
            checks = data.get("results", {}).get("failed_checks", [])
            print(f"📄 Report contains {len(checks)} failed checks.")
            
            for c in checks[:3]: # Print first 3
                raw_id = c.get("resource", "")
                print(f"   Checkov sees: '{raw_id}'")
                checkov_ids.add(raw_id)
    except Exception as e:
        print(f"❌ Error reading report: {e}")
        return

    # 2. READ TERRAFORM FILES (OUR GRAPH IDs)
    print("🔍 Scanning Terraform files...")
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".tf"):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        data = hcl2.load(f)
                        for resource_entry in data.get('resource', []):
                            for res_type, res_instances in resource_entry.items():
                                for res_name, _ in res_instances.items():
                                    
                                    # Create our ID
                                    clean_type = res_type.replace("aws_aws_", "aws_")
                                    if not clean_type.startswith("aws_"): clean_type = f"aws_{clean_type}"
                                    
                                    graph_id = f"{clean_type}.{res_name}"
                                    
                                    # CHECK MATCH
                                    # We use strict and fuzzy checking to see what works
                                    match = "NO"
                                    if graph_id in checkov_ids:
                                        match = "EXACT"
                                    else:
                                        # Check for partials
                                        for cid in checkov_ids:
                                            if graph_id in cid: match = "PARTIAL (Graph in Checkov)"
                                            if cid in graph_id: match = "PARTIAL (Checkov in Graph)"
                                    
                                    print(f"   Graph sees:   '{graph_id}' -> Match? {match}")
                                    
                                    # Only print 3 nodes per repo to keep log short
                                    if len(checkov_ids) > 0:
                                        return 
                except: continue

def main():
    repos = os.listdir(SOURCE_DIR)
    # Scan the first 5 repos that actually have a report
    scanned = 0
    for repo in repos:
        path = os.path.join(SOURCE_DIR, repo)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "checkov_report.json")):
            debug_repo(path)
            scanned += 1
            if scanned >= 5: break

if __name__ == "__main__":
    main()
