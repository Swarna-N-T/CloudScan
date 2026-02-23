import os
import json
import hcl2
import networkx as nx
import pickle
import re

# --- CONFIGURATION ---
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../AWSonly_graph_Dataset"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Processed_Graphs"))
MAX_REPOS = 1500
# ---------------------

# Keywords for heuristic severity scoring if Checkov doesn't provide it
HIGH_RISK_KEYWORDS = ['0.0.0.0', 'public', 'unencrypted', 'clear text', 'hardcoded', 'admin', 'root', 'all ports']
MEDIUM_RISK_KEYWORDS = ['logging', 'backup', 'versioning', 'rotation', 'http']
LOW_RISK_KEYWORDS = ['tag', 'description', 'metadata']

STOP_WORDS = {
    'this', 'main', 'default', 'example', 'test', 'prod', 'stage', 'dev',
    'id', 'arn', 'name', 'type', 'aws', 'resource', 'module', 'data', 'var',
    'policy', 'role', 'bucket', 'cluster', 'group', 'rule', 'log', 'vpc'
}

def get_risk_score(check_data):
    """
    Returns a score: 3 (High), 2 (Medium), 1 (Low).
    Defaults to 2 (Medium) if unknown.
    """
    # 1. Try to read explicit severity if available (some versions have it)
    severity = check_data.get('severity')
    if severity:
        sev_str = str(severity).upper()
        if 'CRITICAL' in sev_str or 'HIGH' in sev_str: return 3
        if 'MEDIUM' in sev_str: return 2
        if 'LOW' in sev_str: return 1
    
    # 2. Heuristic based on Check Name/ID
    name = check_data.get('check_name', '').lower()
    check_id = check_data.get('check_id', '').upper()
    
    # Specific ID Overrides (Examples)
    if 'CKV_AWS_24' in check_id: return 3 # SSH to 0.0.0.0
    
    for kw in HIGH_RISK_KEYWORDS:
        if kw in name: return 3
    for kw in MEDIUM_RISK_KEYWORDS:
        if kw in name: return 2
    for kw in LOW_RISK_KEYWORDS:
        if kw in name: return 1
        
    return 2 # Default to Medium if it failed a check

def load_checkov_risks(repo_path):
    """
    Reads checkov_report.json and returns a dict: { resource_id -> max_risk_score }
    """
    report_path = os.path.join(repo_path, "checkov_report.json")
    resource_risks = {}
    
    if not os.path.exists(report_path):
        return resource_risks

    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
            
            # Handle different JSON structures (list of results vs dict)
            results = data.get("results", {})
            failed_checks = []
            if isinstance(results, dict):
                failed_checks = results.get("failed_checks", [])
            elif isinstance(results, list):
                # meaningful structure might be different, but assuming dict based on sample
                pass
            
            for check in failed_checks:
                resource_id = check.get("resource", "")
                risk = get_risk_score(check)
                
                # Keep the highest risk found for this resource
                if resource_id not in resource_risks:
                    resource_risks[resource_id] = risk
                else:
                    resource_risks[resource_id] = max(resource_risks[resource_id], risk)
                    
    except Exception:
        pass 
    return resource_risks

def extract_unique_tokens(text):
    tokens = set(re.split(r'[_\.\-]+', str(text).lower()))
    unique_tokens = {t for t in tokens if t not in STOP_WORDS and len(t) > 3 and not t.isdigit()}
    return unique_tokens

def build_graph_for_repo(repo_path, repo_name):
    G = nx.DiGraph()
    resource_risks = load_checkov_risks(repo_path)
    node_registry = {} 
    
    # 1. PARSE & REGISTER NODES
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".tf"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path) 
                
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw_config = f.read().lower()
                        f.seek(0)
                        data = hcl2.load(f)
                        
                        for resource_entry in data.get('resource', []):
                            for res_type, res_instances in resource_entry.items():
                                for res_name, res_config in res_instances.items():
                                    
                                    # Normalize Type
                                    clean_type = res_type
                                    if not clean_type.startswith("aws_"): clean_type = f"aws_{clean_type}"
                                    
                                    # Construct Checkov-style ID for lookup
                                    # Checkov often uses: "aws_type.name"
                                    checkov_id = f"{clean_type}.{res_name}"
                                    
                                    # Determine Risk
                                    # Default 0 (Safe)
                                    risk_score = resource_risks.get(checkov_id, 0)
                                    
                                    # Node ID
                                    node_id = f"{rel_path}::{checkov_id}"
                                    
                                    G.add_node(node_id, 
                                               type=clean_type, 
                                               config=res_config, 
                                               risk_score=risk_score) # Changed from 'label' to 'risk_score'
                                    
                                    name_tokens = extract_unique_tokens(res_name)
                                    node_registry[node_id] = {
                                        "simple_id": checkov_id,
                                        "tokens": name_tokens,
                                        "raw_config": raw_config,
                                        "type": clean_type
                                    }
                except: continue

    # 2. LINKING (Semantic Logic)
    if G.number_of_nodes() > 0:
        nodes = list(G.nodes())
        for source_id in nodes:
            src_data = node_registry[source_id]
            src_config = src_data["raw_config"]
            sec_keywords = ['iam', 'policy', 'role', 'user', 'security_group', 'sg_', 'acl']
            is_src_sec = any(k in src_data["type"] for k in sec_keywords)

            for target_id in nodes:
                if source_id == target_id: continue
                tgt_data = node_registry[target_id]
                target_simple_id = tgt_data["simple_id"]
                
                match_found = False
                # Direct Reference
                if target_simple_id in src_config: match_found = True
                # Keyword overlapping (fuzzy matching)
                if not match_found and src_data["tokens"].intersection(tgt_data["tokens"]): match_found = True

                if match_found:
                    is_tgt_sec = any(k in tgt_data["type"] for k in sec_keywords)
                    e_type = 'permission' if (is_src_sec or is_tgt_sec) else 'dependency'
                    G.add_edge(source_id, target_id, edge_type=e_type)

    return G

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: {SOURCE_DIR} missing.")
        return
        
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    repos = os.listdir(SOURCE_DIR)
    
    # Sort or Shuffle if needed, or just take first N
    # repos.sort() 
    
    print(f"Building Graphs with Risk Scores for up to {MAX_REPOS} repos...")
    
    count = 0
    
    for i, repo in enumerate(repos):
        if count >= MAX_REPOS:
            break
            
        repo_path = os.path.join(SOURCE_DIR, repo)
        if os.path.isdir(repo_path):
            try:
                G = build_graph_for_repo(repo_path, repo)
                
                # Only save if graph is valid and has contents
                if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                    output_file = os.path.join(OUTPUT_DIR, f"{repo}.gpickle")
                    with open(output_file, 'wb') as f:
                        pickle.dump(G, f)
                    count += 1
                    
                    if count % 100 == 0:
                         print(f"Saved {count} graphs so far...")
            except Exception as e:
                print(f"Failed {repo}: {e}")
                
    print("-" * 30)
    print(f"DONE. Generated {count} graphs in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
