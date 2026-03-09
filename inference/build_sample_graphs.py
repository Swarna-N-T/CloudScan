import os
import json
import hcl2
import networkx as nx
import pickle
import re

# Configuration for evaluation
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_data"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "Processed_Graphs"))

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
    severity = check_data.get('severity')
    if severity:
        sev_str = str(severity).upper()
        if 'CRITICAL' in sev_str or 'HIGH' in sev_str: return 3
        if 'MEDIUM' in sev_str: return 2
        if 'LOW' in sev_str: return 1
    
    name = check_data.get('check_name', '').lower()
    check_id = check_data.get('check_id', '').upper()
    if 'CKV_AWS_24' in check_id: return 3 
    
    for kw in HIGH_RISK_KEYWORDS:
        if kw in name: return 3
    for kw in MEDIUM_RISK_KEYWORDS:
        if kw in name: return 2
    for kw in LOW_RISK_KEYWORDS:
        if kw in name: return 1
        
    return 2 

def load_checkov_risks(repo_path):
    report_path = os.path.join(repo_path, "checkov_report.json")
    resource_risks = {}
    if not os.path.exists(report_path):
        return resource_risks
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
            results = data.get("results", {})
            failed_checks = []
            if isinstance(results, dict):
                failed_checks = results.get("failed_checks", [])
            for check in failed_checks:
                resource_id = check.get("resource", "")
                risk = get_risk_score(check)
                if resource_id not in resource_risks:
                    resource_risks[resource_id] = risk
                else:
                    resource_risks[resource_id] = max(resource_risks[resource_id], risk)
    except: pass 
    return resource_risks

def extract_unique_tokens(text):
    tokens = set(re.split(r'[_\.\-]+', str(text).lower()))
    return {t for t in tokens if t not in STOP_WORDS and len(t) > 3 and not t.isdigit()}

def build_graph(repo_path):
    G = nx.DiGraph()
    resource_risks = load_checkov_risks(repo_path)
    node_registry = {} 
    
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
                                    clean_type = res_type
                                    if not clean_type.startswith("aws_"): clean_type = f"aws_{clean_type}"
                                    checkov_id = f"{clean_type}.{res_name}"
                                    risk_score = resource_risks.get(checkov_id, 0)
                                    node_id = f"{rel_path}::{checkov_id}"
                                    
                                    G.add_node(node_id, 
                                               type=clean_type, 
                                               config=res_config, 
                                               risk_score=risk_score)
                                    
                                    name_tokens = extract_unique_tokens(res_name)
                                    node_registry[node_id] = {
                                        "simple_id": checkov_id,
                                        "tokens": name_tokens,
                                        "raw_config": raw_config,
                                        "type": clean_type
                                    }
                except: continue

    if G.number_of_nodes() > 0:
        nodes = list(G.nodes())
        sec_keywords = ['iam', 'policy', 'role', 'user', 'security_group', 'sg_', 'acl']
        for source_id in nodes:
            src_data = node_registry[source_id]
            src_config = src_data["raw_config"]
            is_src_sec = any(k in src_data["type"] for k in sec_keywords)

            for target_id in nodes:
                if source_id == target_id: continue
                tgt_data = node_registry[target_id]
                target_simple_id = tgt_data["simple_id"]
                
                match_found = False
                if target_simple_id in src_config: match_found = True
                if not match_found and src_data["tokens"].intersection(tgt_data["tokens"]): match_found = True

                if match_found:
                    is_tgt_sec = any(k in tgt_data["type"] for k in sec_keywords)
                    e_type = 'permission' if (is_src_sec or is_tgt_sec) else 'dependency'
                    G.add_edge(source_id, target_id, edge_type=e_type)

    return G

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    repos = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    print(f"Found {len(repos)} sample projects to build graphs for.")
    
    count = 0
    for repo in repos:
        repo_path = os.path.join(SOURCE_DIR, repo)
        G = build_graph(repo_path)
        
        if G.number_of_nodes() > 0:
            output_file = os.path.join(OUTPUT_DIR, f"{repo}.gpickle")
            with open(output_file, 'wb') as f:
                pickle.dump(G, f)
            count += 1
            print(f"-> Generated graph for {repo} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
                
    print(f"DONE. Generated {count} graphs.")

if __name__ == "__main__":
    main()
