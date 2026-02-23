#label filtered repo folders using chekhov

import os
import subprocess
import concurrent.futures

# --- CONFIGURATION ---
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../AWSonly_graph_Dataset")) 
# ---------------------

def scan_repo(repo_name):
    repo_path = os.path.join(DATASET_DIR, repo_name)
    output_file = os.path.join(repo_path, "checkov_report.json")
    
    # Skip if already scanned to save time on re-runs
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        return f"Skipped (Already done): {repo_name}"

    # The Checkov Command (FULL SCAN)
    # We removed the '--check' flag, so it runs EVERYTHING.
    cmd = [
        "checkov",
        "-d", repo_path,
        "--framework", "terraform",
        "--output", "json",
        "--quiet" 
    ]

    try:
        # Run Checkov and capture output
        with open(output_file, "w") as f:
            # Increased timeout to 120s because full scans take longer
            subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL, timeout=120)
        return f"Scanned: {repo_name}"
    except subprocess.TimeoutExpired:
        return f"Timeout (Skipped): {repo_name}"
    except Exception as e:
        return f"Error on {repo_name}: {str(e)}"

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset folder '{DATASET_DIR}' not found!")
        return

    repos = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    total = len(repos)
    print(f"Starting FULL labeling for {total} repositories...")
    print("This will take significantly longer (approx 45-60 mins).")

    # Run in parallel (Use 4 CPU cores)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(scan_repo, repos)
        
        count = 0
        for res in results:
            count += 1
            if count % 20 == 0:
                print(f"[{count}/{total}] Progress update...")

    print("Done! All JSON reports generated.")

if __name__ == "__main__":
    main()
