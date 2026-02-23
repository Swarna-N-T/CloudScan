#filter te TerraDS dataset and extract only repo folders containing aws and atleast 3 terraform files


import os
import shutil

# --- CONFIGURATION ---
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../TerraDS_extracted"))     # Ensure this matches your folder name
DEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../AWSonly_graph_Dataset"))   # Destination
TARGET_COUNT = 1500                    # Goal
# ---------------------

def is_valid_aws_repo(repo_path):
    tf_file_count = 0
    has_aws_provider = False
    
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".tf"):
                tf_file_count += 1
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if 'resource "aws_' in content or 'provider "aws"' in content:
                            has_aws_provider = True
                except:
                    pass
    return has_aws_provider and tf_file_count >= 3

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: {SOURCE_DIR} not found.")
        return

    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    copied_count = 0
    scanned_count = 0
    
    all_items = os.listdir(SOURCE_DIR)
    
    print(f"Resuming scan... Target: {TARGET_COUNT}")

    for item in all_items:
        if copied_count >= TARGET_COUNT:
            break
            
        repo_path = os.path.join(SOURCE_DIR, item)
        dest_path = os.path.join(DEST_DIR, item)
        scanned_count += 1

        # Skip if already copied (so you don't start from zero)
        if os.path.exists(dest_path):
            copied_count += 1
            continue

        if os.path.isdir(repo_path):
            # Check validity
            if is_valid_aws_repo(repo_path):
                try:
                    # FIX 1: symlinks=True (Copy the link, don't follow it)
                    # FIX 2: ignore_dangling_symlinks=True (Don't crash on broken links)
                    shutil.copytree(repo_path, dest_path, symlinks=True, ignore_dangling_symlinks=True)
                    
                    copied_count += 1
                    print(f"[{copied_count}/{TARGET_COUNT}] Copied: {item}")
                
                except shutil.Error as e:
                    # FIX 3: If it still errors, print it but DO NOT CRASH
                    print(f"  [!] Warning: Issues copying {item}. Partial copy kept.")
                except OSError as e:
                    print(f"  [!] Error: Could not copy {item} - {e}")

    print("-" * 30)
    print(f"DONE. Total valid AWS repos in {DEST_DIR}: {copied_count}")

if __name__ == "__main__":
    main()
