"""
Gemini LoRA Fine-Tuning Script for Terraform Remediation
=========================================================
This script prepares training data from your Terraform dataset and launches
a supervised fine-tuning job via the Gemini API.

Usage:
    1. Set GEMINI_API_KEY in your .env file
    2. Prepare training pairs in training_data/ directory (see below)
    3. Run:  python finetune_gemini.py

Training Data Format:
    Place JSON files in the training_data/ directory.
    Each file should be a JSON array of objects:
    [
        {
            "input": "resource \"aws_s3_bucket\" \"bad\" { acl = \"public-read\" }",
            "output": "Risk: Public S3 bucket.\\nFix:\\nresource \"aws_s3_bucket\" \"bad\" {\\n  acl = \"private\"\\n}"
        },
        ...
    ]
"""

import os
import json
import glob
import argparse
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("⚠️  Missing GEMINI_API_KEY in .env file")

genai.configure(api_key=api_key)

TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), "training_data")


def load_training_pairs(data_dir):
    """
    Load all JSON training pair files from the training_data directory.
    Each file should contain a list of {"input": ..., "output": ...} dicts.
    """
    pairs = []
    json_files = glob.glob(os.path.join(data_dir, "*.json"))

    if not json_files:
        print(f"⚠️  No JSON files found in {data_dir}")
        print("   Please create training data files first. See the docstring for format.")
        return pairs

    for path in sorted(json_files):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                pairs.extend(data)
                print(f"  Loaded {len(data)} pairs from {os.path.basename(path)}")
        except Exception as e:
            print(f"  ⚠️  Failed to load {path}: {e}")

    return pairs


def build_training_examples(pairs):
    """
    Convert training pairs into the format expected by the Gemini tuning API.
    """
    examples = []
    for pair in pairs:
        text_input = pair.get("input", "")
        text_output = pair.get("output", "")
        if text_input and text_output:
            examples.append({
                "text_input": text_input,
                "output": text_output,
            })
    return examples


def generate_training_data_from_samples(sample_data_dir, output_path):
    """
    Auto-generate training data from the sample_data directory.
    This creates input-output pairs where:
      - Input:  The raw Terraform HCL code
      - Output: A description of potential risks (placeholder — you should review and edit)
    """
    import hcl2

    pairs = []
    repos = [d for d in os.listdir(sample_data_dir)
             if os.path.isdir(os.path.join(sample_data_dir, d))]

    for repo in repos:
        repo_path = os.path.join(sample_data_dir, repo)
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".tf"):
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            tf_content = f.read()

                        if len(tf_content.strip()) < 10:
                            continue

                        # Create a training pair
                        # NOTE: You must manually review and correct the "output" field
                        # to provide the actual remediation for each misconfiguration
                        pairs.append({
                            "input": (
                                "Analyze the following Terraform configuration for "
                                "security misconfigurations and provide remediation:\n\n"
                                f"{tf_content}"
                            ),
                            "output": (
                                "TODO: Replace this with the actual remediation advice "
                                "for this Terraform configuration. Include:\n"
                                "1. What the risk is\n"
                                "2. The corrected Terraform code"
                            ),
                        })
                    except:
                        pass

    if pairs:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2)
        print(f"\n✅ Generated {len(pairs)} training pairs → {output_path}")
        print("⚠️  IMPORTANT: You must manually review and fill in the 'output' fields!")
    else:
        print("No .tf files found to generate training data from.")

    return pairs


def start_finetuning(training_examples, model_name="terraform-remediation", epochs=5):
    """
    Launch a Gemini supervised fine-tuning job with LoRA.
    """
    if not training_examples:
        print("❌ No training examples provided. Aborting.")
        return None

    print(f"\n🚀 Starting fine-tuning job:")
    print(f"   Model display name: {model_name}")
    print(f"   Training examples:  {len(training_examples)}")
    print(f"   Epochs:             {epochs}")
    print(f"   Base model:         gemini-2.0-flash")
    print()

    try:
        operation = genai.create_tuned_model(
            display_name=model_name,
            source_model="models/gemini-2.0-flash-001",
            epoch_count=epochs,
            training_data=training_examples,
        )

        print("⏳ Fine-tuning job submitted. Waiting for completion...")
        print(f"   (This may take several minutes to hours depending on dataset size)\n")

        # Wait for the tuning to complete
        result = operation.result()

        print(f"\n✅ Fine-tuning complete!")
        print(f"   Tuned model name: {result.name}")
        print(f"\n   To use this model, set in your .env file:")
        print(f"   GEMINI_FINETUNED_MODEL={result.name}")

        return result

    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        print("\n   Common issues:")
        print("   - Insufficient training examples (need at least 10-20)")
        print("   - API quota limits")
        print("   - Invalid training data format")
        return None


def list_tuned_models():
    """List all existing tuned models."""
    print("\n📋 Your Tuned Models:")
    print("-" * 50)
    found = False
    for model in genai.list_tuned_models():
        found = True
        print(f"  Name:   {model.name}")
        print(f"  Display: {model.display_name}")
        print(f"  State:  {model.state}")
        print(f"  ---")
    if not found:
        print("  (No tuned models found)")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Gemini for Terraform remediation")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate — auto-create training data from sample .tf files
    gen_parser = subparsers.add_parser("generate", help="Generate training data from sample .tf files")
    gen_parser.add_argument("--samples", type=str,
                            default=os.path.join(os.path.dirname(__file__), "sample_data"),
                            help="Path to sample_data directory")

    # train — launch the fine-tuning job
    train_parser = subparsers.add_parser("train", help="Start fine-tuning job")
    train_parser.add_argument("--name",   type=str, default="terraform-remediation", help="Model display name")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")

    # list — show existing tuned models
    subparsers.add_parser("list", help="List existing tuned models")

    args = parser.parse_args()

    if args.command == "generate":
        output_path = os.path.join(TRAINING_DATA_DIR, "auto_generated.json")
        generate_training_data_from_samples(args.samples, output_path)

    elif args.command == "train":
        pairs = load_training_pairs(TRAINING_DATA_DIR)
        if pairs:
            examples = build_training_examples(pairs)
            start_finetuning(examples, model_name=args.name, epochs=args.epochs)
        else:
            print("No training data found. Run 'python finetune_gemini.py generate' first.")

    elif args.command == "list":
        list_tuned_models()

    else:
        parser.print_help()
