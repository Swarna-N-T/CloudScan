#!/usr/bin/env python3
import os
import re
import pandas as pd
import joblib
import argparse
from pathlib import Path

class SecurityPredictor:
    def __init__(self, model_path="ml_results/rf_model.joblib"):
        print(f"🔄 Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        # Expected feature order from training
        self.feature_cols = [
            'sg_has_wide_cidr', 'sg_ssh_exposed', 'sg_rdp_exposed', 
            'sg_unrestricted_egress', 's3_public_access', 's3_no_encryption', 
            's3_no_versioning', 's3_no_logging', 'ec2_public_ip', 
            'ec2_unencrypted_ebs', 'ec2_no_monitoring', 'iam_wildcard_actions', 
            'iam_wildcard_resources', 'encryption_missing', 
            'encryption_kms_missing', 'has_hardcoded_secrets', 
            'line_count', 'resource_count'
        ]

    def extract_features(self, file_path):
        """Extract the 18 security features from a single .tf file"""
        with open(file_path, 'r', errors='ignore') as f:
            content = f.read()
            lines = content.splitlines()

        features = {col: 0 for col in self.feature_cols}
        
        # 1. Line and Resource Count
        features['line_count'] = len(lines)
        features['resource_count'] = len(re.findall(r'resource\s+"', content))

        # 2. Security Patterns
        # Wide CIDR (0.0.0.0/0)
        if re.search(r'0\.0\.0\.0/0', content):
            features['sg_has_wide_cidr'] = 1
        
        # Exposed Ports
        if re.search(r'from_port\s*=\s*22', content) or re.search(r'to_port\s*=\s*22', content):
            features['sg_ssh_exposed'] = 1
        if re.search(r'from_port\s*=\s*3389', content) or re.search(r'to_port\s*=\s*3389', content):
            features['sg_rdp_exposed'] = 1
            
        # Unrestricted Egress
        if re.search(r'egress\s*\{[^}]*0\.0\.0\.0/0', content, re.DOTALL):
            features['sg_unrestricted_egress'] = 1
            
        # S3 Features
        if re.search(r'acl\s*=\s*"public-read"', content):
            features['s3_public_access'] = 1
        if "server_side_encryption_configuration" not in content and 'resource "aws_s3_bucket"' in content:
            features['s3_no_encryption'] = 1
        if "versioning" not in content and 'resource "aws_s3_bucket"' in content:
            features['s3_no_versioning'] = 1
        if "logging" not in content and 'resource "aws_s3_bucket"' in content:
            features['s3_no_logging'] = 1

        # EC2 Features
        if re.search(r'associate_public_ip_address\s*=\s*true', content):
            features['ec2_public_ip'] = 1
        if 'root_block_device' in content and 'encrypted = true' not in content:
            features['ec2_unencrypted_ebs'] = 1
        if 'monitoring = true' not in content and 'resource "aws_instance"' in content:
            features['ec2_no_monitoring'] = 1

        # IAM Wildcards
        if re.search(r'"Action":\s*"\*"', content) or re.search(r'"Action":\s*\[\s*"\*"', content):
            features['iam_wildcard_actions'] = 1
        if re.search(r'"Resource":\s*"\*"', content) or re.search(r'"Resource":\s*\[\s*"\*"', content):
            features['iam_wildcard_resources'] = 1

        # Generic Missing Encryption/KMS
        if 'kms_key_id' not in content and ('encryption' in content.lower() or 'encrypted' in content.lower()):
            features['encryption_kms_missing'] = 1
        if 'encrypted = true' not in content and 'encryption' in content.lower():
            features['encryption_missing'] = 1

        # Hardcoded Secrets
        secret_patterns = [r'password\s*=\s*"[^"]+"', r'aws_secret_access_key\s*=\s*"[^"]+"', r'api_key\s*=\s*"[^"]+"']
        for pattern in secret_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                features['has_hardcoded_secrets'] = 1
                break

        return pd.DataFrame([features])[self.feature_cols]

    def predict(self, path):
        path = Path(path)
        if path.is_file():
            files = [path]
        else:
            files = list(path.glob('**/*.tf'))

        if not files:
            print("❌ No .tf files found.")
            return

        print(f"🔍 Analyzing {len(files)} file(s)...")
        print("="*60)
        print(f"{'FILE':<40} | {'RISK':<10} | {'CONFIDENCE'}")
        print("-"*60)

        for f in files:
            X = self.extract_features(f)
            pred = self.model.predict(X)[0]
            prob = self.model.predict_proba(X)[0][pred]
            
            risk = "⚠️ INSECURE" if pred == 1 else "✅ SECURE"
            color = "\033[91m" if pred == 1 else "\033[92m"
            reset = "\033[0m"
            
            # Show relative path instead of just filename
            rel_path = f.relative_to(path)
            print(f"{str(rel_path):<40} | {color}{risk:<10}{reset} | {prob:.1%}")

        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Predict security risks in Terraform files")
    parser.add_argument("path", help="Path to a .tf file or directory containing .tf files")
    parser.add_argument("--model", default="ml_results/rf_model.joblib", help="Path to the trained model")
    
    args = parser.parse_args()
    
    predictor = SecurityPredictor(args.model)
    predictor.predict(args.path)

if __name__ == "__main__":
    main()
