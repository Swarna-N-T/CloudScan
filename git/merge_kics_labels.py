#!/usr/bin/env python3
"""
Merge KICS Labels with Terraform Dataset
Parses KICS scan results and adds labels to the existing dataset
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

class KICSLabelMerger:
    """Merge KICS scan results with existing dataset"""
    
    def __init__(self, kics_results_file: str, dataset_file: str):
        self.kics_results_file = Path(kics_results_file)
        self.dataset_file = Path(dataset_file)
        self.kics_data = None
        self.dataset = []
        self.file_findings = defaultdict(list)
        
    def load_kics_results(self):
        """Load KICS scan results"""
        print(f"📂 Loading KICS results from: {self.kics_results_file}")
        
        with open(self.kics_results_file, 'r') as f:
            self.kics_data = json.load(f)
        
        total_findings = len(self.kics_data.get('queries', []))
        print(f"✓ Loaded {total_findings:,} KICS findings")
        
        # Index findings by file path
        for query in self.kics_data.get('queries', []):
            batch_name = query.get('batch', '')
            
            for file_info in query.get('files', []):
                file_path = file_info.get('file_name', '')
                
                # Normalize path:
                # 1. Strip KICS-internal prefixes (../../path/, ./path/, etc.)
                stripped_path = file_path
                prefixes_to_strip = ['../../path/', '../path/', './path/', '/path/', 'path/']
                for prefix in prefixes_to_strip:
                    if stripped_path.startswith(prefix):
                        stripped_path = stripped_path[len(prefix):]
                        break
                
                # 2. Prepend batch name to match the dataset format: batch_NNNN/path/to/file
                if batch_name:
                    normalized_path = f"{batch_name}/{stripped_path.lstrip('/')}"
                else:
                    normalized_path = stripped_path.lstrip('/')
                
                self.file_findings[normalized_path].append({
                    'query_id': query.get('query_id'),
                    'query_name': query.get('query_name'),
                    'severity': query.get('severity'),
                    'category': query.get('category'),
                    'description': query.get('description'),
                    'line': file_info.get('line', 0)
                })
        
        print(f"✓ Indexed findings for {len(self.file_findings):,} files")
    
    def load_dataset(self):
        """Load existing dataset"""
        print(f"\n📂 Loading dataset from: {self.dataset_file}")
        
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.fieldnames = reader.fieldnames
            for row in reader:
                self.dataset.append(row)
        
        print(f"✓ Loaded {len(self.dataset):,} rows")
    
    def add_kics_labels(self):
        """Add KICS labels to each dataset row"""
        print(f"\n🏷️  Adding KICS labels to dataset...")
        
        labeled_count = 0
        
        for row in self.dataset:
            file_path = row['file_path']
            
            # Get findings for this file
            findings = self.file_findings.get(file_path, [])
            
            # Add KICS label columns
            row['kics_has_issues'] = len(findings) > 0
            row['kics_total_issues'] = len(findings)
            
            # Count by severity
            severity_counts = defaultdict(int)
            for finding in findings:
                severity = finding['severity'].upper()
                severity_counts[severity] += 1
            
            row['kics_severity_high'] = severity_counts.get('HIGH', 0)
            row['kics_severity_medium'] = severity_counts.get('MEDIUM', 0)
            row['kics_severity_low'] = severity_counts.get('LOW', 0)
            
            # Collect unique categories and query IDs
            categories = set(f['category'] for f in findings if f.get('category'))
            query_ids = set(f['query_id'] for f in findings if f.get('query_id'))
            
            row['kics_categories'] = '|'.join(sorted(categories)) if categories else ''
            row['kics_query_ids'] = '|'.join(sorted(query_ids)) if query_ids else ''
            
            if findings:
                labeled_count += 1
        
        print(f"✓ Added labels to {len(self.dataset):,} rows")
        print(f"  {labeled_count:,} rows have KICS findings ({labeled_count/len(self.dataset)*100:.1f}%)")
    
    def save_labeled_dataset(self, output_file: str):
        """Save the labeled dataset"""
        output_path = Path(output_file)
        print(f"\n💾 Saving labeled dataset to: {output_path}")
        
        # Define new fieldnames with KICS columns
        new_fieldnames = list(self.fieldnames) + [
            'kics_has_issues',
            'kics_total_issues',
            'kics_severity_high',
            'kics_severity_medium',
            'kics_severity_low',
            'kics_categories',
            'kics_query_ids'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(self.dataset)
        
        print(f"✓ Saved {len(self.dataset):,} rows to {output_path}")
        
        # Print statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print labeling statistics"""
        print("\n" + "=" * 60)
        print("KICS LABELING STATISTICS")
        print("=" * 60)
        
        total_rows = len(self.dataset)
        rows_with_issues = sum(1 for row in self.dataset if row['kics_has_issues'])
        rows_clean = total_rows - rows_with_issues
        
        print(f"\nTotal Rows:           {total_rows:,}")
        print(f"Rows with Issues:     {rows_with_issues:,} ({rows_with_issues/total_rows*100:.1f}%)")
        print(f"Clean Rows:           {rows_clean:,} ({rows_clean/total_rows*100:.1f}%)")
        
        # Severity distribution
        total_high = sum(int(row['kics_severity_high']) for row in self.dataset)
        total_medium = sum(int(row['kics_severity_medium']) for row in self.dataset)
        total_low = sum(int(row['kics_severity_low']) for row in self.dataset)
        total_issues = sum(int(row['kics_total_issues']) for row in self.dataset)
        
        print(f"\nTotal KICS Issues:    {total_issues:,}")
        print(f"  HIGH severity:      {total_high:,} ({total_high/total_issues*100 if total_issues else 0:.1f}%)")
        print(f"  MEDIUM severity:    {total_medium:,} ({total_medium/total_issues*100 if total_issues else 0:.1f}%)")
        print(f"  LOW severity:       {total_low:,} ({total_low/total_issues*100 if total_issues else 0:.1f}%)")
        
        # Top categories
        category_counts = defaultdict(int)
        for row in self.dataset:
            if row['kics_categories']:
                for cat in row['kics_categories'].split('|'):
                    category_counts[cat] += 1
        
        if category_counts:
            print(f"\nTop 10 Issue Categories:")
            for i, (cat, count) in enumerate(sorted(category_counts.items(), 
                                                    key=lambda x: x[1], 
                                                    reverse=True)[:10], 1):
                print(f"  {i:2d}. {cat:30s}: {count:6,}")
    
    def run(self, output_file: str):
        """Run the complete labeling process"""
        print("=" * 60)
        print("KICS Label Merger")
        print("=" * 60)
        
        self.load_kics_results()
        self.load_dataset()
        self.add_kics_labels()
        self.save_labeled_dataset(output_file)
        
        print("\n✓ Labeling complete!")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge KICS labels with Terraform dataset')
    parser.add_argument('--kics-results', 
                       default='/home/swarna/Downloads/mainproo/dataset_new/sub-data/kics_results/kics_results_merged.json',
                       help='KICS results JSON file')
    parser.add_argument('--dataset',
                       default='/home/swarna/Downloads/mainproo/dataset_new/sub-data/terraform_dataset.csv',
                       help='Input dataset CSV file')
    parser.add_argument('--output',
                       default='/home/swarna/Downloads/mainproo/dataset_new/sub-data/terraform_dataset_labeled.csv',
                       help='Output labeled dataset CSV file')
    
    args = parser.parse_args()
    
    merger = KICSLabelMerger(args.kics_results, args.dataset)
    merger.run(args.output)


if __name__ == '__main__':
    main()
