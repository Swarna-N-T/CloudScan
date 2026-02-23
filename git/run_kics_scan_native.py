#!/usr/bin/env python3
"""
KICS Native Scanner for Terraform Dataset
Scans all Terraform files using KICS native binary and generates labeled results
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import time

class KICSNativeScanner:
    """Run KICS scans using native binary"""
    
    def __init__(self, base_dir: str, output_dir: str = None, kics_path: str = "kics"):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir) if output_dir else self.base_dir
        self.kics_path = kics_path
        self.results = []
        self.scan_stats = {
            'total_batches': 0,
            'scanned_batches': 0,
            'failed_batches': 0,
            'total_files': 0,
            'total_findings': 0,
            'start_time': None,
            'end_time': None
        }
        
    def check_kics(self) -> bool:
        """Check if KICS is available"""
        try:
            result = subprocess.run(
                [self.kics_path, 'version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ KICS found: {result.stdout.strip()}")
                return True
            return False
        except FileNotFoundError:
            print(f"❌ KICS not found at: {self.kics_path}")
            print("Please install KICS or provide the correct path")
            return False
        except Exception as e:
            print(f"❌ KICS check failed: {e}")
            return False
    
    def get_batch_directories(self) -> List[Path]:
        """Get all batch directories to scan"""
        batch_dirs = sorted([d for d in self.base_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('batch_')])
        return batch_dirs
    
    def scan_batch(self, batch_dir: Path) -> Dict[str, Any]:
        """Scan a single batch directory using KICS native"""
        batch_name = batch_dir.name
        output_file = self.output_dir / f"kics_results_{batch_name}.json"
        
        print(f"\n📂 Scanning {batch_name}...")
        
        # Prepare KICS command
        kics_cmd = [
            self.kics_path,
            'scan',
            '-p', str(batch_dir.absolute()),
            '-o', str(self.output_dir.absolute()),
            '--output-name', f'kics_results_{batch_name}',
            '--report-formats', 'json',
            '--exclude-severities', 'info',  # Exclude info-level findings
            '--silent'  # Reduce console output
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(
                kics_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per batch
            )
            elapsed = time.time() - start_time
            
            # KICS returns non-zero exit code when findings are found
            # So we check if output file exists rather than return code
            if output_file.exists():
                with open(output_file, 'r') as f:
                    scan_result = json.load(f)
                
                num_findings = len(scan_result.get('queries', []))
                num_files = scan_result.get('files_scanned', 0)
                
                print(f"  ✓ Scanned {num_files} files, found {num_findings} issues ({elapsed:.1f}s)")
                
                return {
                    'batch': batch_name,
                    'success': True,
                    'findings': num_findings,
                    'files_scanned': num_files,
                    'result_file': str(output_file),
                    'elapsed_time': elapsed
                }
            else:
                print(f"  ⚠️  No output file generated")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")
                return {
                    'batch': batch_name,
                    'success': False,
                    'error': 'No output file generated',
                    'elapsed_time': elapsed
                }
                
        except subprocess.TimeoutExpired:
            print(f"  ❌ Timeout scanning {batch_name}")
            return {
                'batch': batch_name,
                'success': False,
                'error': 'Timeout'
            }
        except Exception as e:
            print(f"  ❌ Error scanning {batch_name}: {e}")
            return {
                'batch': batch_name,
                'success': False,
                'error': str(e)
            }
    
    def merge_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge all batch results into a single JSON file"""
        print("\n📊 Merging results...")
        
        merged = {
            'scan_metadata': {
                'scan_date': datetime.now().isoformat(),
                'base_directory': str(self.base_dir),
                'total_batches': self.scan_stats['total_batches'],
                'scanned_batches': self.scan_stats['scanned_batches'],
                'failed_batches': self.scan_stats['failed_batches'],
                'total_files': self.scan_stats['total_files'],
                'total_findings': self.scan_stats['total_findings'],
                'scan_duration_seconds': (self.scan_stats['end_time'] - 
                                         self.scan_stats['start_time'])
            },
            'queries': [],
            'files': {}
        }
        
        # Merge all individual batch results
        for batch_result in batch_results:
            if not batch_result['success']:
                continue
                
            result_file = Path(batch_result['result_file'])
            if not result_file.exists():
                continue
            
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # Merge queries (findings)
                if 'queries' in data:
                    for query in data['queries']:
                        # Add batch info to each query
                        query['batch'] = batch_result['batch']
                        merged['queries'].append(query)
                
                # Merge file information
                if 'files' in data:
                    merged['files'].update(data['files'])
                    
            except Exception as e:
                print(f"  ⚠️  Error merging {result_file}: {e}")
        
        return merged
    
    def scan_all_batches(self, max_batches: int = None, test_mode: bool = False):
        """Scan all batch directories"""
        print("=" * 60)
        print("KICS Native Batch Scanner")
        print("=" * 60)
        
        # Pre-flight checks
        if not self.check_kics():
            print("\n❌ KICS is not available.")
            print("Install KICS: https://docs.kics.io/latest/getting-started/")
            sys.exit(1)
        
        # Get batch directories
        batch_dirs = self.get_batch_directories()
        
        if test_mode:
            print("\n🧪 TEST MODE: Scanning only first batch")
            batch_dirs = batch_dirs[:1]
        elif max_batches:
            batch_dirs = batch_dirs[:max_batches]
        
        self.scan_stats['total_batches'] = len(batch_dirs)
        print(f"\n📁 Found {len(batch_dirs)} batch directories to scan")
        
        # Start scanning
        self.scan_stats['start_time'] = time.time()
        batch_results = []
        
        for i, batch_dir in enumerate(batch_dirs, 1):
            print(f"\n[{i}/{len(batch_dirs)}]", end=" ")
            result = self.scan_batch(batch_dir)
            batch_results.append(result)
            
            if result['success']:
                self.scan_stats['scanned_batches'] += 1
                self.scan_stats['total_files'] += result.get('files_scanned', 0)
                self.scan_stats['total_findings'] += result.get('findings', 0)
            else:
                self.scan_stats['failed_batches'] += 1
        
        self.scan_stats['end_time'] = time.time()
        
        # Merge all results
        merged_results = self.merge_results(batch_results)
        
        # Save merged results
        output_file = self.output_dir / 'kics_results_merged.json'
        with open(output_file, 'w') as f:
            json.dump(merged_results, f, indent=2)
        
        print(f"\n✓ Merged results saved to: {output_file}")
        
        # Print summary
        self.print_summary()
        
        # Cleanup individual batch result files
        if not test_mode:
            print("\n🧹 Cleaning up individual batch result files...")
            for batch_result in batch_results:
                if batch_result['success'] and 'result_file' in batch_result:
                    try:
                        Path(batch_result['result_file']).unlink()
                    except:
                        pass
        
        return output_file
    
    def print_summary(self):
        """Print scan summary statistics"""
        print("\n" + "=" * 60)
        print("SCAN SUMMARY")
        print("=" * 60)
        
        duration = self.scan_stats['end_time'] - self.scan_stats['start_time']
        
        print(f"Total Batches:       {self.scan_stats['total_batches']}")
        print(f"Successfully Scanned: {self.scan_stats['scanned_batches']}")
        print(f"Failed:              {self.scan_stats['failed_batches']}")
        print(f"Total Files Scanned: {self.scan_stats['total_files']:,}")
        print(f"Total Findings:      {self.scan_stats['total_findings']:,}")
        print(f"Scan Duration:       {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        if self.scan_stats['scanned_batches'] > 0:
            avg_time = duration / self.scan_stats['scanned_batches']
            print(f"Avg Time per Batch:  {avg_time:.1f} seconds")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scan Terraform files with KICS (native)')
    parser.add_argument('--base-dir', default='/home/swarna/Downloads/mainproo/dataset_new/sub-data',
                       help='Base directory containing batch folders')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for results (default: same as base-dir)')
    parser.add_argument('--kics-path', default='kics',
                       help='Path to KICS binary (default: kics in PATH)')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum number of batches to scan (for testing)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: scan only first batch')
    
    args = parser.parse_args()
    
    scanner = KICSNativeScanner(args.base_dir, args.output_dir, args.kics_path)
    output_file = scanner.scan_all_batches(
        max_batches=args.max_batches,
        test_mode=args.test_mode
    )
    
    print(f"\n✓ Scan complete! Results: {output_file}")


if __name__ == '__main__':
    main()
