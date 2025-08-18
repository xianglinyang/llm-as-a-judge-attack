#!/usr/bin/env python3
"""
Utility to find and organize warmup metrics files.
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def find_all_warmup_files(base_dir: str) -> List[str]:
    """Find all warmup metrics files recursively."""
    patterns = [
        "**/*warmup*.json",
        "**/init_ucb_warmup*.json", 
        "**/ucb_warmup*.json"
    ]
    
    files = []
    for pattern in patterns:
        full_pattern = os.path.join(base_dir, pattern)
        files.extend(glob.glob(full_pattern, recursive=True))
    
    return sorted(list(set(files)))


def extract_metadata(file_path: str) -> Dict[str, Any]:
    """Extract key metadata from a warmup file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle nested structure
        if 'warmup_summary' in data:
            metrics = data['warmup_summary']
        else:
            metrics = data
        
        # Extract key info
        metadata = {
            'file_path': file_path,
            'file_size_kb': os.path.getsize(file_path) / 1024,
            'phase': metrics.get('phase', 'unknown'),
            'rounds': metrics.get('rounds', 0),
            'alpha': metrics.get('alpha', 'N/A'),
            'lambda_reg': metrics.get('lambda_reg', 'N/A'),
            'dataset': metrics.get('dataset_name', 'N/A'),
            'judge': metrics.get('judge_backbone', 'N/A'),
            'time_taken': metrics.get('time_taken', 0),
            'timestamp': metrics.get('timestamp', 'N/A'),
            'convergence': 'Yes' if metrics.get('phase') == 'complete' else 'No',
            'ci_data_points': len(metrics.get('median_ci', [])),
            'gap_data_points': len(metrics.get('median_gap', [])),
        }
        
        # Add final convergence values
        median_ci = metrics.get('median_ci', [])
        median_gap = metrics.get('median_gap', [])
        
        if median_ci:
            metadata['final_ci'] = median_ci[-1]
            metadata['initial_ci'] = median_ci[0] if len(median_ci) > 0 else None
        else:
            metadata['final_ci'] = None
            metadata['initial_ci'] = None
            
        if median_gap:
            metadata['final_gap'] = median_gap[-1]
            metadata['initial_gap'] = median_gap[0] if len(median_gap) > 0 else None
        else:
            metadata['final_gap'] = None
            metadata['initial_gap'] = None
            
        return metadata
        
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'file_size_kb': os.path.getsize(file_path) / 1024 if os.path.exists(file_path) else 0
        }


def create_summary_table(metadata_list: List[Dict[str, Any]]) -> str:
    """Create a formatted summary table."""
    if not metadata_list:
        return "No warmup files found."
    
    # Filter out files with errors
    valid_files = [m for m in metadata_list if 'error' not in m]
    error_files = [m for m in metadata_list if 'error' in m]
    
    if not valid_files:
        return f"Found {len(error_files)} files with errors."
    
    # Create table
    headers = ["File", "Phase", "Rounds", "α", "λ", "Dataset", "Final CI", "Final Gap", "Time(s)"]
    rows = []
    
    for meta in valid_files:
        filename = os.path.basename(meta['file_path'])
        final_ci = f"{meta['final_ci']:.4f}" if meta['final_ci'] is not None else "N/A"
        final_gap = f"{meta['final_gap']:.4f}" if meta['final_gap'] is not None else "N/A"
        
        row = [
            filename[:30] + "..." if len(filename) > 30 else filename,
            meta['phase'],
            str(meta['rounds']),
            str(meta['alpha']),
            str(meta['lambda_reg']),
            str(meta['dataset'])[:10],
            final_ci,
            final_gap,
            f"{meta['time_taken']:.1f}"
        ]
        rows.append(row)
    
    # Calculate column widths
    all_rows = [headers] + rows
    col_widths = [max(len(str(item)) for item in col) for col in zip(*all_rows)]
    
    def format_row(row):
        return " | ".join(str(item).ljust(width) for item, width in zip(row, col_widths))
    
    table_lines = [
        format_row(headers),
        "-" * (sum(col_widths) + 3 * (len(headers) - 1)),
        *[format_row(row) for row in rows]
    ]
    
    if error_files:
        table_lines.extend([
            "",
            f"Files with errors ({len(error_files)}):",
            *[f"  - {os.path.basename(m['file_path'])}: {m['error']}" for m in error_files]
        ])
    
    return "\n".join(table_lines)


def group_by_experiment(metadata_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group files by experimental conditions."""
    groups = {}
    
    for meta in metadata_list:
        if 'error' in meta:
            continue
            
        # Create grouping key
        key_parts = [
            f"dataset_{meta['dataset']}",
            f"alpha_{meta['alpha']}",
            f"lambda_{meta['lambda_reg']}"
        ]
        key = "_".join(str(part) for part in key_parts)
        
        if key not in groups:
            groups[key] = []
        groups[key].append(meta)
    
    return groups


def suggest_plotting_commands(metadata_list: List[Dict[str, Any]], base_dir: str):
    """Suggest useful plotting commands based on found files."""
    valid_files = [m for m in metadata_list if 'error' not in m]
    
    if not valid_files:
        return "No valid files found for plotting."
    
    suggestions = []
    
    # Single file examples
    if len(valid_files) >= 1:
        best_file = max(valid_files, key=lambda x: x['rounds'])
        suggestions.append(f"# Plot single best warmup run ({best_file['rounds']} rounds):")
        suggestions.append(f"python scripts/visualization/plot_warmup_metrics.py \\")
        suggestions.append(f"  --metrics_file '{best_file['file_path']}' \\")
        suggestions.append(f"  --show_table")
        suggestions.append("")
    
    # Directory comparison
    if len(valid_files) > 1:
        suggestions.append(f"# Compare all {len(valid_files)} warmup runs:")
        suggestions.append(f"python scripts/visualization/plot_warmup_metrics.py \\")
        suggestions.append(f"  --metrics_dir '{base_dir}' \\")
        suggestions.append(f"  --compare --show_table \\")
        suggestions.append(f"  --save_dir warmup_analysis/")
        suggestions.append("")
    
    # Group-specific analysis
    groups = group_by_experiment(valid_files)
    if len(groups) > 1:
        suggestions.append(f"# Found {len(groups)} different experimental conditions:")
        for i, (group_key, group_files) in enumerate(list(groups.items())[:3]):
            suggestions.append(f"#   Group {i+1}: {group_key} ({len(group_files)} files)")
        suggestions.append("")
    
    return "\n".join(suggestions)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Find and analyze warmup metrics files")
    parser.add_argument("base_dir", help="Base directory to search for warmup files")
    parser.add_argument("--show_files", action="store_true", help="Show individual file paths")
    parser.add_argument("--show_suggestions", action="store_true", help="Show plotting command suggestions")
    parser.add_argument("--group_by", choices=["experiment", "dataset", "alpha"], help="Group files by criteria")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_dir):
        print(f"Directory not found: {args.base_dir}")
        return
    
    print(f"Searching for warmup files in: {args.base_dir}")
    print("="*60)
    
    # Find all files
    files = find_all_warmup_files(args.base_dir)
    print(f"Found {len(files)} warmup files")
    
    if not files:
        print("No warmup files found. Make sure you've run the warmup script first:")
        print("python src/evolve_agent/bandit/init_LinUCB_warmup.py")
        return
    
    # Extract metadata
    print("Analyzing files...")
    metadata_list = []
    for file_path in files:
        metadata = extract_metadata(file_path)
        metadata_list.append(metadata)
    
    # Show summary table
    print("\n" + "="*60)
    print("WARMUP FILES SUMMARY")
    print("="*60)
    print(create_summary_table(metadata_list))
    
    # Show individual files if requested
    if args.show_files:
        print(f"\n" + "="*60)
        print("INDIVIDUAL FILES")
        print("="*60)
        for meta in metadata_list:
            if 'error' not in meta:
                print(f"{meta['file_path']}")
                print(f"  Phase: {meta['phase']}, Rounds: {meta['rounds']}, CI points: {meta['ci_data_points']}")
            else:
                print(f"{meta['file_path']} (ERROR: {meta['error']})")
    
    # Show grouping if requested
    if args.group_by:
        groups = group_by_experiment(metadata_list)
        print(f"\n" + "="*60)
        print(f"GROUPED BY EXPERIMENT")
        print("="*60)
        for group_key, group_files in groups.items():
            print(f"\n{group_key}: {len(group_files)} files")
            for meta in group_files:
                print(f"  - {os.path.basename(meta['file_path'])}")
    
    # Show plotting suggestions
    if args.show_suggestions:
        print(f"\n" + "="*60)
        print("SUGGESTED PLOTTING COMMANDS")
        print("="*60)
        suggestions = suggest_plotting_commands(metadata_list, args.base_dir)
        print(suggestions)


if __name__ == "__main__":
    main()
