#!/usr/bin/env python3
"""
Score plotting and analysis for prompt injection attack results.
Analyzes original vs final scores from various prompt injection attack trajectory files including:
- naive_attack
- null_model_attack (and null_model_attack_rs)
- fake_completion_attack  
- escape_attack
- context_ignore_attack

Usage:
    python plot_scores.py --trajectories_dir path/to/trajectories --show_table
    python plot_scores.py --trajectories_dir path/to/trajectories --group_by judge_model,dataset
    python plot_scores.py --trajectories_dir path/to/trajectories --filter judge_backbone=gpt-4
    python plot_scores.py --trajectories_dir path/to/trajectories --strategies naive_attack,fake_completion_attack
"""

import argparse
import json
import os
import glob
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Supported prompt injection attack strategies
SUPPORTED_ATTACK_STRATEGIES = {
    'naive_attack',
    'null_model_attack', 
    'null_model_attack_rs',
    'fake_completion_attack',
    'escape_attack',
    'context_ignore_attack'
}

def load_prompt_injection_attack_data(file_path: str, exclude_filters: Dict[str, List[str]] = None, allowed_strategies: List[str] = None) -> Dict[str, Any]:
    """Load prompt injection attack data from a trajectory JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if this is a supported prompt injection attack file
        strategy = data.get('strategy', '')
        if strategy not in SUPPORTED_ATTACK_STRATEGIES:
            return None
            
        # Check if strategy is in allowed list (if specified)
        if allowed_strategies and strategy not in allowed_strategies:
            return None
            
        # Apply exclude filters if provided
        if exclude_filters:
            for key, values in exclude_filters.items():
                # Map metadata field names to actual JSON field names
                field_mapping = {
                    'llm_agent': 'llm_agent_name',
                    'judge_model': 'judge_backbone',
                    'response_model': 'response_model_name',
                    'baseline_model': 'baseline_response_model_name',
                    'dataset': 'dataset_name',
                    'budget': 'budget',
                    'pool_size': 'pool_size',
                    'judge_type': 'judge_type',
                    'reward_type': 'reward_type',
                    'strategy': 'strategy'
                }
                
                actual_field = field_mapping.get(key, key)
                data_value = str(data.get(actual_field, '')).lower()
                
                # Check if any of the exclude values match
                for value in values:
                    value_lower = value.lower()
                    if data_value == value_lower or value_lower in data_value:
                        return None
        
        return data
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def parse_exclude_filters(exclude_string: str) -> Dict[str, List[str]]:
    """Parse exclude filters from 'key1=value1,key2=value2' format."""
    if not exclude_string:
        return {}
    
    filters = {}
    pairs = exclude_string.split(',')
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            if key in filters:
                filters[key].append(value)
            else:
                filters[key] = [value]
        else:
            print(f"Warning: Invalid filter format '{pair}'. Expected 'key=value'")
    
    return filters

def extract_metadata_from_data(data: Dict[str, Any]) -> Dict[str, str]:
    """Extract metadata for grouping from loaded prompt injection attack data."""
    metadata = {}
    
    # Strategy information (any supported prompt injection attack)
    metadata['strategy'] = data.get('strategy', 'unknown')
    
    # Model information
    metadata['judge_model'] = data.get('judge_backbone', 'unknown')
    metadata['llm_agent'] = data.get('llm_agent_name', 'unknown')
    metadata['response_model'] = data.get('response_model_name', 'unknown')
    metadata['baseline_model'] = data.get('baseline_response_model_name', 'unknown')
    
    # Dataset information
    metadata['dataset'] = data.get('dataset_name', 'unknown')
    
    # Configuration information
    metadata['budget'] = str(data.get('budget', 'unknown'))
    metadata['pool_size'] = str(data.get('pool_size', 'unknown'))
    metadata['judge_type'] = data.get('judge_type', 'unknown')
    metadata['reward_type'] = data.get('reward_type', 'unknown')
    metadata['answer_position'] = data.get('answer_position', 'unknown')
    
    # Time information
    metadata['timestamp'] = data.get('timestamp', 'unknown')
    
    return metadata

def extract_scores_from_data(data: Dict[str, Any]) -> Dict[str, List[float]]:
    """Extract original and final scores from prompt injection attack data."""
    scores = {
        'original_scores': [],
        'final_scores': [],
        'improvements': [],
        'categories': []
    }
    
    if 'trajectories' in data and isinstance(data['trajectories'], list):
        for item in data['trajectories']:
            original_score = float(item.get('original_score', 0))
            final_score = float(item.get('final_score', 0))
            category = item.get('category', 'unknown')
            
            scores['original_scores'].append(original_score)
            scores['final_scores'].append(final_score)
            scores['improvements'].append(final_score - original_score)
            scores['categories'].append(category)
    
    return scores

def find_all_prompt_injection_attack_files(trajectories_dir: str) -> List[str]:
    """Find all prompt injection attack trajectory files."""
    attack_patterns = [
        "naive_attack_*.json",
        "null_model_attack_*.json", 
        "fake_completion_attack_*.json",
        "escape_attack_*.json",
        "context_ignore_attack_*.json"
    ]
    
    files = []
    for pattern in attack_patterns:
        # Search both in root and subdirectories
        files.extend(glob.glob(os.path.join(trajectories_dir, pattern), recursive=False))
        files.extend(glob.glob(os.path.join(trajectories_dir, "**", pattern), recursive=True))
    
    return sorted(list(set(files)))

def categorize_data_by_groups(all_data: List[Dict[str, Any]], group_by: List[str]) -> Dict[Tuple, List[Dict[str, Any]]]:
    """Categorize data by specified grouping dimensions."""
    grouped_data = defaultdict(list)
    
    for item in all_data:
        metadata = item['metadata']
        
        # Create grouping key based on specified dimensions
        group_key = tuple(metadata.get(dim, 'unknown') for dim in group_by)
        grouped_data[group_key].append(item)
    
    return dict(grouped_data)

def plot_score_comparison(grouped_data: Dict[Tuple, List[Dict]], group_by: List[str], save_dir: str = None):
    """Plot original vs final scores comparison."""
    
    # Determine subplot layout
    n_groups = len(grouped_data)
    if n_groups == 0:
        print("No data to plot")
        return
    
    # Calculate subplot grid
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Ensure axes is always a 2D array for consistent indexing
    if n_groups == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (group_key, group_items) in enumerate(grouped_data.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Group title
        group_title = ' | '.join([f"{dim}: {val}" for dim, val in zip(group_by, group_key)])
        ax.set_title(group_title, fontsize=10, pad=20)
        
        # Collect all scores for this group
        all_original = []
        all_final = []
        
        for item in group_items:
            scores = item['scores']
            all_original.extend(scores['original_scores'])
            all_final.extend(scores['final_scores'])
        
        if all_original and all_final:
            # Create scatter plot
            ax.scatter(all_original, all_final, alpha=0.6, s=50)
            
            # Add diagonal line (y=x) for reference
            min_score = min(min(all_original), min(all_final))
            max_score = max(max(all_original), max(all_final))
            ax.plot([min_score, max_score], [min_score, max_score], 'r--', alpha=0.8, label='No Change')
            
            ax.set_xlabel('Original Score')
            ax.set_ylabel('Final Score (After Attack)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            improvement = np.mean(all_final) - np.mean(all_original)
            success_rate = np.mean(np.array(all_final) > np.array(all_original)) * 100
            ax.text(0.05, 0.95, f'Avg Improvement: {improvement:.3f}\nSuccess Rate: {success_rate:.1f}%', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(n_groups, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle(f'Prompt Injection Attack Score Comparison Grouped by {" | ".join(group_by)}', fontsize=14)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        group_name = "_".join(group_by)
        plt.savefig(os.path.join(save_dir, f'prompt_injection_scores_{group_name}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'prompt_injection_scores_{group_name}.pdf'), bbox_inches='tight')
    
    plt.show()

def plot_improvement_distribution(grouped_data: Dict[Tuple, List[Dict]], group_by: List[str], save_dir: str = None):
    """Plot distribution of score improvements."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for group_key, group_items in grouped_data.items():
        group_label = ' | '.join([f"{dim}={val}" for dim, val in zip(group_by, group_key)])
        
        # Collect all improvements for this group
        all_improvements = []
        for item in group_items:
            scores = item['scores']
            all_improvements.extend(scores['improvements'])
        
        if all_improvements:
            ax.hist(all_improvements, bins=20, alpha=0.6, label=group_label)
    
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='No Change')
    ax.set_xlabel('Score Improvement (Final - Original)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Score Improvements Grouped by {" | ".join(group_by)}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        group_name = "_".join(group_by)
        plt.savefig(os.path.join(save_dir, f'prompt_injection_improvements_{group_name}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'prompt_injection_improvements_{group_name}.pdf'), bbox_inches='tight')
    
    plt.show()

def plot_summary_table_by_groups(grouped_data: Dict[Tuple, List[Dict]], group_by: List[str]):
    """Create summary table showing performance across groups."""
    
    # Collect data for table
    table_data = []
    
    for group_key, group_items in grouped_data.items():
        group_dict = {dim: val for dim, val in zip(group_by, group_key)}
        
        # Collect all scores for this group
        all_original = []
        all_final = []
        all_improvements = []
        total_items = 0
        
        for item in group_items:
            scores = item['scores']
            all_original.extend(scores['original_scores'])
            all_final.extend(scores['final_scores'])
            all_improvements.extend(scores['improvements'])
            total_items += len(scores['original_scores'])
        
        if all_original and all_final:
            # Calculate statistics
            mean_original = np.mean(all_original)
            mean_final = np.mean(all_final)
            mean_improvement = np.mean(all_improvements)
            std_improvement = np.std(all_improvements)
            success_rate = np.mean(np.array(all_final) > np.array(all_original)) * 100
            max_improvement = np.max(all_improvements)
            min_improvement = np.min(all_improvements)
            
            row = group_dict.copy()
            row.update({
                'runs': len(group_items),
                'total_items': total_items,
                'mean_original_score': mean_original,
                'mean_final_score': mean_final,
                'mean_improvement': mean_improvement,
                'std_improvement': std_improvement,
                'success_rate_%': success_rate,
                'max_improvement': max_improvement,
                'min_improvement': min_improvement
            })
            table_data.append(row)
    
    # Create DataFrame and display
    df = pd.DataFrame(table_data)
    
    if df.empty:
        print("No data available for summary table")
        return df
    
    # Sort by group dimensions then by mean improvement
    sort_cols = group_by + ['mean_improvement']
    df = df.sort_values(sort_cols, ascending=[True]*len(group_by) + [False])
    
    print(f"\n{'='*150}")
    print(f"PROMPT INJECTION ATTACK PERFORMANCE SUMMARY GROUPED BY: {', '.join(group_by).upper()}")
    print(f"{'='*150}")
    
    # Display with better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(df.to_string(index=False))
    print(f"{'='*150}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Prompt injection attack score analysis and plotting")
    parser.add_argument("--trajectories_dir", type=str, help="Directory containing prompt injection attack trajectory files", 
                       default="/data2/xianglin/A40/llm-as-a-judge-attack/trajectories")
    parser.add_argument("--group_by", type=str, default="judge_model,dataset", 
                       help="Comma-separated list of dimensions to group by. Options: strategy, dataset, judge_model, llm_agent, response_model, baseline_model, judge_type, reward_type, answer_position")
    parser.add_argument("--save_dir", type=str, help="Directory to save plots", default="./plots")
    parser.add_argument("--show_table", action="store_true", help="Show summary table")
    parser.add_argument("--plot_type", type=str, default="both", choices=["scatter", "histogram", "both"],
                       help="Type of plot to generate")
    parser.add_argument("--filter", type=str, help="Filter data (e.g., 'dataset=AlpacaEval,judge_type=pointwise')")
    parser.add_argument("--exclude", type=str, help="Exclude files matching criteria (format: 'key1=value1,key2=value2')")
    parser.add_argument("--strategies", type=str, help="Comma-separated list of attack strategies to include (e.g., 'naive_attack,fake_completion_attack'). If not specified, includes all supported strategies.")
    
    args = parser.parse_args()
    
    # Parse exclude filters
    exclude_filters = parse_exclude_filters(args.exclude)
    if exclude_filters:
        print(f"Excluding files with: {exclude_filters}")
    
    # Parse allowed strategies
    allowed_strategies = None
    if args.strategies:
        allowed_strategies = [s.strip() for s in args.strategies.split(',')]
        # Validate strategies
        invalid_strategies = set(allowed_strategies) - SUPPORTED_ATTACK_STRATEGIES
        if invalid_strategies:
            print(f"Warning: Unsupported strategies: {invalid_strategies}")
            print(f"Supported strategies: {SUPPORTED_ATTACK_STRATEGIES}")
        allowed_strategies = [s for s in allowed_strategies if s in SUPPORTED_ATTACK_STRATEGIES]
        print(f"Using attack strategies: {allowed_strategies}")
    else:
        print(f"Using all supported attack strategies: {SUPPORTED_ATTACK_STRATEGIES}")
    
    # Parse grouping dimensions
    group_by = [dim.strip() for dim in args.group_by.split(',')]
    
    # Find all prompt injection attack files
    files = find_all_prompt_injection_attack_files(args.trajectories_dir)
    if not files:
        print(f"No prompt injection attack files found in {args.trajectories_dir}")
        return
    
    print(f"Found {len(files)} prompt injection attack files")
    
    # Load all data
    all_data = []
    for file_path in files:
        data = load_prompt_injection_attack_data(file_path, exclude_filters, allowed_strategies)
        if data is None:
            continue
        
        metadata = extract_metadata_from_data(data)
        scores = extract_scores_from_data(data)
        
        all_data.append({
            'file_path': file_path,
            'metadata': metadata,
            'scores': scores
        })
    
    if not all_data:
        print("No valid prompt injection attack data loaded")
        return
    
    print(f"Successfully loaded {len(all_data)} files")
    
    # Apply filters if specified
    if args.filter:
        filter_pairs = [pair.strip().split('=') for pair in args.filter.split(',')]
        filter_dict = {k: v for k, v in filter_pairs}
        
        filtered_data = []
        for item in all_data:
            include = True
            for filter_key, filter_value in filter_dict.items():
                if item['metadata'].get(filter_key, '') != filter_value:
                    include = False
                    break
            if include:
                filtered_data.append(item)
        
        print(f"Applied filters: {filter_dict}")
        print(f"Data after filtering: {len(filtered_data)} files")
        all_data = filtered_data
    
    # Group data by specified dimensions
    grouped_data = categorize_data_by_groups(all_data, group_by)
    
    print(f"\nGrouped by {group_by}:")
    for group_key, group_items in grouped_data.items():
        group_label = ' | '.join([f"{dim}={val}" for dim, val in zip(group_by, group_key)])
        total_items = sum(len(item['scores']['original_scores']) for item in group_items)
        print(f"  {group_label}: {len(group_items)} files, {total_items} items")
    
    # Show summary table
    if args.show_table:
        summary_df = plot_summary_table_by_groups(grouped_data, group_by)
    
    # Generate plots
    if args.plot_type in ["scatter", "both"]:
        print(f"\nGenerating scatter plots grouped by {group_by}...")
        plot_score_comparison(grouped_data, group_by, args.save_dir)
    
    if args.plot_type in ["histogram", "both"]:
        print(f"\nGenerating improvement distribution plots grouped by {group_by}...")
        plot_improvement_distribution(grouped_data, group_by, args.save_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()
