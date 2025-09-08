#!/usr/bin/env python3
"""
Multi-dimensional comparison visualization for exploration strategies.
Handles comparisons across:
- Strategies (UCB, Random, Baselines)
- Models (Judge models, LLM agents, Response models)
- Datasets (AlpacaEval, UltraFeedback, etc.)
- Configurations (Budget, pool size, hyperparameters)

Usage:
    python plot_multi_dimensional_comparison.py --metrics_dir path/to/metrics --group_by model
    python plot_multi_dimensional_comparison.py --metrics_dir path/to/metrics --group_by dataset
    python plot_multi_dimensional_comparison.py --metrics_dir path/to/metrics --group_by strategy,dataset
"""

import argparse
import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from itertools import product

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_exploration_metrics(file_path: str, exclude_filters: Dict[str, List[str]] = None) -> Dict[str, Any]:
    """Load exploration metrics from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Apply exclude filters if provided
    if exclude_filters:
        for key, values in exclude_filters.items():
            # Map metadata field names to actual JSON field names
            field_mapping = {
                'llm_agent': 'llm_agent_name',
                'judge_model': 'judge_backbone',  # or 'judge_model_name'
                'response_model': 'response_model_name',
                'baseline_model': 'baseline_response_model_name',
                'dataset': 'dataset_name',
                'budget': 'budget',  # or 'Budget'
                'pool_size': 'pool_size',
                'judge_type': 'judge_type',
                'reward_type': 'reward_type',
                'strategy': 'strategy',
                'alpha': 'alpha',
                'lambda_reg': 'lambda_reg'
            }
            
            # Get the actual field name from the mapping, or use the key directly if not mapped
            actual_field = field_mapping.get(key, key)
            
            # Handle special cases where there might be multiple possible field names
            if actual_field == 'judge_backbone':
                data_value = data.get('judge_backbone', data.get('judge_model_name', ''))
            elif actual_field == 'budget':
                data_value = data.get('budget', data.get('Budget', ''))
            else:
                data_value = data.get(actual_field, '')
            
            data_value_str = str(data_value).lower()
            
            # Check if any of the exclude values match
            for value in values:
                value_lower = value.lower()
                if data_value_str == value_lower or value_lower in data_value_str:
                    return None
    
    return data


def parse_exclude_filters(exclude_string: str) -> Dict[str, List[str]]:
    """Parse exclude filters from 'key1=value1,key2=value2' format.
    
    Supports multiple values for the same key:
    'judge_model=model1,judge_model=model2' -> {'judge_model': ['model1', 'model2']}
    """
    if not exclude_string:
        return {}
    
    filters = {}
    pairs = exclude_string.split(',')
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)  # Split only on first =
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
    """Extract metadata for grouping from loaded data."""
    metadata = {}
    
    # Strategy information
    strategy = data.get('strategy', 'unknown')
    if 'simple_rewrite' in strategy.lower():
        template_name = data.get('template_name', '')
        if template_name:
            strategy = f"simple_rewrite_{template_name}"
    elif strategy == 'UCB-Warmup':
        # Skip warmup files
        return None
    metadata['strategy'] = strategy
    
    # Model information
    metadata['judge_model'] = data.get('judge_backbone', data.get('judge_model_name', 'unknown'))
    metadata['llm_agent'] = data.get('llm_agent_name', 'unknown')
    metadata['response_model'] = data.get('response_model_name', 'unknown')
    metadata['baseline_model'] = data.get('baseline_response_model_name', 'unknown')
    
    # Dataset information
    metadata['dataset'] = data.get('dataset_name', 'unknown')
    
    # Configuration information
    metadata['budget'] = str(data.get('budget', data.get('Budget', 'unknown')))
    metadata['pool_size'] = str(data.get('pool_size', 'unknown'))
    metadata['judge_type'] = data.get('judge_type', 'unknown')
    metadata['reward_type'] = data.get('reward_type', 'unknown')
    
    # UCB-specific parameters
    if 'alpha' in data:
        metadata['alpha'] = str(data.get('alpha', 'unknown'))
    if 'lambda_reg' in data:
        metadata['lambda_reg'] = str(data.get('lambda_reg', 'unknown'))
    
    # Time information
    metadata['timestamp'] = data.get('timestamp', 'unknown')
    
    return metadata

def extract_metrics_from_data(data: Dict[str, Any]) -> Dict[str, List[float]]:
    """Extract and calculate metrics from loaded data."""
    metrics = {}
    
    # Handle different data structures
    if 'metrics' in data and isinstance(data['metrics'], list):
        # Multiple questions - aggregate metrics
        all_metrics = data['metrics']
        if all_metrics:
            # Extract basic recorded metrics (aggregate across questions)
            basic_metrics = ['best_so_far', 'pool_mean', 'replacement_ratio', 'lift_per_1k_tokens', 'ci_width', 'ucb_gap']
            
            for metric_name in basic_metrics:
                if metric_name in all_metrics[0]:
                    # Aggregate across questions by taking mean at each round
                    max_rounds = max(len(m.get(metric_name, [])) for m in all_metrics)
                    aggregated = []
                    for round_idx in range(max_rounds):
                        round_values = []
                        for question_metrics in all_metrics:
                            if metric_name in question_metrics and round_idx < len(question_metrics[metric_name]):
                                round_values.append(question_metrics[metric_name][round_idx])
                        if round_values:
                            aggregated.append(np.mean(round_values))
                    if aggregated:
                        metrics[metric_name] = aggregated
            
            # Calculate stability (coefficient of variation across questions)
            if len(all_metrics) > 1:
                # Final scores for stability calculation
                final_scores = []
                for question_metrics in all_metrics:
                    if 'best_so_far' in question_metrics and question_metrics['best_so_far']:
                        final_scores.append(question_metrics['best_so_far'][-1])
                
                if len(final_scores) > 1:
                    # Stability (coefficient of variation) - single value, not per-round
                    cv = np.std(final_scores) / np.mean(final_scores) if np.mean(final_scores) != 0 else 0
                    # Make it a constant series for plotting compatibility
                    if 'best_so_far' in metrics:
                        metrics['stability'] = [cv] * len(metrics['best_so_far'])
                    else:
                        metrics['stability'] = [cv]
    else:
        # Legacy format or single aggregated metrics
        for key, value in data.items():
            if isinstance(value, list) and key in ['best_so_far', 'pool_mean', 'ci_width', 'ucb_gap', 'replacement_ratio', 'lift_per_1k_tokens']:
                metrics[key] = value
        
        # Calculate stability for legacy format
        if 'best_so_far' in metrics:
            # For single run, stability is 0
            metrics['stability'] = [0.0] * len(metrics['best_so_far'])
    
    return metrics

def find_all_exploration_files(metrics_dir: str) -> List[str]:
    """Find all exploration metrics files."""
    patterns = [
        os.path.join(metrics_dir, "**/*.json"),
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    
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



def plot_grouped_performance_comparison(grouped_data: Dict[Tuple, List[Dict]], group_by: List[str], report_metric: str = "best_so_far", save_dir: str = None):
    """Plot performance comparison grouped by specified dimensions."""
    
    # Determine subplot layout
    n_groups = len(grouped_data)
    if n_groups == 0:
        print("No data to plot")
        return
    
    # Calculate subplot grid
    n_cols = min(4, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Ensure axes is always a 2D array for consistent indexing
    if n_groups == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Color-blind friendly styling (consistent with other visualization files)
    strategy_styles = {
        'ucb': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
        'ucb_with_warmup': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-'},
        'random': {'color': '#2ca02c', 'marker': '^', 'linestyle': '--'},
        'simple_rewrite_holistic': {'color': '#d62728', 'marker': 'D', 'linestyle': '-.'},
        'simple_rewrite_improve': {'color': '#9467bd', 'marker': 'v', 'linestyle': ':'},
        'holistic_rewrite': {'color': '#d62728', 'marker': 'D', 'linestyle': '-.'},
        'improve': {'color': '#9467bd', 'marker': 'v', 'linestyle': ':'},
        # Fallback styles for other dimensions
        'relative': {'color': '#17becf', 'marker': 'o', 'linestyle': '-'},
        'absolute': {'color': '#bcbd22', 'marker': 's', 'linestyle': '-'},
        'pointwise': {'color': '#e377c2', 'marker': 'o', 'linestyle': '-'},
        'pairwise': {'color': '#7f7f7f', 'marker': 's', 'linestyle': '-'}
    }
    
    def get_style(key: str) -> dict:
        """Get color-blind friendly style for a given key."""
        return strategy_styles.get(key, {'color': '#8c564b', 'marker': 'p', 'linestyle': '-'})
    
    for idx, (group_key, group_items) in enumerate(grouped_data.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Group title
        group_title = ' | '.join([f"{dim}: {val}" for dim, val in zip(group_by, group_key)])
        ax.set_title(group_title, fontsize=10, pad=20)
        
        # Extract strategies within this group
        strategies_in_group = defaultdict(list)
        for item in group_items:
            strategy = item['metadata']['strategy']
            strategies_in_group[strategy].append(item['metrics'])
        
        # Plot each strategy
        for strategy, metrics_list in strategies_in_group.items():
            if not metrics_list:
                continue
                
            style = get_style(strategy)
            
            # Aggregate specified metric across runs
            all_metric_values = []
            for metrics in metrics_list:
                if report_metric in metrics and metrics[report_metric]:
                    all_metric_values.append(metrics[report_metric])
            
            if all_metric_values:
                max_rounds = max(len(run) for run in all_metric_values)
                mean_values = []
                std_values = []
                
                for round_idx in range(max_rounds):
                    round_values = [run[round_idx] for run in all_metric_values if round_idx < len(run)]
                    if round_values:
                        mean_values.append(np.mean(round_values))
                        std_values.append(np.std(round_values))
                
                rounds = list(range(1, len(mean_values) + 1))
                ax.plot(rounds, mean_values, 
                       marker=style['marker'], 
                       color=style['color'], 
                       linestyle=style['linestyle'],
                       label=f'{strategy} (n={len(all_metric_values)})', 
                       linewidth=2, markersize=6)
                
                if len(all_metric_values) > 1:
                    ax.fill_between(rounds, 
                                   np.array(mean_values) - np.array(std_values), 
                                   np.array(mean_values) + np.array(std_values), 
                                   color=style['color'], alpha=0.15)
        
        ax.set_xlabel('Exploration Round')
        
        # Set appropriate y-axis labels for different metrics
        if report_metric == 'stability':
            ax.set_ylabel('Stability (CV)')
        else:
            ax.set_ylabel(report_metric.replace('_', ' ').title())
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide empty subplots
    for idx in range(n_groups, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Set appropriate title for different metrics
    if report_metric == 'stability':
        metric_title = 'Stability (CV)'
    else:
        metric_title = report_metric.replace("_", " ").title()
    
    fig.suptitle(f'{metric_title} Comparison Grouped by {" | ".join(group_by)}', fontsize=14)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        group_name = "_".join(group_by)
        plt.savefig(os.path.join(save_dir, f'grouped_performance_{group_name}_{report_metric}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'grouped_performance_{group_name}_{report_metric}.pdf'), bbox_inches='tight')
    
    plt.show()

def plot_heatmap_comparison(grouped_data: Dict[Tuple, List[Dict]], group_by: List[str], report_metric: str = "best_so_far", save_dir: str = None):
    """Create heatmap showing final performance across different dimensions."""
    
    if len(group_by) != 2:
        print("Heatmap requires exactly 2 grouping dimensions")
        return
    
    # Extract all unique values for each dimension
    dim1_values = set()
    dim2_values = set()
    strategy_values = set()
    
    for group_key, group_items in grouped_data.items():
        dim1_values.add(group_key[0])
        dim2_values.add(group_key[1])
        for item in group_items:
            strategy_values.add(item['metadata']['strategy'])
    
    dim1_values = sorted(list(dim1_values))
    dim2_values = sorted(list(dim2_values))
    strategy_values = sorted(list(strategy_values))
    
    # Create subplots for each strategy
    n_strategies = len(strategy_values)
    fig, axes = plt.subplots(1, n_strategies, figsize=(6*n_strategies, 5))
    if n_strategies == 1:
        axes = [axes]
    
    for strategy_idx, strategy in enumerate(strategy_values):
        ax = axes[strategy_idx]
        
        # Create heatmap matrix
        heatmap_data = np.full((len(dim2_values), len(dim1_values)), np.nan)
        
        for group_key, group_items in grouped_data.items():
            dim1_idx = dim1_values.index(group_key[0])
            dim2_idx = dim2_values.index(group_key[1])
            
            # Find items for this strategy
            strategy_items = [item for item in group_items if item['metadata']['strategy'] == strategy]
            
            if strategy_items:
                # Calculate mean final performance
                final_values = []
                for item in strategy_items:
                    metrics = item['metrics']
                    if report_metric in metrics and metrics[report_metric]:
                        final_values.append(metrics[report_metric][-1])
                
                if final_values:
                    heatmap_data[dim2_idx, dim1_idx] = np.mean(final_values)
        
        # Plot heatmap with color-blind friendly colormap
        im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')  # viridis is color-blind friendly
        ax.set_title(f'{strategy.upper()}', fontsize=12)
        ax.set_xlabel(group_by[0].replace('_', ' ').title())
        ax.set_ylabel(group_by[1].replace('_', ' ').title())
        
        # Set ticks and labels
        ax.set_xticks(range(len(dim1_values)))
        ax.set_xticklabels(dim1_values, rotation=45, ha='right')
        ax.set_yticks(range(len(dim2_values)))
        ax.set_yticklabels(dim2_values)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Final Best Score')
        
        # Add text annotations
        for i in range(len(dim2_values)):
            for j in range(len(dim1_values)):
                if not np.isnan(heatmap_data[i, j]):
                    ax.text(j, i, f'{heatmap_data[i, j]:.3f}', 
                           ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    
    # Set appropriate title for different metrics
    if report_metric == 'stability':
        metric_title = 'Stability (CV)'
    else:
        metric_title = report_metric.replace("_", " ").title()
    
    plt.suptitle(f'{metric_title} Heatmap by {" vs ".join(group_by)}', fontsize=16)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        group_name = "_".join(group_by)
        plt.savefig(os.path.join(save_dir, f'heatmap_{group_name}_{report_metric}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'heatmap_{group_name}_{report_metric}.pdf'), bbox_inches='tight')
    
    plt.show()

def plot_summary_table_by_groups(grouped_data: Dict[Tuple, List[Dict]], group_by: List[str], report_metric: str = "best_so_far"):
    """Create summary table showing performance across groups."""
    
    # Collect data for table
    table_data = []
    
    for group_key, group_items in grouped_data.items():
        group_dict = {dim: val for dim, val in zip(group_by, group_key)}
        
        # Group by strategy within this group
        strategies_in_group = defaultdict(list)
        for item in group_items:
            strategy = item['metadata']['strategy']
            strategies_in_group[strategy].append(item['metrics'])
        
        for strategy, metrics_list in strategies_in_group.items():
            if not metrics_list:
                continue
            
            # Calculate statistics for specified metric
            final_values = []
            
            for metrics in metrics_list:
                if report_metric in metrics and metrics[report_metric]:
                    final_values.append(metrics[report_metric][-1])
            
            row = group_dict.copy()
            row.update({
                'strategy': strategy,
                'runs': len(metrics_list),
                f'mean_{report_metric}': np.mean(final_values) if final_values else np.nan,
                f'std_{report_metric}': np.std(final_values) if final_values else np.nan,
                f'max_{report_metric}': np.max(final_values) if final_values else np.nan
            })
            table_data.append(row)
    
    # Create DataFrame and display
    df = pd.DataFrame(table_data)
    
    if df.empty:
        print("No data available for summary table")
        return df
    
    # Sort by group dimensions then by mean report metric
    sort_cols = group_by + [f'mean_{report_metric}']
    df = df.sort_values(sort_cols, ascending=[True]*len(group_by) + [False])
    
    print(f"\n{'='*120}")
    print(f"PERFORMANCE SUMMARY GROUPED BY: {', '.join(group_by).upper()}")
    print(f"{'='*120}")
    
    # Display with better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(df.to_string(index=False))
    print(f"{'='*120}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Multi-dimensional exploration strategy comparison")
    parser.add_argument("--metrics_dir", type=str, help="Directory containing metrics files", default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/metrics")
    parser.add_argument("--group_by", type=str, default="strategy", 
                       help="Comma-separated list of dimensions to group by. Options: strategy, dataset, judge_model, llm_agent, response_model, budget, pool_size, alpha, lambda_reg, reward_type")
    parser.add_argument("--save_dir", type=str, help="Directory to save plots", default="./plots")
    parser.add_argument("--show_table", action="store_true", help="Show summary table")
    parser.add_argument("--plot_type", type=str, default="line", choices=["line", "heatmap", "both"],
                       help="Type of plot to generate")
    parser.add_argument("--filter", type=str, help="Filter data (e.g., 'dataset=AlpacaEval,strategy=ucb')")
    parser.add_argument("--exclude", type=str, help="Exclude files matching criteria (format: 'key1=value1,key2=value2')")
    parser.add_argument("--report_metric", type=str, default="best_so_far", 
                       help="Metric to visualize. Options: best_so_far, pool_mean, replacement_ratio, lift_per_1k_tokens, ci_width, ucb_gap, stability")
    
    args = parser.parse_args()
    
    # Parse exclude filters
    exclude_filters = parse_exclude_filters(args.exclude)
    if exclude_filters:
        print(f"Excluding files with: {exclude_filters}")
    
    # Parse grouping dimensions
    group_by = [dim.strip() for dim in args.group_by.split(',')]
    
    # Find all metrics files
    files = find_all_exploration_files(args.metrics_dir)
    if not files:
        print(f"No metrics files found in {args.metrics_dir}")
        return
    
    print(f"Found {len(files)} metrics files")
    
    # Load all data
    all_data = []
    for file_path in files:
        try:
            data = load_exploration_metrics(file_path, exclude_filters)
            
            # Skip warmup files or filtered files
            if data is None:
                print(f"Skipping file: {os.path.basename(file_path)}")
                continue
            
            if 'warmup_summary' in data:
                print(f"Skipping warmup file: {os.path.basename(file_path)}")
                continue
            
            # Skip files that are clearly warmup/init related (but not ucb_with_warmup)
            filename = os.path.basename(file_path).lower()
            if any(skip_term in filename for skip_term in ['init_ucb_warmup', 'init_linucb_warmup']) or \
               (('warmup' in filename or 'init_ucb' in filename or 'init_linucb' in filename) and 'ucb_with_warmup' not in filename):
                print(f"Skipping init/warmup file: {os.path.basename(file_path)}")
                continue
            
            metadata = extract_metadata_from_data(data)
            if metadata is None:  # Skip if metadata extraction failed (e.g., warmup files)
                print(f"Skipping file with invalid metadata: {os.path.basename(file_path)}")
                continue
                
            metrics = extract_metrics_from_data(data)
            
            all_data.append({
                'file_path': file_path,
                'metadata': metadata,
                'metrics': metrics
            })
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        print("No valid data loaded")
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
        print(f"  {group_label}: {len(group_items)} files")
    
    # Show summary table
    if args.show_table:
        summary_df = plot_summary_table_by_groups(grouped_data, group_by, args.report_metric)
    
    # Generate plots
    if args.plot_type in ["line", "both"]:
        print(f"\nGenerating line plots for {args.report_metric} grouped by {group_by}...")
        plot_grouped_performance_comparison(grouped_data, group_by, args.report_metric, args.save_dir)
    
    if args.plot_type in ["heatmap", "both"] and len(group_by) == 2:
        print(f"\nGenerating heatmap for {args.report_metric} grouped by {group_by}...")
        plot_heatmap_comparison(grouped_data, group_by, args.report_metric, args.save_dir)
    elif args.plot_type in ["heatmap", "both"] and len(group_by) != 2:
        print("Heatmap requires exactly 2 grouping dimensions. Skipping heatmap.")
    
    print("Done!")

if __name__ == "__main__":
    main()
