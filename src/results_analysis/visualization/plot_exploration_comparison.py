#!/usr/bin/env python3
"""
Plot and compare different exploration strategies (UCB vs Random vs Baselines).

Usage:
    python plot_exploration_comparison.py --metrics_dir path/to/metrics/directory
    python plot_exploration_comparison.py --metrics_files file1.json file2.json file3.json
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
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Strategy-specific styling for accessibility (color + marker combinations)
STRATEGY_STYLES = {
    'ucb': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
    'ucb_with_warmup': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-'},
    'random': {'color': '#2ca02c', 'marker': '^', 'linestyle': '--'},
    'holistic_rewrite': {'color': '#d62728', 'marker': 'D', 'linestyle': '-.'},
    'improve': {'color': '#9467bd', 'marker': 'v', 'linestyle': ':'},
    'simple_rewrite_holistic': {'color': '#d62728', 'marker': 'D', 'linestyle': '-.'},
    'simple_rewrite_improve': {'color': '#9467bd', 'marker': 'v', 'linestyle': ':'},
}

def get_strategy_style(strategy):
    """Get consistent color, marker, and linestyle for a strategy."""
    strategy_lower = strategy.lower()
    return STRATEGY_STYLES.get(strategy_lower, {
        'color': '#8c564b', 
        'marker': 'p', 
        'linestyle': '-'
    })


def load_exploration_metrics(file_path: str, exclude_filters: Dict[str, List[str]] = None) -> Dict[str, Any]:
    """Load exploration metrics from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Skip warmup/init files - we only want exploration results
    if 'warmup_summary' in data:
        return None
    
    # Skip files that are clearly warmup/init related
    filename = os.path.basename(file_path).lower()
    if any(skip_term in filename for skip_term in ['warmup', 'init_ucb', 'init_linucb']):
        return None
    
    # Apply exclude filters if provided
    if exclude_filters:
        for key, values in exclude_filters.items():
            data_value = data.get(key, '')
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


def find_exploration_files(metrics_dir: str) -> Dict[str, List[str]]:
    """Find and categorize exploration metrics files by strategy."""
    patterns = {
        'ucb_with_warmup': [
            os.path.join(metrics_dir, "**/ucb_with_warmup*.json"),
            os.path.join(metrics_dir, "**/*ucb_with_warmup*.json"),
            os.path.join(metrics_dir, "**/UCB_WITH_WARMUP*.json")
        ],
        'ucb': [
            # Use broad patterns, will filter out ucb_with_warmup files later
            os.path.join(metrics_dir, "**/ucb*.json"),
            os.path.join(metrics_dir, "**/*ucb*.json"),
            os.path.join(metrics_dir, "**/UCB*.json")
        ],
        'random': [
            os.path.join(metrics_dir, "**/random*.json"),
            os.path.join(metrics_dir, "**/*random*.json")
        ],
        'holistic_rewrite': [
            os.path.join(metrics_dir, "**/holistic*.json"),
            os.path.join(metrics_dir, "**/*holistic*.json"),
        ],
        "improve": [
            os.path.join(metrics_dir, "**/improve*.json"),
            os.path.join(metrics_dir, "**/*improve*.json"),
        ]
    }
    
    files_by_strategy = {}
    for strategy, patterns_list in patterns.items():
        files = []
        for pattern in patterns_list:
            files.extend(glob.glob(pattern, recursive=True))
        files_by_strategy[strategy] = sorted(list(set(files)))
    
    # Post-process to fix overlapping patterns
    # Order matters: process specific patterns first, then general ones
    
    # 1. Remove ucb_with_warmup files from regular ucb category
    if 'ucb' in files_by_strategy and 'ucb_with_warmup' in files_by_strategy:
        original_ucb_count = len(files_by_strategy['ucb'])
        
        # Filter: keep only files that don't contain 'with_warmup' anywhere in their path
        files_by_strategy['ucb'] = [
            f for f in files_by_strategy['ucb'] 
            if 'with_warmup' not in f.lower()
        ]
        
        removed_count = original_ucb_count - len(files_by_strategy['ucb'])
        if removed_count > 0:
            print(f"Filtered out {removed_count} files containing 'with_warmup' from ucb category")
    
    # 2. Similarly handle any other overlapping patterns in the future
    # (e.g., if we add ucb_without_warmup, etc.)
    
    return files_by_strategy


def extract_metrics_from_data(data: Dict[str, Any]) -> Dict[str, List[float]]:
    """Extract relevant metrics from loaded data."""
    metrics = {}
    
    # Handle different data structures
    if 'metrics' in data and isinstance(data['metrics'], list):
        # Multiple questions - aggregate metrics
        all_metrics = data['metrics']
        if all_metrics:
            # Get all metric keys from first question
            sample_metrics = all_metrics[0]
            for key in sample_metrics:
                if isinstance(sample_metrics[key], list):
                    # Aggregate across questions (take mean at each round)
                    max_rounds = max(len(m.get(key, [])) for m in all_metrics)
                    aggregated = []
                    for round_idx in range(max_rounds):
                        round_values = [m.get(key, [])[round_idx] 
                                      for m in all_metrics 
                                      if round_idx < len(m.get(key, []))]
                        if round_values:
                            aggregated.append(np.mean(round_values))
                    metrics[key] = aggregated
    else:
        # Single question or already aggregated
        for key, value in data.items():
            if isinstance(value, list) and key in ['best_so_far', 'pool_mean', 'ci_width', 'ucb_gap', 'replacement_ratio', 'lift_per_1k_tokens']:
                metrics[key] = value
    
    return metrics


def plot_performance_comparison(metrics_by_strategy: Dict[str, List[Dict]], save_dir: str = None):
    """Plot performance comparison across strategies."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    for strategy, metrics_list in metrics_by_strategy.items():
        if not metrics_list:
            continue
        
        style = get_strategy_style(strategy)
        color = style['color']
        marker = style['marker']
        linestyle = style['linestyle']
        
        # Aggregate metrics across multiple runs
        all_best_so_far = []
        all_pool_mean = []
        all_replacement_ratio = []
        all_lift_per_1k_tokens = []
        
        for metrics in metrics_list:
            if 'best_so_far' in metrics:
                all_best_so_far.append(metrics['best_so_far'])
            if 'pool_mean' in metrics:
                all_pool_mean.append(metrics['pool_mean'])
            if 'replacement_ratio' in metrics:
                all_replacement_ratio.append(metrics['replacement_ratio'])
            if 'lift_per_1k_tokens' in metrics:
                all_lift_per_1k_tokens.append(metrics['lift_per_1k_tokens'])
        
        # Plot best so far
        if all_best_so_far:
            max_rounds = max(len(run) for run in all_best_so_far)
            mean_best = []
            std_best = []
            for round_idx in range(max_rounds):
                round_values = [run[round_idx] for run in all_best_so_far if round_idx < len(run)]
                if round_values:
                    mean_best.append(np.mean(round_values))
                    std_best.append(np.std(round_values))
            
            rounds = list(range(1, len(mean_best) + 1))
            ax1.plot(rounds, mean_best, marker=marker, linestyle=linestyle, color=color, 
                    label=f'{strategy.upper()} (n={len(all_best_so_far)})', linewidth=2, markersize=6)
            if len(all_best_so_far) > 1:
                ax1.fill_between(rounds, 
                               np.array(mean_best) - np.array(std_best), 
                               np.array(mean_best) + np.array(std_best), 
                               color=color, alpha=0.2)
        
        # Plot pool mean
        if all_pool_mean:
            max_rounds = max(len(run) for run in all_pool_mean)
            mean_pool = []
            for round_idx in range(max_rounds):
                round_values = [run[round_idx] for run in all_pool_mean if round_idx < len(run)]
                if round_values:
                    mean_pool.append(np.mean(round_values))
            
            rounds = list(range(1, len(mean_pool) + 1))
            ax2.plot(rounds, mean_pool, marker=marker, linestyle=linestyle, color=color, 
                    label=f'{strategy.upper()}', linewidth=2, markersize=6)
        
        # Plot replacement ratio
        if all_replacement_ratio:
            max_rounds = max(len(run) for run in all_replacement_ratio)
            mean_replacement = []
            for round_idx in range(max_rounds):
                round_values = [run[round_idx] for run in all_replacement_ratio if round_idx < len(run)]
                if round_values:
                    mean_replacement.append(np.mean(round_values))
            
            rounds = list(range(1, len(mean_replacement) + 1))
            ax3.plot(rounds, mean_replacement, marker=marker, linestyle=linestyle, color=color, 
                    label=f'{strategy.upper()}', linewidth=2, markersize=6)
        
        # Plot lift per 1k tokens
        if all_lift_per_1k_tokens:
            max_rounds = max(len(run) for run in all_lift_per_1k_tokens)
            mean_lift = []
            for round_idx in range(max_rounds):
                round_values = [run[round_idx] for run in all_lift_per_1k_tokens if round_idx < len(run)]
                if round_values:
                    mean_lift.append(np.mean(round_values))
            
            rounds = list(range(1, len(mean_lift) + 1))
            ax4.plot(rounds, mean_lift, marker=marker, linestyle=linestyle, color=color, 
                    label=f'{strategy.upper()}', linewidth=2, markersize=6)
    
    # Configure plots
    ax1.set_xlabel('Exploration Round')
    ax1.set_ylabel('Best Score So Far')
    ax1.set_title('Best Performance Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Exploration Round')
    ax2.set_ylabel('Pool Mean Score')
    ax2.set_title('Pool Quality Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3.set_xlabel('Exploration Round')
    ax3.set_ylabel('Replacement Ratio')
    ax3.set_title('Pool Replacement Efficiency')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4.set_xlabel('Exploration Round')
    ax4.set_ylabel('Lift per 1K Tokens')
    ax4.set_title('Token Efficiency')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'exploration_performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'exploration_performance_comparison.pdf'), bbox_inches='tight')
    
    plt.show()


def plot_ucb_specific_metrics(ucb_metrics_list: List[Dict], save_dir: str = None):
    """Plot UCB-specific metrics (CI width and UCB gap)."""
    if not ucb_metrics_list:
        print("No UCB metrics found for UCB-specific plotting")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(ucb_metrics_list)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']  # Different markers for different runs
    
    for i, metrics in enumerate(ucb_metrics_list):
        ci_width = metrics.get('ci_width', [])
        ucb_gap = metrics.get('ucb_gap', [])
        
        label = f'UCB Run {i+1}'
        marker = markers[i % len(markers)]  # Cycle through markers
        
        # Plot CI width
        if ci_width:
            rounds = list(range(1, len(ci_width) + 1))
            ax1.plot(rounds, ci_width, marker=marker, linestyle='-', color=colors[i], 
                    label=label, alpha=0.8, linewidth=2, markersize=6)
        
        # Plot UCB gap
        if ucb_gap:
            rounds = list(range(1, len(ucb_gap) + 1))
            ax2.plot(rounds, ucb_gap, marker=marker, linestyle='-', color=colors[i], 
                    label=label, alpha=0.8, linewidth=2, markersize=6)
    
    # Configure CI width plot
    ax1.set_xlabel('Exploration Round')
    ax1.set_ylabel('CI Width (2α√(x^T A^{-1} x))')
    ax1.set_title('UCB Confidence Interval Width')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Configure UCB gap plot
    ax2.set_xlabel('Exploration Round')
    ax2.set_ylabel('UCB Gap (Best - 2nd Best)')
    ax2.set_title('UCB Gap Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'ucb_specific_metrics.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'ucb_specific_metrics.pdf'), bbox_inches='tight')
    
    plt.show()


def plot_final_performance_comparison(metrics_by_strategy: Dict[str, List[Dict]], save_dir: str = None):
    """Plot final performance comparison with box plots."""
    strategies = []
    final_scores = []
    final_pool_means = []
    token_efficiencies = []
    
    for strategy, metrics_list in metrics_by_strategy.items():
        if not metrics_list:
            continue
        
        strategy_final_scores = []
        strategy_pool_means = []
        strategy_token_eff = []
        
        for metrics in metrics_list:
            # Final best score
            if 'best_so_far' in metrics and metrics['best_so_far']:
                strategy_final_scores.append(metrics['best_so_far'][-1])
            
            # Final pool mean
            if 'pool_mean' in metrics and metrics['pool_mean']:
                strategy_pool_means.append(metrics['pool_mean'][-1])
            
            # Final token efficiency
            if 'lift_per_1k_tokens' in metrics and metrics['lift_per_1k_tokens']:
                strategy_token_eff.append(metrics['lift_per_1k_tokens'][-1])
        
        if strategy_final_scores:
            strategies.extend([strategy.upper()] * len(strategy_final_scores))
            final_scores.extend(strategy_final_scores)
            final_pool_means.extend(strategy_pool_means)
            token_efficiencies.extend(strategy_token_eff)
    
    if not strategies:
        print("No final performance data found")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Box plot for final scores
    unique_strategies = list(set(strategies))
    score_data = [final_scores[i] for i, s in enumerate(strategies)]
    
    scores_by_strategy = {}
    pool_means_by_strategy = {}
    token_eff_by_strategy = {}
    
    for i, strategy in enumerate(strategies):
        if strategy not in scores_by_strategy:
            scores_by_strategy[strategy] = []
            pool_means_by_strategy[strategy] = []
            token_eff_by_strategy[strategy] = []
        
        scores_by_strategy[strategy].append(final_scores[i])
        if i < len(final_pool_means):
            pool_means_by_strategy[strategy].append(final_pool_means[i])
        if i < len(token_efficiencies):
            token_eff_by_strategy[strategy].append(token_efficiencies[i])
    
    # Plot box plots - only for strategies that have data
    strategies_ordered = ['UCB', 'UCB_WITH_WARMUP', 'RANDOM', 'HOLISTIC_REWRITE', 'IMPROVE']
    
    # Hatching patterns for accessibility (color blindness)
    hatching_patterns = ['', '///', '...', '+++', 'xxx', '|||', 'ooo']
    
    # Filter to only strategies with data
    available_strategies = []
    strategy_positions = []
    strategy_styles = []
    
    for strategy in strategies_ordered:
        if strategy in scores_by_strategy:
            available_strategies.append(strategy)
            strategy_positions.append(len(available_strategies) - 1)
            style = get_strategy_style(strategy.lower())
            strategy_styles.append(style)
    
    # Plot box plots with correct positions
    for i, strategy in enumerate(available_strategies):
        pos = strategy_positions[i]
        style = strategy_styles[i]
        color = style['color']
        hatch = hatching_patterns[i % len(hatching_patterns)]
        
        # Plot scores
        if strategy in scores_by_strategy:
            bp1 = ax1.boxplot(scores_by_strategy[strategy], positions=[pos], widths=0.6, 
                             patch_artist=True, boxprops=dict(facecolor=color, alpha=0.7, hatch=hatch))
        
        # Plot pool means
        if strategy in pool_means_by_strategy:
            bp2 = ax2.boxplot(pool_means_by_strategy[strategy], positions=[pos], widths=0.6,
                             patch_artist=True, boxprops=dict(facecolor=color, alpha=0.7, hatch=hatch))
        
        # Plot token efficiency
        if strategy in token_eff_by_strategy:
            bp3 = ax3.boxplot(token_eff_by_strategy[strategy], positions=[pos], widths=0.6,
                             patch_artist=True, boxprops=dict(facecolor=color, alpha=0.7, hatch=hatch))
    
    # Configure plots with matching ticks and labels
    if available_strategies:
        ax1.set_xticks(range(len(available_strategies)))
        ax1.set_xticklabels(available_strategies)
        ax1.set_ylabel('Final Best Score')
        ax1.set_title('Final Performance Comparison')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xticks(range(len(available_strategies)))
        ax2.set_xticklabels(available_strategies)
        ax2.set_ylabel('Final Pool Mean')
        ax2.set_title('Final Pool Quality')
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xticks(range(len(available_strategies)))
        ax3.set_xticklabels(available_strategies)
        ax3.set_ylabel('Final Lift per 1K Tokens')
        ax3.set_title('Token Efficiency')
        ax3.grid(True, alpha=0.3)
    else:
        # Handle case with no data
        for ax in [ax1, ax2, ax3]:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'final_performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'final_performance_comparison.pdf'), bbox_inches='tight')
    
    plt.show()


def create_comparison_summary_table(metrics_by_strategy: Dict[str, List[Dict]]) -> str:
    """Create a summary table comparing strategies."""
    headers = ["Strategy", "Runs", "Avg Final Score", "Avg Pool Mean", "Avg Token Eff", "Best Score", "Win Rate"]
    rows = []
    
    all_final_scores = []
    strategy_data = {}
    
    for strategy, metrics_list in metrics_by_strategy.items():
        if not metrics_list:
            continue
        
        final_scores = []
        pool_means = []
        token_effs = []
        
        for metrics in metrics_list:
            if 'best_so_far' in metrics and metrics['best_so_far']:
                final_scores.append(metrics['best_so_far'][-1])
                all_final_scores.append(metrics['best_so_far'][-1])
            
            if 'pool_mean' in metrics and metrics['pool_mean']:
                pool_means.append(metrics['pool_mean'][-1])
            
            if 'lift_per_1k_tokens' in metrics and metrics['lift_per_1k_tokens']:
                token_effs.append(metrics['lift_per_1k_tokens'][-1])
        
        strategy_data[strategy] = {
            'runs': len(metrics_list),
            'final_scores': final_scores,
            'pool_means': pool_means,
            'token_effs': token_effs
        }
    
    # Calculate win rates (percentage of runs where strategy achieved best score)
    if all_final_scores:
        best_threshold = np.percentile(all_final_scores, 75)  # Top 25%
        
        for strategy, data in strategy_data.items():
            wins = sum(1 for score in data['final_scores'] if score >= best_threshold)
            win_rate = wins / len(data['final_scores']) * 100 if data['final_scores'] else 0
            
            row = [
                strategy.upper(),
                str(data['runs']),
                f"{np.mean(data['final_scores']):.3f}" if data['final_scores'] else "N/A",
                f"{np.mean(data['pool_means']):.3f}" if data['pool_means'] else "N/A",
                f"{np.mean(data['token_effs']):.3f}" if data['token_effs'] else "N/A",
                f"{max(data['final_scores']):.3f}" if data['final_scores'] else "N/A",
                f"{win_rate:.1f}%"
            ]
            rows.append(row)
    
    # Sort by average final score
    rows.sort(key=lambda x: float(x[2]) if x[2] != "N/A" else 0, reverse=True)
    
    # Create formatted table
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *rows)]
    
    def format_row(row):
        return " | ".join(str(item).ljust(width) for item, width in zip(row, col_widths))
    
    table = [
        format_row(headers),
        "-" * (sum(col_widths) + 3 * (len(headers) - 1)),
        *[format_row(row) for row in rows]
    ]
    
    return "\n".join(table)


def main():
    parser = argparse.ArgumentParser(description="Compare exploration strategies (UCB vs Random vs Baseline)")
    parser.add_argument("--metrics_dir", type=str, help="Directory containing metrics files")
    parser.add_argument("--metrics_files", nargs="+", help="Specific metrics files to compare")
    parser.add_argument("--save_dir", type=str, help="Directory to save plots")
    parser.add_argument("--show_table", action="store_true", help="Show comparison summary table")
    parser.add_argument("--ucb_only", action="store_true", help="Plot only UCB-specific metrics")
    parser.add_argument("--group_by_model", action="store_true", help="Group results by judge model")
    parser.add_argument("--group_by_dataset", action="store_true", help="Group results by dataset")
    parser.add_argument("--filter_model", type=str, help="Filter to specific judge model")
    parser.add_argument("--filter_dataset", type=str, help="Filter to specific dataset")
    parser.add_argument("--exclude", type=str, help="Exclude files matching criteria (format: 'key1=value1,key2=value2')")
    
    args = parser.parse_args()
    
    # Parse exclude filters
    exclude_filters = parse_exclude_filters(args.exclude)
    if exclude_filters:
        print(f"Excluding files with: {exclude_filters}")
    
    # Collect metrics files
    if args.metrics_files:
        # Manually specified files - try to categorize by filename
        files_by_strategy = {'ucb': [], 'ucb_with_warmup': [], 'random': [], 'holistic_rewrite': [], 'improve': []}
        for file_path in args.metrics_files:
            filename = os.path.basename(file_path).lower()
            # Check for specific patterns first, then more general ones
            if 'ucb_with_warmup' in filename:
                files_by_strategy['ucb_with_warmup'].append(file_path)
            elif 'ucb' in filename and 'ucb_with_warmup' not in filename:
                # Explicit check to avoid false positives
                files_by_strategy['ucb'].append(file_path)
            elif 'random' in filename:
                files_by_strategy['random'].append(file_path)
            elif 'holistic' in filename:
                files_by_strategy['holistic_rewrite'].append(file_path)
            elif 'improve' in filename:
                files_by_strategy['improve'].append(file_path)
            else:
                # Default to holistic_rewrite for unknown files
                files_by_strategy['holistic_rewrite'].append(file_path)
    elif args.metrics_dir:
        files_by_strategy = find_exploration_files(args.metrics_dir)
    else:
        print("Please provide either --metrics_dir or --metrics_files")
        return
    
    # Print found files
    total_files = sum(len(files) for files in files_by_strategy.values())
    print(f"Found {total_files} metrics files:")
    for strategy, files in files_by_strategy.items():
        print(f"  {strategy.upper()}: {len(files)} files")
        for f in files[:3]:  # Show first 3 files
            print(f"    - {os.path.basename(f)}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more")
    
    # Load all metrics
    metrics_by_strategy = {}
    for strategy, files in files_by_strategy.items():
        strategy_metrics = []
        for file_path in files:
            try:
                data = load_exploration_metrics(file_path, exclude_filters)
                if data is None:  # Skip warmup/init files or filtered files
                    print(f"Skipping file: {os.path.basename(file_path)}")
                    continue
                    
                metrics = extract_metrics_from_data(data)
                if metrics:  # Only add if we extracted some metrics
                    strategy_metrics.append(metrics)
                    print(f"Loaded {strategy}: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        metrics_by_strategy[strategy] = strategy_metrics
    
    if not any(metrics_by_strategy.values()):
        print("No valid metrics loaded")
        return
    
    # Show summary table
    if args.show_table:
        print("\n" + "="*100)
        print("EXPLORATION STRATEGY COMPARISON TABLE")
        print("="*100)
        print(create_comparison_summary_table(metrics_by_strategy))
        print("="*100)
    
    # Create plots
    if args.ucb_only:
        ucb_metrics = metrics_by_strategy.get('ucb', [])
        if ucb_metrics:
            print(f"\nPlotting UCB-specific metrics for {len(ucb_metrics)} runs...")
            plot_ucb_specific_metrics(ucb_metrics, args.save_dir)
        else:
            print("No UCB metrics found for UCB-specific plotting")
    else:
        print(f"\nPlotting exploration strategy comparison...")
        plot_performance_comparison(metrics_by_strategy, args.save_dir)
        plot_final_performance_comparison(metrics_by_strategy, args.save_dir)
        
        # Also plot UCB-specific if available
        ucb_metrics = metrics_by_strategy.get('ucb', [])
        if ucb_metrics:
            plot_ucb_specific_metrics(ucb_metrics, args.save_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()
