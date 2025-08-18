#!/usr/bin/env python3
"""
Plot LinUCB warmup process metrics from saved files.

Usage:
    python scripts/visualization/plot_warmup_metrics.py --metrics_file path/to/warmup_metrics.json
    python scripts/visualization/plot_warmup_metrics.py --metrics_dir path/to/metrics/directory
"""

import argparse
import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_warmup_metrics(file_path: str) -> Dict[str, Any]:
    """Load warmup metrics from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the warmup summary
    if 'warmup_summary' in data:
        return data['warmup_summary']
    else:
        return data


def find_warmup_files(metrics_dir: str) -> List[str]:
    """Find all warmup metrics files in a directory."""
    patterns = [
        os.path.join(metrics_dir, "**/init_ucb_warmup*.json"),
        os.path.join(metrics_dir, "**/warmup*.json"),
        os.path.join(metrics_dir, "**/*warmup*.json")
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(list(set(files)))


def plot_convergence_curves(metrics_list: List[Dict[str, Any]], save_dir: str = None):
    """Plot confidence interval and UCB gap convergence curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))
    
    for i, metrics in enumerate(metrics_list):
        median_ci = metrics.get('median_ci', [])
        median_gap = metrics.get('median_gap', [])  # warmup method
        ucb_gap = metrics.get('ucb_gap', [])        # explore methods
        
        # Use either warmup gap or explore gap data
        gap_data = median_gap if median_gap else ucb_gap
        rounds = list(range(1, len(median_ci) + 1))
        
        # Create label from metadata
        label_parts = []
        if 'alpha' in metrics:
            label_parts.append(f"α={metrics['alpha']}")
        if 'lambda_reg' in metrics:
            label_parts.append(f"λ={metrics['lambda_reg']}")
        if 'dataset_name' in metrics:
            label_parts.append(f"{metrics['dataset_name']}")
        
        label = ", ".join(label_parts) if label_parts else f"Run {i+1}"
        
        # Plot confidence interval width
        if median_ci:
            ax1.plot(rounds, median_ci, 'o-', color=colors[i], label=label, alpha=0.8, linewidth=2)
            
            # Add convergence threshold line if available
            if 'ci_width_threshold' in metrics:
                ax1.axhline(y=metrics['ci_width_threshold'], 
                           color=colors[i], linestyle='--', alpha=0.5, linewidth=1)
        
        # Plot UCB gap (if available)
        if gap_data:
            rounds_gap = list(range(1, len(gap_data) + 1))
            ax2.plot(rounds_gap, gap_data, 's-', color=colors[i], label=label, alpha=0.8, linewidth=2)
    
    # Configure CI plot
    ax1.set_xlabel('Warmup Round')
    ax1.set_ylabel('Median CI Width (2α√(x^T A^{-1} x))')
    ax1.set_title('Confidence Interval Width Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Configure UCB gap plot
    ax2.set_xlabel('Warmup Round')
    ax2.set_ylabel('Median UCB Gap (Best - 2nd Best)')
    ax2.set_title('UCB Gap Evolution')
    ax2.grid(True, alpha=0.3)
    
    # Only show legend and log scale if we have gap data
    gap_data_exists = any(metrics.get('median_gap') or metrics.get('ucb_gap') for metrics in metrics_list)
    if gap_data_exists:
        ax2.legend()
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'No UCB gap data available\n(Check warmup implementation)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'warmup_convergence.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'warmup_convergence.pdf'), bbox_inches='tight')
    
    plt.show()


def plot_warmup_phases(metrics: Dict[str, Any], save_dir: str = None):
    """Plot warmup phases with burnin vs UCB+ε phases highlighted."""
    median_ci = metrics.get('median_ci', [])
    median_gap = metrics.get('median_gap', [])  # warmup method
    ucb_gap = metrics.get('ucb_gap', [])        # explore methods
    
    # Use either warmup gap or explore gap data
    gap_data = median_gap if median_gap else ucb_gap
    
    if not median_ci:
        print("No CI data found for phase plotting")
        return
    
    rounds = list(range(1, len(median_ci) + 1))
    
    # Estimate phase boundaries
    burnin_passes = metrics.get('burnin_passes', 1)
    n_arms = metrics.get('n_arms', 8)
    burnin_rounds = burnin_passes * n_arms
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot CI with phases
    ax1.plot(rounds, median_ci, 'o-', color='steelblue', linewidth=2, markersize=4)
    
    # Highlight phases
    if burnin_rounds < len(rounds):
        ax1.axvspan(1, burnin_rounds, alpha=0.2, color='orange', label='Burn-in Phase')
        ax1.axvspan(burnin_rounds, len(rounds), alpha=0.2, color='green', label='UCB+ε Phase')
    
    # Add convergence threshold
    if 'ci_width_threshold' in metrics:
        ax1.axhline(y=metrics['ci_width_threshold'], 
                   color='red', linestyle='--', alpha=0.7, 
                   label=f'Threshold ({metrics["ci_width_threshold"]})')
    
    ax1.set_xlabel('Warmup Round')
    ax1.set_ylabel('Median CI Width')
    ax1.set_title('LinUCB Warmup: Confidence Interval Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Plot UCB gap with phases (if available)
    if gap_data:
        rounds_gap = list(range(1, len(gap_data) + 1))
        ax2.plot(rounds_gap, gap_data, 's-', color='coral', linewidth=2, markersize=4)
        
        # Highlight phases
        if burnin_rounds < len(rounds_gap):
            ax2.axvspan(1, min(burnin_rounds, len(rounds_gap)), alpha=0.2, color='orange')
            ax2.axvspan(burnin_rounds, len(rounds_gap), alpha=0.2, color='green')
        
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'No UCB gap data available\n(Check warmup implementation)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax2.set_xlabel('Warmup Round')
    ax2.set_ylabel('Median UCB Gap')
    ax2.set_title('LinUCB Warmup: UCB Gap Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'warmup_phases.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'warmup_phases.pdf'), bbox_inches='tight')
    
    plt.show()


def plot_hyperparameter_comparison(metrics_list: List[Dict[str, Any]], save_dir: str = None):
    """Compare different hyperparameter settings."""
    if len(metrics_list) < 2:
        print("Need at least 2 runs for comparison")
        return
    
    # Group by hyperparameters
    alpha_groups = {}
    lambda_groups = {}
    
    for metrics in metrics_list:
        alpha = metrics.get('alpha', 'Unknown')
        lambda_reg = metrics.get('lambda_reg', 'Unknown')
        
        if alpha not in alpha_groups:
            alpha_groups[alpha] = []
        alpha_groups[alpha].append(metrics)
        
        if lambda_reg not in lambda_groups:
            lambda_groups[lambda_reg] = []
        lambda_groups[lambda_reg].append(metrics)
    
    # Plot alpha comparison
    if len(alpha_groups) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_groups)))
        
        for i, (alpha, group_metrics) in enumerate(alpha_groups.items()):
            # Average the CI curves for this alpha
            max_rounds = max(len(m.get('median_ci', [])) for m in group_metrics)
            ci_matrix = np.full((len(group_metrics), max_rounds), np.nan)
            
            for j, metrics in enumerate(group_metrics):
                ci_data = metrics.get('median_ci', [])
                ci_matrix[j, :len(ci_data)] = ci_data
            
            # Compute mean and std
            mean_ci = np.nanmean(ci_matrix, axis=0)
            std_ci = np.nanstd(ci_matrix, axis=0)
            rounds = list(range(1, len(mean_ci) + 1))
            
            # Plot mean with confidence band
            ax1.plot(rounds, mean_ci, 'o-', color=colors[i], label=f'α = {alpha}', linewidth=2)
            ax1.fill_between(rounds, mean_ci - std_ci, mean_ci + std_ci, 
                           color=colors[i], alpha=0.2)
        
        ax1.set_xlabel('Warmup Round')
        ax1.set_ylabel('Median CI Width')
        ax1.set_title('Effect of α (Exploration Parameter)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_yscale('log')
        
        # Plot final CI values
        final_ci_by_alpha = {}
        for alpha, group_metrics in alpha_groups.items():
            final_cis = [m.get('median_ci', [0])[-1] if m.get('median_ci') else 0 
                        for m in group_metrics]
            final_ci_by_alpha[alpha] = final_cis
        
        alphas = list(final_ci_by_alpha.keys())
        positions = range(len(alphas))
        
        ax2.boxplot([final_ci_by_alpha[alpha] for alpha in alphas], positions=positions)
        ax2.set_xticklabels([f'α={alpha}' for alpha in alphas])
        ax2.set_ylabel('Final CI Width')
        ax2.set_title('Final Convergence by α')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'alpha_comparison.png'), dpi=300, bbox_inches='tight')
        
        plt.show()


def create_summary_table(metrics_list: List[Dict[str, Any]]) -> str:
    """Create a summary table of warmup runs."""
    if not metrics_list:
        return "No metrics data found."
    
    headers = ["Run", "Phase", "Rounds", "Final CI", "Final Gap", "α", "λ", "Dataset", "Time (s)"]
    rows = []
    
    for i, metrics in enumerate(metrics_list):
        median_ci = metrics.get('median_ci', [])
        median_gap = metrics.get('median_gap', [])  # warmup method
        ucb_gap = metrics.get('ucb_gap', [])        # explore methods
        
        # Use either warmup gap or explore gap data
        gap_data = median_gap if median_gap else ucb_gap
        
        final_ci = median_ci[-1] if median_ci else "N/A"
        final_gap = gap_data[-1] if gap_data else "N/A"
        
        row = [
            f"{i+1}",
            metrics.get('phase', 'Unknown'),
            str(metrics.get('rounds', 'N/A')),
            f"{final_ci:.4f}" if isinstance(final_ci, (int, float)) else final_ci,
            f"{final_gap:.4f}" if isinstance(final_gap, (int, float)) else final_gap,
            str(metrics.get('alpha', 'N/A')),
            str(metrics.get('lambda_reg', 'N/A')),
            metrics.get('dataset_name', 'N/A'),
            f"{metrics.get('time_taken', 0):.1f}"
        ]
        rows.append(row)
    
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
    parser = argparse.ArgumentParser(description="Plot LinUCB warmup metrics")
    parser.add_argument("--metrics_file", type=str, help="Path to a single metrics JSON file")
    parser.add_argument("--metrics_dir", type=str, help="Directory containing metrics files")
    parser.add_argument("--save_dir", type=str, help="Directory to save plots")
    parser.add_argument("--compare", action="store_true", help="Create comparison plots if multiple files")
    parser.add_argument("--show_table", action="store_true", help="Show summary table")
    
    args = parser.parse_args()
    
    # Collect metrics files
    if args.metrics_file:
        metrics_files = [args.metrics_file]
    elif args.metrics_dir:
        metrics_files = find_warmup_files(args.metrics_dir)
        if not metrics_files:
            print(f"No warmup metrics files found in {args.metrics_dir}")
            return
    else:
        print("Please provide either --metrics_file or --metrics_dir")
        return
    
    print(f"Found {len(metrics_files)} metrics file(s)")
    for f in metrics_files:
        print(f"  - {f}")
    
    # Load all metrics
    metrics_list = []
    for file_path in metrics_files:
        try:
            metrics = load_warmup_metrics(file_path)
            metrics_list.append(metrics)
            print(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not metrics_list:
        print("No valid metrics loaded")
        return
    
    # Show summary table
    if args.show_table:
        print("\n" + "="*80)
        print("WARMUP SUMMARY TABLE")
        print("="*80)
        print(create_summary_table(metrics_list))
        print("="*80)
    
    # Create plots
    if len(metrics_list) == 1:
        print(f"\nPlotting single warmup run...")
        plot_warmup_phases(metrics_list[0], args.save_dir)
    else:
        print(f"\nPlotting {len(metrics_list)} warmup runs...")
        plot_convergence_curves(metrics_list, args.save_dir)
        
        if args.compare:
            plot_hyperparameter_comparison(metrics_list, args.save_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()
