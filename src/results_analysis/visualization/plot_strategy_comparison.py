#!/usr/bin/env python3
"""
Simple comparison of exploration strategies with clean visualizations.

Usage:
    python plot_strategy_comparison.py --ucb_file ucb_metrics.json --random_file random_metrics.json
    python plot_strategy_comparison.py --metrics_dir /path/to/metrics/
"""

import argparse
import json
import os
import glob
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_metrics(file_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_trajectory_metrics(data: Dict[str, Any]) -> Dict[str, List[float]]:
    """Extract trajectory metrics from loaded data."""
    metrics = {}
    
    if 'metrics' in data and isinstance(data['metrics'], list):
        # Multiple questions - aggregate
        all_metrics = data['metrics']
        if all_metrics:
            # Average across questions for each round
            sample_metrics = all_metrics[0]
            for key in ['best_so_far', 'pool_mean', 'replacement_ratio', 'lift_per_1k_tokens', 'ci_width', 'ucb_gap']:
                if key in sample_metrics:
                    max_rounds = max(len(m.get(key, [])) for m in all_metrics)
                    aggregated = []
                    for round_idx in range(max_rounds):
                        round_values = [m.get(key, [])[round_idx] 
                                      for m in all_metrics 
                                      if round_idx < len(m.get(key, []))]
                        if round_values:
                            aggregated.append(np.mean(round_values))
                    metrics[key] = aggregated
    
    return metrics


def find_strategy_files(metrics_dir: str) -> Dict[str, str]:
    """Find strategy files automatically."""
    files = {}
    
    # Look for UCB files
    ucb_patterns = ["**/ucb*.json", "**/*UCB*.json"]
    for pattern in ucb_patterns:
        ucb_files = glob.glob(os.path.join(metrics_dir, pattern), recursive=True)
        if ucb_files:
            files['ucb'] = ucb_files[0]  # Take first found
            break
    
    # Look for random files
    random_patterns = ["**/random*.json", "**/*random*.json"]
    for pattern in random_patterns:
        random_files = glob.glob(os.path.join(metrics_dir, pattern), recursive=True)
        if random_files:
            files['random'] = random_files[0]  # Take first found
            break
    
    return files


def plot_comparison(ucb_metrics: Dict[str, List[float]], 
                   random_metrics: Dict[str, List[float]], 
                   save_dir: Optional[str] = None):
    """Plot side-by-side comparison of UCB vs Random."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main performance plots (top row)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # UCB-specific plots (middle row)
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Efficiency plots (bottom row)
    ax6 = fig.add_subplot(gs[2, 0])
    ax7 = fig.add_subplot(gs[2, 1])
    ax8 = fig.add_subplot(gs[2, 2])
    
    colors = {'UCB': 'steelblue', 'Random': 'orange'}
    
    # 1. Best Score Evolution
    for name, metrics in [('UCB', ucb_metrics), ('Random', random_metrics)]:
        if 'best_so_far' in metrics:
            rounds = list(range(1, len(metrics['best_so_far']) + 1))
            ax1.plot(rounds, metrics['best_so_far'], 'o-', 
                    color=colors[name], label=name, linewidth=2, markersize=4)
    
    ax1.set_xlabel('Exploration Round')
    ax1.set_ylabel('Best Score So Far')
    ax1.set_title('Performance Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Pool Quality
    for name, metrics in [('UCB', ucb_metrics), ('Random', random_metrics)]:
        if 'pool_mean' in metrics:
            rounds = list(range(1, len(metrics['pool_mean']) + 1))
            ax2.plot(rounds, metrics['pool_mean'], 's-', 
                    color=colors[name], label=name, linewidth=2, markersize=4)
    
    ax2.set_xlabel('Exploration Round')
    ax2.set_ylabel('Pool Mean Score')
    ax2.set_title('Pool Quality Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Replacement Efficiency
    for name, metrics in [('UCB', ucb_metrics), ('Random', random_metrics)]:
        if 'replacement_ratio' in metrics:
            rounds = list(range(1, len(metrics['replacement_ratio']) + 1))
            ax3.plot(rounds, metrics['replacement_ratio'], '^-', 
                    color=colors[name], label=name, linewidth=2, markersize=4)
    
    ax3.set_xlabel('Exploration Round')
    ax3.set_ylabel('Replacement Ratio')
    ax3.set_title('Pool Replacement Efficiency')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. UCB CI Width (UCB only)
    if 'ci_width' in ucb_metrics:
        rounds = list(range(1, len(ucb_metrics['ci_width']) + 1))
        ax4.plot(rounds, ucb_metrics['ci_width'], 'o-', 
                color=colors['UCB'], linewidth=2, markersize=4)
        ax4.set_xlabel('Exploration Round')
        ax4.set_ylabel('CI Width')
        ax4.set_title('UCB Confidence Interval Width')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    else:
        ax4.text(0.5, 0.5, 'No CI Width Data\n(UCB Only)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('UCB Confidence Interval Width')
    
    # 5. UCB Gap (UCB only)
    if 'ucb_gap' in ucb_metrics:
        rounds = list(range(1, len(ucb_metrics['ucb_gap']) + 1))
        ax5.plot(rounds, ucb_metrics['ucb_gap'], 's-', 
                color=colors['UCB'], linewidth=2, markersize=4)
        ax5.set_xlabel('Exploration Round')
        ax5.set_ylabel('UCB Gap')
        ax5.set_title('UCB Gap (Best - 2nd Best)')
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')
    else:
        ax5.text(0.5, 0.5, 'No UCB Gap Data\n(UCB Only)', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('UCB Gap Evolution')
    
    # 6. Token Efficiency
    for name, metrics in [('UCB', ucb_metrics), ('Random', random_metrics)]:
        if 'lift_per_1k_tokens' in metrics:
            rounds = list(range(1, len(metrics['lift_per_1k_tokens']) + 1))
            ax6.plot(rounds, metrics['lift_per_1k_tokens'], 'd-', 
                    color=colors[name], label=name, linewidth=2, markersize=4)
    
    ax6.set_xlabel('Exploration Round')
    ax6.set_ylabel('Lift per 1K Tokens')
    ax6.set_title('Token Efficiency')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 7. Final Performance Comparison (Bar Chart)
    strategies = []
    final_scores = []
    final_pool_means = []
    
    for name, metrics in [('UCB', ucb_metrics), ('Random', random_metrics)]:
        if 'best_so_far' in metrics and metrics['best_so_far']:
            strategies.append(name)
            final_scores.append(metrics['best_so_far'][-1])
            final_pool_means.append(metrics['pool_mean'][-1] if 'pool_mean' in metrics and metrics['pool_mean'] else 0)
    
    if strategies:
        x_pos = np.arange(len(strategies))
        bars = ax7.bar(x_pos, final_scores, color=[colors[s] for s in strategies], alpha=0.7)
        ax7.set_xlabel('Strategy')
        ax7.set_ylabel('Final Best Score')
        ax7.set_title('Final Performance')
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(strategies)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars, final_scores):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01*height,
                    f'{score:.3f}', ha='center', va='bottom')
    
    # 8. Performance Gap Over Time
    if ('best_so_far' in ucb_metrics and 'best_so_far' in random_metrics and
        ucb_metrics['best_so_far'] and random_metrics['best_so_far']):
        
        min_rounds = min(len(ucb_metrics['best_so_far']), len(random_metrics['best_so_far']))
        ucb_scores = ucb_metrics['best_so_far'][:min_rounds]
        random_scores = random_metrics['best_so_far'][:min_rounds]
        
        performance_gap = [u - r for u, r in zip(ucb_scores, random_scores)]
        rounds = list(range(1, min_rounds + 1))
        
        ax8.plot(rounds, performance_gap, 'o-', color='purple', linewidth=2, markersize=4)
        ax8.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax8.set_xlabel('Exploration Round')
        ax8.set_ylabel('UCB Score - Random Score')
        ax8.set_title('Performance Gap (UCB - Random)')
        ax8.grid(True, alpha=0.3)
        
        # Fill positive gap
        ax8.fill_between(rounds, performance_gap, 0, 
                        where=[gap > 0 for gap in performance_gap], 
                        alpha=0.3, color='green', label='UCB Better')
        ax8.fill_between(rounds, performance_gap, 0, 
                        where=[gap < 0 for gap in performance_gap], 
                        alpha=0.3, color='red', label='Random Better')
        ax8.legend()
    
    # Overall title
    fig.suptitle('UCB vs Random Exploration Strategy Comparison', fontsize=16, y=0.95)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'strategy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'strategy_comparison.pdf'), bbox_inches='tight')
    
    plt.show()


def print_summary_stats(ucb_metrics: Dict[str, List[float]], 
                       random_metrics: Dict[str, List[float]]):
    """Print summary statistics comparison."""
    print("\n" + "="*80)
    print("EXPLORATION STRATEGY SUMMARY")
    print("="*80)
    
    def get_final_stats(metrics, name):
        stats = {}
        if 'best_so_far' in metrics and metrics['best_so_far']:
            stats['final_score'] = metrics['best_so_far'][-1]
            stats['max_score'] = max(metrics['best_so_far'])
            stats['improvement'] = metrics['best_so_far'][-1] - metrics['best_so_far'][0]
        
        if 'pool_mean' in metrics and metrics['pool_mean']:
            stats['final_pool_mean'] = metrics['pool_mean'][-1]
        
        if 'replacement_ratio' in metrics and metrics['replacement_ratio']:
            stats['final_replacement_ratio'] = metrics['replacement_ratio'][-1]
        
        if 'lift_per_1k_tokens' in metrics and metrics['lift_per_1k_tokens']:
            stats['token_efficiency'] = metrics['lift_per_1k_tokens'][-1]
        
        return stats
    
    ucb_stats = get_final_stats(ucb_metrics, 'UCB')
    random_stats = get_final_stats(random_metrics, 'Random')
    
    # Print comparison table
    print(f"{'Metric':<25} {'UCB':<12} {'Random':<12} {'Difference':<12}")
    print("-" * 65)
    
    for key in ['final_score', 'max_score', 'improvement', 'final_pool_mean', 'final_replacement_ratio', 'token_efficiency']:
        if key in ucb_stats and key in random_stats:
            ucb_val = ucb_stats[key]
            random_val = random_stats[key]
            diff = ucb_val - random_val
            
            metric_name = key.replace('_', ' ').title()
            print(f"{metric_name:<25} {ucb_val:<12.3f} {random_val:<12.3f} {diff:<+12.3f}")
    
    # UCB-specific metrics
    print(f"\n{'UCB-Specific Metrics':<25}")
    print("-" * 40)
    if 'ci_width' in ucb_metrics and ucb_metrics['ci_width']:
        print(f"{'Final CI Width':<25} {ucb_metrics['ci_width'][-1]:<12.4f}")
        print(f"{'Initial CI Width':<25} {ucb_metrics['ci_width'][0]:<12.4f}")
        ci_reduction = (ucb_metrics['ci_width'][0] - ucb_metrics['ci_width'][-1]) / ucb_metrics['ci_width'][0] * 100
        print(f"{'CI Reduction %':<25} {ci_reduction:<12.1f}")
    
    if 'ucb_gap' in ucb_metrics and ucb_metrics['ucb_gap']:
        print(f"{'Final UCB Gap':<25} {ucb_metrics['ucb_gap'][-1]:<12.4f}")
        print(f"{'Initial UCB Gap':<25} {ucb_metrics['ucb_gap'][0]:<12.4f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare UCB vs Random exploration strategies")
    parser.add_argument("--ucb_file", type=str, help="UCB metrics JSON file")
    parser.add_argument("--random_file", type=str, help="Random metrics JSON file")
    parser.add_argument("--metrics_dir", type=str, help="Directory to auto-find metrics files")
    parser.add_argument("--save_dir", type=str, help="Directory to save plots")
    parser.add_argument("--stats_only", action="store_true", help="Show only summary statistics")
    
    args = parser.parse_args()
    
    # Find files
    if args.ucb_file and args.random_file:
        ucb_file = args.ucb_file
        random_file = args.random_file
    elif args.metrics_dir:
        files = find_strategy_files(args.metrics_dir)
        ucb_file = files.get('ucb')
        random_file = files.get('random')
        
        if not ucb_file:
            print("No UCB metrics file found in directory")
            return
        if not random_file:
            print("No Random metrics file found in directory")
            return
        
        print(f"Found UCB file: {os.path.basename(ucb_file)}")
        print(f"Found Random file: {os.path.basename(random_file)}")
    else:
        print("Please provide either --ucb_file and --random_file, or --metrics_dir")
        return
    
    # Load metrics
    try:
        ucb_data = load_metrics(ucb_file)
        ucb_metrics = extract_trajectory_metrics(ucb_data)
        print(f"Loaded UCB metrics: {list(ucb_metrics.keys())}")
    except Exception as e:
        print(f"Error loading UCB file: {e}")
        return
    
    try:
        random_data = load_metrics(random_file)
        random_metrics = extract_trajectory_metrics(random_data)
        print(f"Loaded Random metrics: {list(random_metrics.keys())}")
    except Exception as e:
        print(f"Error loading Random file: {e}")
        return
    
    # Print summary
    print_summary_stats(ucb_metrics, random_metrics)
    
    # Create plots (unless stats only)
    if not args.stats_only:
        print("\nGenerating comparison plots...")
        plot_comparison(ucb_metrics, random_metrics, args.save_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()
