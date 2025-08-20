#!/usr/bin/env python3
"""
Interactive dashboard for exploration results analysis.

Usage:
    python exploration_dashboard.py --metrics_dir /path/to/metrics/
    python exploration_dashboard.py --config dashboard_config.json
"""

import argparse
import json
import os
import glob
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ExplorationDashboard:
    """Dashboard for analyzing exploration results."""
    
    def __init__(self, metrics_dir: str = None, metrics_files: List[str] = None):
        self.metrics_dir = metrics_dir
        self.metrics_files = metrics_files or []
        self.data = {}
        self.strategies = set()
        
        # Load data
        self.load_all_data()
    
    def load_all_data(self):
        """Load and organize all metrics data."""
        if self.metrics_dir:
            self.load_from_directory()
        elif self.metrics_files:
            self.load_from_files()
    
    def load_from_directory(self):
        """Load data from directory structure."""
        patterns = {
            'ucb_with_warmup': ["**/ucb_with_warmup*.json", "**/*UCB_WITH_WARMUP*.json"],
            'ucb': ["**/ucb*.json", "**/*UCB*.json"],  # Will filter out with_warmup files later
            'random': ["**/random*.json", "**/*random*.json"], 
            'baseline': ["**/baseline*.json", "**/direct*.json"],
            'warmup': ["**/warmup*.json", "**/init_ucb_warmup*.json"]
        }
        
        # Collect files by strategy
        files_by_strategy = {}
        for strategy, pattern_list in patterns.items():
            files = []
            for pattern in pattern_list:
                files.extend(glob.glob(os.path.join(self.metrics_dir, pattern), recursive=True))
            files_by_strategy[strategy] = list(set(files))
        
        # Post-process to fix overlapping patterns from glob
        # Remove ucb_with_warmup files from regular ucb category
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
        
        # Process files for each strategy
        for strategy, files in files_by_strategy.items():
            if files:
                self.strategies.add(strategy)
                self.data[strategy] = []
                
                for file_path in files:
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        processed = self.process_metrics(data, strategy)
                        if processed:
                            processed['file_path'] = file_path
                            processed['strategy'] = strategy
                            self.data[strategy].append(processed)
                            
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    
    def load_from_files(self):
        """Load data from specific files."""
        for file_path in self.metrics_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Try to infer strategy from filename
                filename = os.path.basename(file_path).lower()
                if 'ucb_with_warmup' in filename:
                    strategy = 'ucb_with_warmup'
                elif 'ucb' in filename and 'ucb_with_warmup' not in filename:
                    # Explicit check to avoid false positives
                    strategy = 'ucb'
                elif 'random' in filename:
                    strategy = 'random'
                elif 'warmup' in filename:
                    strategy = 'warmup'
                else:
                    strategy = 'baseline'
                
                self.strategies.add(strategy)
                if strategy not in self.data:
                    self.data[strategy] = []
                
                processed = self.process_metrics(data, strategy)
                if processed:
                    processed['file_path'] = file_path
                    processed['strategy'] = strategy
                    self.data[strategy].append(processed)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def process_metrics(self, data: Dict[str, Any], strategy: str) -> Optional[Dict[str, Any]]:
        """Process raw metrics data into standardized format."""
        result = {}
        
        # Handle different data structures
        if strategy == 'warmup':
            # Warmup data structure
            if 'warmup_summary' in data:
                warmup_data = data['warmup_summary']
                result['median_ci'] = warmup_data.get('median_ci', [])
                result['median_gap'] = warmup_data.get('median_gap', [])
                result['rounds'] = warmup_data.get('rounds', 0)
                result['alpha'] = warmup_data.get('alpha', None)
                result['lambda_reg'] = warmup_data.get('lambda_reg', None)
                result['dataset_name'] = warmup_data.get('dataset_name', 'Unknown')
            
        elif 'metrics' in data and isinstance(data['metrics'], list):
            # Multiple questions - aggregate metrics
            all_metrics = data['metrics']
            if all_metrics:
                sample_metrics = all_metrics[0]
                for key in ['best_so_far', 'pool_mean', 'replacement_ratio', 'lift_per_1k_tokens', 'ci_width', 'ucb_gap']:
                    if key in sample_metrics:
                        max_rounds = max(len(m.get(key, [])) for m in all_metrics if m.get(key))
                        if max_rounds > 0:
                            aggregated = []
                            for round_idx in range(max_rounds):
                                round_values = [m.get(key, [])[round_idx] 
                                              for m in all_metrics 
                                              if m.get(key) and round_idx < len(m.get(key, []))]
                                if round_values:
                                    aggregated.append(np.mean(round_values))
                            result[key] = aggregated
                
                # Extract metadata
                if 'alpha' in data:
                    result['alpha'] = data['alpha']
                if 'lambda_reg' in data:
                    result['lambda_reg'] = data['lambda_reg']
                if 'dataset_name' in data:
                    result['dataset_name'] = data['dataset_name']
                
                result['num_questions'] = len(all_metrics)
        
        else:
            # Single question or direct metrics
            for key in ['best_so_far', 'pool_mean', 'replacement_ratio', 'lift_per_1k_tokens', 'ci_width', 'ucb_gap', 'median_ci', 'median_gap']:
                if key in data and isinstance(data[key], list):
                    result[key] = data[key]
        
        return result if result else None
    
    def create_overview_dashboard(self, save_dir: Optional[str] = None):
        """Create comprehensive overview dashboard."""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Performance Evolution (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_performance_evolution(ax1)
        
        # 2. Final Performance Comparison (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self.plot_final_performance_bars(ax2)
        
        # 3. UCB-specific metrics (second row)
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_ucb_metrics(ax3, ax4)
        
        # 4. Token Efficiency (second row right)
        ax5 = fig.add_subplot(gs[1, 2:])
        self.plot_token_efficiency(ax5)
        
        # 5. Pool Quality and Replacement (third row)
        ax6 = fig.add_subplot(gs[2, :2])
        ax7 = fig.add_subplot(gs[2, 2:])
        self.plot_pool_metrics(ax6, ax7)
        
        # 6. Strategy Statistics (bottom row)
        ax8 = fig.add_subplot(gs[3, :])
        self.plot_strategy_statistics(ax8)
        
        fig.suptitle('Exploration Strategy Analysis Dashboard', fontsize=18, y=0.98)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'exploration_dashboard.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, 'exploration_dashboard.pdf'), bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_evolution(self, ax):
        """Plot performance evolution over rounds."""
        colors = {'ucb': 'steelblue', 'random': 'orange', 'baseline': 'green', 'warmup': 'purple'}
        
        for strategy in self.strategies:
            if strategy not in self.data:
                continue
                
            strategy_data = self.data[strategy]
            if not strategy_data:
                continue
            
            # Aggregate across runs
            all_trajectories = []
            for run_data in strategy_data:
                if 'best_so_far' in run_data:
                    all_trajectories.append(run_data['best_so_far'])
            
            if all_trajectories:
                max_rounds = max(len(traj) for traj in all_trajectories)
                mean_trajectory = []
                std_trajectory = []
                
                for round_idx in range(max_rounds):
                    round_values = [traj[round_idx] for traj in all_trajectories if round_idx < len(traj)]
                    if round_values:
                        mean_trajectory.append(np.mean(round_values))
                        std_trajectory.append(np.std(round_values))
                
                rounds = list(range(1, len(mean_trajectory) + 1))
                ax.plot(rounds, mean_trajectory, 'o-', 
                       color=colors.get(strategy, 'gray'), 
                       label=f'{strategy.upper()} (n={len(all_trajectories)})', 
                       linewidth=2, markersize=4)
                
                if len(all_trajectories) > 1:
                    ax.fill_between(rounds, 
                                   np.array(mean_trajectory) - np.array(std_trajectory),
                                   np.array(mean_trajectory) + np.array(std_trajectory),
                                   color=colors.get(strategy, 'gray'), alpha=0.2)
        
        ax.set_xlabel('Exploration Round')
        ax.set_ylabel('Best Score So Far')
        ax.set_title('Performance Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def plot_final_performance_bars(self, ax):
        """Plot final performance comparison as bars."""
        strategies = []
        mean_scores = []
        std_scores = []
        colors = {'ucb': 'steelblue', 'random': 'orange', 'baseline': 'green', 'warmup': 'purple'}
        
        for strategy in sorted(self.strategies):
            if strategy not in self.data:
                continue
            
            final_scores = []
            for run_data in self.data[strategy]:
                if 'best_so_far' in run_data and run_data['best_so_far']:
                    final_scores.append(run_data['best_so_far'][-1])
            
            if final_scores:
                strategies.append(strategy.upper())
                mean_scores.append(np.mean(final_scores))
                std_scores.append(np.std(final_scores))
        
        if strategies:
            x_pos = np.arange(len(strategies))
            bars = ax.bar(x_pos, mean_scores, 
                         color=[colors.get(s.lower(), 'gray') for s in strategies], 
                         alpha=0.7, yerr=std_scores, capsize=5)
            
            ax.set_xlabel('Strategy')
            ax.set_ylabel('Final Best Score')
            ax.set_title('Final Performance Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(strategies)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, score, std in zip(bars, mean_scores, std_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01*height,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    def plot_ucb_metrics(self, ax1, ax2):
        """Plot UCB-specific metrics (CI width and gap)."""
        if 'ucb' not in self.data:
            ax1.text(0.5, 0.5, 'No UCB Data Available', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No UCB Data Available', ha='center', va='center', transform=ax2.transAxes)
            ax1.set_title('UCB CI Width')
            ax2.set_title('UCB Gap')
            return
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.data['ucb'])))
        
        # CI Width
        for i, run_data in enumerate(self.data['ucb']):
            if 'ci_width' in run_data:
                rounds = list(range(1, len(run_data['ci_width']) + 1))
                ax1.plot(rounds, run_data['ci_width'], 'o-', 
                        color=colors[i], alpha=0.7, linewidth=1.5, markersize=3)
        
        ax1.set_xlabel('Round')
        ax1.set_ylabel('CI Width')
        ax1.set_title('UCB Confidence Interval Width')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # UCB Gap
        for i, run_data in enumerate(self.data['ucb']):
            if 'ucb_gap' in run_data:
                rounds = list(range(1, len(run_data['ucb_gap']) + 1))
                ax2.plot(rounds, run_data['ucb_gap'], 's-', 
                        color=colors[i], alpha=0.7, linewidth=1.5, markersize=3)
        
        ax2.set_xlabel('Round')
        ax2.set_ylabel('UCB Gap')
        ax2.set_title('UCB Gap Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    def plot_token_efficiency(self, ax):
        """Plot token efficiency comparison."""
        colors = {'ucb': 'steelblue', 'random': 'orange', 'baseline': 'green', 'warmup': 'purple'}
        
        for strategy in self.strategies:
            if strategy not in self.data:
                continue
            
            all_efficiency = []
            for run_data in self.data[strategy]:
                if 'lift_per_1k_tokens' in run_data:
                    all_efficiency.append(run_data['lift_per_1k_tokens'])
            
            if all_efficiency:
                max_rounds = max(len(eff) for eff in all_efficiency)
                mean_efficiency = []
                
                for round_idx in range(max_rounds):
                    round_values = [eff[round_idx] for eff in all_efficiency if round_idx < len(eff)]
                    if round_values:
                        mean_efficiency.append(np.mean(round_values))
                
                rounds = list(range(1, len(mean_efficiency) + 1))
                ax.plot(rounds, mean_efficiency, 'o-', 
                       color=colors.get(strategy, 'gray'), 
                       label=f'{strategy.upper()}', linewidth=2, markersize=4)
        
        ax.set_xlabel('Exploration Round')
        ax.set_ylabel('Lift per 1K Tokens')
        ax.set_title('Token Efficiency Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def plot_pool_metrics(self, ax1, ax2):
        """Plot pool quality and replacement metrics."""
        colors = {'ucb': 'steelblue', 'random': 'orange', 'baseline': 'green', 'warmup': 'purple'}
        
        # Pool Mean
        for strategy in self.strategies:
            if strategy not in self.data:
                continue
            
            all_pool_mean = []
            for run_data in self.data[strategy]:
                if 'pool_mean' in run_data:
                    all_pool_mean.append(run_data['pool_mean'])
            
            if all_pool_mean:
                max_rounds = max(len(pm) for pm in all_pool_mean)
                mean_pool = []
                
                for round_idx in range(max_rounds):
                    round_values = [pm[round_idx] for pm in all_pool_mean if round_idx < len(pm)]
                    if round_values:
                        mean_pool.append(np.mean(round_values))
                
                rounds = list(range(1, len(mean_pool) + 1))
                ax1.plot(rounds, mean_pool, 'o-', 
                        color=colors.get(strategy, 'gray'), 
                        label=f'{strategy.upper()}', linewidth=2, markersize=4)
        
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Pool Mean Score')
        ax1.set_title('Pool Quality Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Replacement Ratio
        for strategy in self.strategies:
            if strategy not in self.data:
                continue
            
            all_replacement = []
            for run_data in self.data[strategy]:
                if 'replacement_ratio' in run_data:
                    all_replacement.append(run_data['replacement_ratio'])
            
            if all_replacement:
                max_rounds = max(len(rr) for rr in all_replacement)
                mean_replacement = []
                
                for round_idx in range(max_rounds):
                    round_values = [rr[round_idx] for rr in all_replacement if round_idx < len(rr)]
                    if round_values:
                        mean_replacement.append(np.mean(round_values))
                
                rounds = list(range(1, len(mean_replacement) + 1))
                ax2.plot(rounds, mean_replacement, 's-', 
                        color=colors.get(strategy, 'gray'), 
                        label=f'{strategy.upper()}', linewidth=2, markersize=4)
        
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Replacement Ratio')
        ax2.set_title('Pool Replacement Efficiency')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    def plot_strategy_statistics(self, ax):
        """Plot strategy statistics table as text."""
        ax.axis('off')
        
        stats_data = []
        headers = ['Strategy', 'Runs', 'Avg Final Score', 'Best Score', 'Avg Token Eff', 'Success Rate %']
        
        all_final_scores = []
        for strategy_data in self.data.values():
            for run_data in strategy_data:
                if 'best_so_far' in run_data and run_data['best_so_far']:
                    all_final_scores.append(run_data['best_so_far'][-1])
        
        if all_final_scores:
            success_threshold = np.percentile(all_final_scores, 75)  # Top 25%
        else:
            success_threshold = 0
        
        for strategy in sorted(self.strategies):
            if strategy not in self.data:
                continue
            
            final_scores = []
            token_effs = []
            
            for run_data in self.data[strategy]:
                if 'best_so_far' in run_data and run_data['best_so_far']:
                    final_scores.append(run_data['best_so_far'][-1])
                
                if 'lift_per_1k_tokens' in run_data and run_data['lift_per_1k_tokens']:
                    token_effs.append(run_data['lift_per_1k_tokens'][-1])
            
            if final_scores:
                success_rate = sum(1 for score in final_scores if score >= success_threshold) / len(final_scores) * 100
                
                stats_data.append([
                    strategy.upper(),
                    len(self.data[strategy]),
                    f"{np.mean(final_scores):.3f}",
                    f"{max(final_scores):.3f}",
                    f"{np.mean(token_effs):.3f}" if token_effs else "N/A",
                    f"{success_rate:.1f}%"
                ])
        
        # Sort by average final score
        stats_data.sort(key=lambda x: float(x[2]), reverse=True)
        
        # Create table
        table_text = []
        col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *stats_data)]
        
        def format_row(row):
            return "  ".join(str(item).ljust(width) for item, width in zip(row, col_widths))
        
        table_text.append(format_row(headers))
        table_text.append("-" * sum(col_widths + [2] * (len(headers) - 1)))
        table_text.extend([format_row(row) for row in stats_data])
        
        ax.text(0.05, 0.95, '\n'.join(table_text), transform=ax.transAxes, 
               fontfamily='monospace', fontsize=10, verticalalignment='top')
        ax.set_title('Strategy Performance Summary', fontsize=12, pad=20)
    
    def print_summary(self):
        """Print text summary of loaded data."""
        print("\n" + "="*80)
        print("EXPLORATION DASHBOARD SUMMARY")
        print("="*80)
        
        total_runs = sum(len(runs) for runs in self.data.values())
        print(f"Total runs loaded: {total_runs}")
        print(f"Strategies found: {', '.join(sorted(self.strategies))}")
        
        for strategy in sorted(self.strategies):
            if strategy in self.data:
                runs = self.data[strategy]
                print(f"\n{strategy.upper()}: {len(runs)} runs")
                
                for i, run in enumerate(runs[:3]):  # Show first 3 runs
                    metrics = list(run.keys())
                    print(f"  Run {i+1}: {', '.join(m for m in metrics if not m.startswith('file_'))}")
                
                if len(runs) > 3:
                    print(f"  ... and {len(runs) - 3} more runs")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Exploration Results Dashboard")
    parser.add_argument("--metrics_dir", type=str, help="Directory containing metrics files")
    parser.add_argument("--metrics_files", nargs="+", help="Specific metrics files")
    parser.add_argument("--save_dir", type=str, help="Directory to save plots")
    parser.add_argument("--summary_only", action="store_true", help="Show only summary, no plots")
    
    args = parser.parse_args()
    
    if not args.metrics_dir and not args.metrics_files:
        print("Please provide either --metrics_dir or --metrics_files")
        return
    
    # Create dashboard
    dashboard = ExplorationDashboard(
        metrics_dir=args.metrics_dir,
        metrics_files=args.metrics_files
    )
    
    # Print summary
    dashboard.print_summary()
    
    # Create plots unless summary only
    if not args.summary_only:
        print("\nGenerating dashboard...")
        dashboard.create_overview_dashboard(args.save_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()
