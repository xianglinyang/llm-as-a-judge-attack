#!/usr/bin/env python3
"""
Specialized dashboard for comparing performance across models and datasets.
Creates comprehensive visualizations showing:
- Model performance across different datasets
- Dataset difficulty across different models  
- Strategy effectiveness per model-dataset combination
- Statistical significance testing

Usage:
    python model_dataset_dashboard.py --metrics_dir path/to/metrics
    python model_dataset_dashboard.py --metrics_dir path/to/metrics --focus_metric best_so_far
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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_parse_metrics(metrics_dir: str) -> pd.DataFrame:
    """Load all metrics files and create a structured DataFrame."""
    
    files = glob.glob(os.path.join(metrics_dir, "**/*.json"), recursive=True)
    
    all_records = []
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Skip warmup/init files - we only want exploration results
            if 'warmup_summary' in data:
                print(f"Skipping warmup file: {os.path.basename(file_path)}")
                continue
            
            # Skip files that are clearly warmup/init related
            filename = os.path.basename(file_path).lower()
            if any(skip_term in filename for skip_term in ['warmup', 'init_ucb', 'init_linucb']):
                print(f"Skipping init/warmup file: {os.path.basename(file_path)}")
                continue
            
            # Extract metadata
            record = {}
            
            # Strategy information
            strategy = data.get('strategy', 'unknown')
            if 'simple_rewrite' in strategy.lower():
                template_name = data.get('template_name', '')
                if template_name:
                    strategy = f"simple_rewrite_{template_name}"
            elif strategy == 'UCB-Warmup':
                # This should be skipped above, but just in case
                print(f"Skipping UCB-Warmup strategy in file: {os.path.basename(file_path)}")
                continue
            record['strategy'] = strategy
            
            # Model information
            record['judge_model'] = data.get('judge_backbone', data.get('judge_model_name', 'unknown'))
            record['llm_agent'] = data.get('llm_agent_name', 'unknown')
            record['response_model'] = data.get('response_model_name', 'unknown')
            record['baseline_model'] = data.get('baseline_response_model_name', 'unknown')
            
            # Dataset and configuration
            record['dataset'] = data.get('dataset_name', 'unknown')
            record['budget'] = data.get('budget', data.get('Budget', 0))
            record['pool_size'] = data.get('pool_size', 0)
            record['judge_type'] = data.get('judge_type', 'unknown')
            record['reward_type'] = data.get('reward_type', 'unknown')
            record['eval_num'] = data.get('eval_num', 0)
            
            # UCB-specific parameters
            record['alpha'] = data.get('alpha', None)
            record['lambda_reg'] = data.get('lambda_reg', None)
            
            # Extract performance metrics
            metrics = {}
            if 'metrics' in data and isinstance(data['metrics'], list):
                # Multiple questions - aggregate
                all_metrics = data['metrics']
                if all_metrics:
                    sample_metrics = all_metrics[0]
                    for key in sample_metrics:
                        if isinstance(sample_metrics[key], list):
                            # Aggregate across questions
                            max_rounds = max(len(m.get(key, [])) for m in all_metrics)
                            aggregated = []
                            for round_idx in range(max_rounds):
                                round_values = [m.get(key, [])[round_idx] 
                                              for m in all_metrics 
                                              if round_idx < len(m.get(key, []))]
                                if round_values:
                                    aggregated.append(np.mean(round_values))
                            metrics[key] = aggregated
            
            # Calculate summary statistics
            if 'best_so_far' in metrics and metrics['best_so_far']:
                record['final_score'] = metrics['best_so_far'][-1]
                record['max_score'] = max(metrics['best_so_far'])
                record['improvement'] = record['final_score'] - metrics['best_so_far'][0]
                record['rounds_to_convergence'] = len(metrics['best_so_far'])
            else:
                record['final_score'] = np.nan
                record['max_score'] = np.nan
                record['improvement'] = np.nan
                record['rounds_to_convergence'] = np.nan
            
            if 'pool_mean' in metrics and metrics['pool_mean']:
                record['final_pool_mean'] = metrics['pool_mean'][-1]
            else:
                record['final_pool_mean'] = np.nan
            
            if 'lift_per_1k_tokens' in metrics and metrics['lift_per_1k_tokens']:
                record['token_efficiency'] = metrics['lift_per_1k_tokens'][-1]
            else:
                record['token_efficiency'] = np.nan
            
            # Store raw metrics for detailed analysis
            record['raw_metrics'] = metrics
            record['file_path'] = file_path
            
            all_records.append(record)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    df = pd.DataFrame(all_records)
    print(f"Loaded {len(df)} experiments from {len(files)} files")
    return df

def plot_model_dataset_heatmap(df: pd.DataFrame, metric: str = 'final_score', save_dir: str = None):
    """Create heatmap showing performance across models and datasets."""
    
    # Filter to main models and datasets for clarity
    main_strategies = ['ucb', 'ucb_with_warmup', 'random', 'simple_rewrite_holistic', 'simple_rewrite_improve']
    df_filtered = df[df['strategy'].isin(main_strategies)]
    
    if df_filtered.empty:
        print("No data for main strategies")
        return
    
    # Group by reward type if multiple types exist
    reward_types = df_filtered['reward_type'].unique()
    reward_types = [rt for rt in reward_types if rt != 'unknown']
    
    if len(reward_types) > 1:
        # Create separate plots for each reward type
        for reward_type in reward_types:
            reward_data = df_filtered[df_filtered['reward_type'] == reward_type]
            if reward_data.empty:
                continue
                
            strategies = reward_data['strategy'].unique()
            n_strategies = len(strategies)
            
            fig, axes = plt.subplots(2, (n_strategies + 1) // 2, figsize=(6 * ((n_strategies + 1) // 2), 10))
            axes = axes.flatten() if n_strategies > 1 else [axes]
            
            fig.suptitle(f'Model-Dataset Performance ({reward_type.title()} Reward)', fontsize=16)
            
            for idx, strategy in enumerate(strategies):
                if idx >= len(axes):
                    break
                    
                ax = axes[idx]
                strategy_data = reward_data[reward_data['strategy'] == strategy]
                
                # Create pivot table
                pivot = strategy_data.pivot_table(
                    values=metric, 
                    index='dataset', 
                    columns='judge_model', 
                    aggfunc='mean'
                )
                
                if pivot.empty:
                    ax.text(0.5, 0.5, f'No data for {strategy}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{strategy.upper()}')
                    continue
                
                # Plot heatmap
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax, 
                           cbar_kws={'label': metric.replace('_', ' ').title()})
                ax.set_title(f'{strategy.upper()}')
                ax.set_xlabel('Judge Model')
                ax.set_ylabel('Dataset')
            
            # Hide unused subplots
            for idx in range(len(strategies), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'model_dataset_heatmap_{metric}_{reward_type}.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(save_dir, f'model_dataset_heatmap_{metric}_{reward_type}.pdf'), bbox_inches='tight')
            
            plt.show()
    else:
        # Single reward type - original behavior
        strategies = df_filtered['strategy'].unique()
        n_strategies = len(strategies)
        
        fig, axes = plt.subplots(2, (n_strategies + 1) // 2, figsize=(6 * ((n_strategies + 1) // 2), 10))
        axes = axes.flatten() if n_strategies > 1 else [axes]
        
        for idx, strategy in enumerate(strategies):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            strategy_data = df_filtered[df_filtered['strategy'] == strategy]
            
            # Create pivot table
            pivot = strategy_data.pivot_table(
                values=metric, 
                index='dataset', 
                columns='judge_model', 
                aggfunc='mean'
            )
            
            if pivot.empty:
                ax.text(0.5, 0.5, f'No data for {strategy}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{strategy.upper()}')
                continue
            
            # Plot heatmap
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax, 
                       cbar_kws={'label': metric.replace('_', ' ').title()})
            ax.set_title(f'{strategy.upper()}')
            ax.set_xlabel('Judge Model')
            ax.set_ylabel('Dataset')
        
        # Hide unused subplots
        for idx in range(len(strategies), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'model_dataset_heatmap_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'model_dataset_heatmap_{metric}.pdf'), bbox_inches='tight')
        
        plt.show()

def plot_strategy_effectiveness_by_context(df: pd.DataFrame, save_dir: str = None):
    """Plot strategy effectiveness across different model-dataset combinations."""
    
    # Create combination labels
    df['model_dataset'] = df['judge_model'] + ' + ' + df['dataset']
    
    # Filter to meaningful combinations (at least 2 different strategies)
    combination_counts = df.groupby('model_dataset')['strategy'].nunique()
    valid_combinations = combination_counts[combination_counts >= 2].index
    
    df_filtered = df[df['model_dataset'].isin(valid_combinations)]
    
    if df_filtered.empty:
        print("No model-dataset combinations with multiple strategies")
        return
    
    # Create box plot
    plt.figure(figsize=(15, 8))
    
    sns.boxplot(data=df_filtered, x='model_dataset', y='final_score', hue='strategy')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Model + Dataset Combination')
    plt.ylabel('Final Score')
    plt.title('Strategy Performance Across Model-Dataset Combinations')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'strategy_effectiveness_by_context.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'strategy_effectiveness_by_context.pdf'), bbox_inches='tight')
    
    plt.show()

def plot_dataset_difficulty_ranking(df: pd.DataFrame, save_dir: str = None):
    """Rank datasets by difficulty (lower scores = harder)."""
    
    # Calculate mean performance per dataset across all strategies/models
    dataset_stats = df.groupby('dataset').agg({
        'final_score': ['mean', 'std', 'count'],
        'improvement': ['mean'],
        'token_efficiency': ['mean']
    }).round(4)
    
    dataset_stats.columns = ['mean_score', 'std_score', 'count', 'mean_improvement', 'mean_token_eff']
    dataset_stats = dataset_stats.sort_values('mean_score')
    
    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Dataset difficulty (mean score)
    ax1.barh(range(len(dataset_stats)), dataset_stats['mean_score'], 
             xerr=dataset_stats['std_score'], capsize=5)
    ax1.set_yticks(range(len(dataset_stats)))
    ax1.set_yticklabels(dataset_stats.index)
    ax1.set_xlabel('Mean Final Score')
    ax1.set_title('Dataset Difficulty Ranking\n(Lower = Harder)')
    ax1.grid(True, alpha=0.3)
    
    # Dataset improvement potential
    ax2.barh(range(len(dataset_stats)), dataset_stats['mean_improvement'])
    ax2.set_yticks(range(len(dataset_stats)))
    ax2.set_yticklabels(dataset_stats.index)
    ax2.set_xlabel('Mean Improvement')
    ax2.set_title('Improvement Potential by Dataset')
    ax2.grid(True, alpha=0.3)
    
    # Sample count
    ax3.barh(range(len(dataset_stats)), dataset_stats['count'])
    ax3.set_yticks(range(len(dataset_stats)))
    ax3.set_yticklabels(dataset_stats.index)
    ax3.set_xlabel('Number of Experiments')
    ax3.set_title('Experiment Coverage by Dataset')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'dataset_difficulty_ranking.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'dataset_difficulty_ranking.pdf'), bbox_inches='tight')
    
    plt.show()
    
    return dataset_stats

def plot_model_performance_comparison(df: pd.DataFrame, save_dir: str = None):
    """Compare judge model performance across datasets and strategies."""
    
    # Calculate statistics per judge model
    model_stats = df.groupby(['judge_model', 'strategy']).agg({
        'final_score': ['mean', 'std', 'count'],
        'improvement': ['mean'],
        'token_efficiency': ['mean']
    }).round(4)
    
    model_stats.columns = ['mean_score', 'std_score', 'count', 'mean_improvement', 'mean_token_eff']
    model_stats = model_stats.reset_index()
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Mean performance by model and strategy
    pivot_score = model_stats.pivot(index='judge_model', columns='strategy', values='mean_score')
    sns.heatmap(pivot_score, annot=True, fmt='.3f', cmap='viridis', ax=axes[0,0])
    axes[0,0].set_title('Mean Final Score by Model & Strategy')
    
    # Improvement by model and strategy
    pivot_improvement = model_stats.pivot(index='judge_model', columns='strategy', values='mean_improvement')
    sns.heatmap(pivot_improvement, annot=True, fmt='.3f', cmap='viridis', ax=axes[0,1])
    axes[0,1].set_title('Mean Improvement by Model & Strategy')
    
    # Token efficiency
    pivot_token_eff = model_stats.pivot(index='judge_model', columns='strategy', values='mean_token_eff')
    sns.heatmap(pivot_token_eff, annot=True, fmt='.3f', cmap='viridis', ax=axes[1,0])
    axes[1,0].set_title('Token Efficiency by Model & Strategy')
    
    # Experiment count
    pivot_count = model_stats.pivot(index='judge_model', columns='strategy', values='count')
    sns.heatmap(pivot_count, annot=True, fmt='.0f', cmap='Blues', ax=axes[1,1])
    axes[1,1].set_title('Experiment Count by Model & Strategy')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'model_performance_comparison.pdf'), bbox_inches='tight')
    
    plt.show()

def plot_reward_type_comparison(df: pd.DataFrame, save_dir: str = None):
    """Compare performance between different reward types."""
    
    # Filter out unknown reward types
    df_filtered = df[df['reward_type'] != 'unknown']
    reward_types = df_filtered['reward_type'].unique()
    
    if len(reward_types) < 2:
        print(f"Only one reward type found: {reward_types}. Need at least 2 for comparison.")
        return
    
    print(f"Comparing reward types: {reward_types}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Overall performance by reward type
    ax1 = axes[0, 0]
    sns.boxplot(data=df_filtered, x='reward_type', y='final_score', ax=ax1)
    ax1.set_title('Final Score by Reward Type')
    ax1.set_xlabel('Reward Type')
    ax1.set_ylabel('Final Score')
    
    # Add sample counts
    for i, reward_type in enumerate(reward_types):
        count = len(df_filtered[df_filtered['reward_type'] == reward_type])
        ax1.text(i, ax1.get_ylim()[0], f'n={count}', ha='center', va='bottom')
    
    # 2. Strategy performance by reward type
    ax2 = axes[0, 1]
    sns.boxplot(data=df_filtered, x='reward_type', y='final_score', hue='strategy', ax=ax2)
    ax2.set_title('Strategy Performance by Reward Type')
    ax2.set_xlabel('Reward Type')
    ax2.set_ylabel('Final Score')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Improvement by reward type
    ax3 = axes[1, 0]
    sns.boxplot(data=df_filtered, x='reward_type', y='improvement', ax=ax3)
    ax3.set_title('Improvement by Reward Type')
    ax3.set_xlabel('Reward Type')
    ax3.set_ylabel('Score Improvement')
    
    # 4. Token efficiency by reward type
    ax4 = axes[1, 1]
    df_token_filtered = df_filtered.dropna(subset=['token_efficiency'])
    if not df_token_filtered.empty:
        sns.boxplot(data=df_token_filtered, x='reward_type', y='token_efficiency', ax=ax4)
        ax4.set_title('Token Efficiency by Reward Type')
        ax4.set_xlabel('Reward Type')
        ax4.set_ylabel('Lift per 1K Tokens')
    else:
        ax4.text(0.5, 0.5, 'No token efficiency data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Token Efficiency by Reward Type')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'reward_type_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'reward_type_comparison.pdf'), bbox_inches='tight')
    
    plt.show()
    
    # Statistical comparison
    print(f"\n{'='*60}")
    print("REWARD TYPE STATISTICAL COMPARISON")
    print(f"{'='*60}")
    
    for strategy in df_filtered['strategy'].unique():
        strategy_data = df_filtered[df_filtered['strategy'] == strategy]
        if len(strategy_data) < 2:
            continue
            
        print(f"\nStrategy: {strategy}")
        print("-" * 30)
        
        for rt1 in reward_types:
            for rt2 in reward_types:
                if rt1 >= rt2:  # Avoid duplicate comparisons
                    continue
                    
                data1 = strategy_data[strategy_data['reward_type'] == rt1]['final_score'].dropna()
                data2 = strategy_data[strategy_data['reward_type'] == rt2]['final_score'].dropna()
                
                if len(data1) < 2 or len(data2) < 2:
                    continue
                
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(data1, data2)
                mean1, mean2 = data1.mean(), data2.mean()
                
                sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"{rt1} vs {rt2}: μ1={mean1:.3f}, μ2={mean2:.3f}, p={p_value:.4f} {sig_marker}")

def analyze_statistical_significance(df: pd.DataFrame):
    """Perform statistical significance testing between strategies."""
    
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Group by model-dataset combinations
    df['context'] = df['judge_model'] + '_' + df['dataset']
    
    significant_results = []
    
    for context in df['context'].unique():
        context_data = df[df['context'] == context]
        strategies = context_data['strategy'].unique()
        
        if len(strategies) < 2:
            continue
        
        print(f"\nContext: {context}")
        print("-" * 50)
        
        # Compare all pairs of strategies
        for i, strat1 in enumerate(strategies):
            for strat2 in strategies[i+1:]:
                data1 = context_data[context_data['strategy'] == strat1]['final_score'].dropna()
                data2 = context_data[context_data['strategy'] == strat2]['final_score'].dropna()
                
                if len(data1) < 2 or len(data2) < 2:
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(data1, data2)
                
                mean1, mean2 = data1.mean(), data2.mean()
                effect_size = (mean1 - mean2) / np.sqrt(((data1.std()**2 + data2.std()**2) / 2))
                
                significant = p_value < 0.05
                
                result = {
                    'context': context,
                    'strategy1': strat1,
                    'strategy2': strat2,
                    'mean1': mean1,
                    'mean2': mean2,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': significant
                }
                significant_results.append(result)
                
                sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"{strat1} vs {strat2}: μ1={mean1:.3f}, μ2={mean2:.3f}, p={p_value:.4f} {sig_marker}")
    
    # Summary of significant differences
    sig_df = pd.DataFrame(significant_results)
    if not sig_df.empty:
        sig_only = sig_df[sig_df['significant']]
        print(f"\n{'='*80}")
        print(f"SUMMARY: Found {len(sig_only)} significant differences out of {len(sig_df)} comparisons")
        print(f"{'='*80}")
        
        for _, row in sig_only.iterrows():
            better = row['strategy1'] if row['mean1'] > row['mean2'] else row['strategy2']
            worse = row['strategy2'] if row['mean1'] > row['mean2'] else row['strategy1']
            print(f"{row['context']}: {better} > {worse} (p={row['p_value']:.4f}, effect={row['effect_size']:.3f})")
    
    return significant_results

def create_comprehensive_summary(df: pd.DataFrame):
    """Create comprehensive summary statistics."""
    
    print(f"\n{'='*100}")
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print(f"{'='*100}")
    
    # Overall statistics
    print(f"\nDataset Coverage:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Unique datasets: {df['dataset'].nunique()}")
    print(f"  Unique judge models: {df['judge_model'].nunique()}")
    print(f"  Unique strategies: {df['strategy'].nunique()}")
    print(f"  Unique reward types: {df['reward_type'].nunique()}")
    
    datasets = df['dataset'].value_counts()
    models = df['judge_model'].value_counts()
    strategies = df['strategy'].value_counts()
    reward_types = df['reward_type'].value_counts()
    
    print(f"\n  Datasets: {dict(datasets)}")
    print(f"  Judge models: {dict(models)}")
    print(f"  Strategies: {dict(strategies)}")
    print(f"  Reward types: {dict(reward_types)}")
    
    # Performance summary by strategy
    print(f"\nPerformance by Strategy:")
    strategy_summary = df.groupby('strategy').agg({
        'final_score': ['mean', 'std', 'count'],
        'improvement': ['mean', 'std'],
        'token_efficiency': ['mean', 'std']
    }).round(4)
    
    strategy_summary.columns = ['mean_final', 'std_final', 'count', 'mean_improvement', 'std_improvement', 'mean_token_eff', 'std_token_eff']
    print(strategy_summary.to_string())
    
    # Best performing combinations
    print(f"\nTop 10 Model-Dataset-Strategy Combinations (by final score):")
    df['combination'] = df['judge_model'] + ' + ' + df['dataset'] + ' + ' + df['strategy']
    top_combinations = df.nlargest(10, 'final_score')[['combination', 'final_score', 'improvement', 'token_efficiency']]
    print(top_combinations.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Model and Dataset Performance Dashboard")
    parser.add_argument("--metrics_dir", type=str, required=True, help="Directory containing metrics files")
    parser.add_argument("--save_dir", type=str, help="Directory to save plots")
    parser.add_argument("--focus_metric", type=str, default="final_score", 
                       choices=["final_score", "improvement", "token_efficiency"],
                       help="Primary metric to focus on")
    parser.add_argument("--skip_stats", action="store_true", help="Skip statistical significance testing")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading and parsing metrics...")
    df = load_and_parse_metrics(args.metrics_dir)
    
    if df.empty:
        print("No valid data found")
        return
    
    # Create comprehensive summary
    create_comprehensive_summary(df)
    
    # Generate visualizations
    print("\nGenerating model-dataset heatmap...")
    plot_model_dataset_heatmap(df, args.focus_metric, args.save_dir)
    
    print("Generating strategy effectiveness comparison...")
    plot_strategy_effectiveness_by_context(df, args.save_dir)
    
    print("Generating dataset difficulty ranking...")
    dataset_stats = plot_dataset_difficulty_ranking(df, args.save_dir)
    
    print("Generating model performance comparison...")
    plot_model_performance_comparison(df, args.save_dir)
    
    print("Generating reward type comparison...")
    plot_reward_type_comparison(df, args.save_dir)
    
    # Statistical analysis
    if not args.skip_stats:
        print("Performing statistical significance analysis...")
        sig_results = analyze_statistical_significance(df)
    
    print(f"\nDashboard complete! Generated visualizations" + (f" saved to {args.save_dir}" if args.save_dir else ""))

if __name__ == "__main__":
    main()
