import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Any

from src.results_analysis.results_loader.metrics_loader import MetricsLoader


def plot_4x5_main_figure(all_data: List[Dict[str, Any]], metric_name: str = "best_so_far", save_dir: str = None):
    """
    Creates the specific 4x5 visualization for the main paper text.
    - 4 rows: Alpaca/Pairwise, Alpaca/Pointwise, ArenaHard/Pairwise, ArenaHard/Pointwise
    - 5 columns: o3-mini, gemini-2.5-flash, llama-70b, qwen3-235b, deepseek-r1-0528
    """
    """
    Creates the specific 4x4 visualization for the main paper text.
    - 4 rows: Alpaca/Pairwise, Alpaca/Pointwise, ArenaHard/Pairwise, ArenaHard/Pointwise
    - 4 columns: o3-mini, gemini-2.5-flash, llama-70b, qwen3-235b
    """
    # --- Configuration for the 4x4 Grid ---
    # Define the exact models and their partial names for matching
    MODELS_ORDER = {
        'o3-mini': 'o3-mini',
        'Gemini-2.5-Flash': 'gemini-2.5-flash',
        'Llama3-70B': 'llama-3.3-70b-instruct',
        'Qwen3-235b': 'qwen3-235b-a22b-2507',
        "DeepSeek-R1": "deepseek-r1-0528"
    }

    # Define the configuration for each row
    ROW_CONFIG = [
        {'dataset': 'AlpacaEval', 'judge_type': 'pointwise'},
        {'dataset': 'AlpacaEval', 'judge_type': 'alpaca_eval'},
        {'dataset': 'ArenaHard', 'judge_type': 'pointwise'},
        {'dataset': 'ArenaHard', 'judge_type': 'arena_hard_auto'},
    ]
    
    # Define consistent styles for each attack strategy
    STRATEGY_STYLES = {
        'ucb': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'BITE (ours)'},
        'random': {'color': '#2ca02c', 'marker': '^', 'linestyle': '--', 'label': 'Random'},
        'simple_rewrite_improve': {'color': '#9467bd', 'marker': 'v', 'linestyle': ':', 'label': 'Holistic Rewrite'},
    }

    # Create the 4x5 subplot grid
    # sharey='row' is crucial for making plots in the same row comparable
    fig, axes = plt.subplots(4, 5, figsize=(26, 20), sharex=True, sharey='row')
    # fig.suptitle('Attack Performance Across Judges, Datasets, and Evaluation Types', fontsize=20, y=0.97)

    # --- Main Loop to Populate the Grid ---
    for row_idx, row_spec in enumerate(ROW_CONFIG):
        for col_idx, (model_display_name, model_file_str) in enumerate(MODELS_ORDER.items()):
            ax = axes[row_idx, col_idx]

            # 1. Filter data for the current subplot
            subplot_data = [
                item for item in all_data
                if item['metadata']['dataset'] == row_spec['dataset'] and
                   item['metadata']['judge_type'] == row_spec['judge_type'] and
                   model_file_str in item['metadata']['judge_model']
            ]

            # 2. Group the filtered data by strategy
            strategies_in_subplot = defaultdict(list)
            for item in subplot_data:
                strategy = item['metadata']['strategy']
                if strategy in STRATEGY_STYLES:
                    strategies_in_subplot[strategy].append(item['metrics'])
                else:
                    # Debug: print unknown strategies
                    print(f"Debug: Unknown strategy '{strategy}' for {model_display_name}, {row_spec}")

            # 3. Plot each strategy's learning curve
            if not strategies_in_subplot:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=17, alpha=0.5)
                print(f"Debug: No data for {model_display_name}, {row_spec}. Found {len(subplot_data)} items but none matched strategy styles.")

            for strategy, metrics_list in strategies_in_subplot.items():
                all_runs = [m[metric_name] for m in metrics_list if metric_name in m and m[metric_name]]
                if not all_runs:
                    continue

                # Aggregate metrics across runs (mean and std)
                max_len = max(len(run) for run in all_runs)
                mean_curve = [np.mean([run[i] for run in all_runs if i < len(run)]) for i in range(max_len)]
                std_curve = [np.std([run[i] for run in all_runs if i < len(run)]) for i in range(max_len)]
                rounds = np.arange(1, len(mean_curve) + 1)
                
                style = STRATEGY_STYLES[strategy]
                # ax.plot(rounds, mean_curve, label=f"{style['label']} (n={len(all_runs)})", **{k: v for k, v in style.items() if k != 'label'})
                ax.plot(rounds, mean_curve, label=f"{style['label']}", **{k: v for k, v in style.items() if k != 'label'})
                ax.fill_between(rounds, np.array(mean_curve) - np.array(std_curve), np.array(mean_curve) + np.array(std_curve), color=style['color'], alpha=0.15)

            # --- 4. Formatting and Labeling ---
            # Set model titles only for the top row
            if row_idx == 0:
                ax.set_title(model_display_name, fontsize=20, pad=10)

            # Set y-axis labels only for the leftmost column
            if col_idx == 0:
                ax.set_ylabel(f"{row_spec['dataset']}\n({row_spec['judge_type']})", fontsize=17, labelpad=10)
            
            # Set x-axis labels only for the bottom row
            if row_idx == 3:
                ax.set_xlabel('Exploration Round', fontsize=17)
            
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='lower right', fontsize=17)

            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for suptitle

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"main_figure_4x5_{metric_name}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_4x4_main_figure(all_data: List[Dict[str, Any]], metric_name: str = "best_so_far", save_dir: str = None):
    """
    Creates the specific 4x4 visualization for the main paper text.
    - 4 rows: Alpaca/Pairwise, Alpaca/Pointwise, ArenaHard/Pairwise, ArenaHard/Pointwise
    - 4 columns: o3-mini, gemini-2.5-flash, llama-70b, qwen3-235b
    """
    # --- Configuration for the 4x4 Grid ---
    # Define the exact models and their partial names for matching
    MODELS_ORDER = {
        'o3-mini': 'o3-mini',
        'Gemini-2.5-Flash': 'gemini-2.5-flash',
        'Llama3-70B': 'llama-3.3-70b-instruct',
        'Qwen3-235b': 'qwen3-235b-a22b-2507'
    }

    # Define the configuration for each row
    ROW_CONFIG = [
        {'dataset': 'AlpacaEval', 'judge_type': 'pointwise'},
        {'dataset': 'AlpacaEval', 'judge_type': 'alpaca_eval'},
        {'dataset': 'ArenaHard', 'judge_type': 'pointwise'},
        {'dataset': 'ArenaHard', 'judge_type': 'arena_hard_auto'},
    ]
    
    # Define consistent styles for each attack strategy
    STRATEGY_STYLES = {
        'ucb': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'BITE (ours)'},
        'random': {'color': '#2ca02c', 'marker': '^', 'linestyle': '--', 'label': 'Random'},
        'simple_rewrite_improve': {'color': '#9467bd', 'marker': 'v', 'linestyle': ':', 'label': 'Holistic Rewrite'},
    }

    # Create the 4x4 subplot grid
    # sharey='row' is crucial for making plots in the same row comparable
    fig, axes = plt.subplots(4, 4, figsize=(20, 18), sharex=True, sharey='row')
    # fig.suptitle('Attack Performance Across Judges, Datasets, and Evaluation Types', fontsize=20, y=0.97)

    # --- Main Loop to Populate the Grid ---
    for row_idx, row_spec in enumerate(ROW_CONFIG):
        for col_idx, (model_display_name, model_file_str) in enumerate(MODELS_ORDER.items()):
            ax = axes[row_idx, col_idx]

            # 1. Filter data for the current subplot
            subplot_data = [
                item for item in all_data
                if item['metadata']['dataset'] == row_spec['dataset'] and
                   item['metadata']['judge_type'] == row_spec['judge_type'] and
                   model_file_str in item['metadata']['judge_model']
            ]

            # 2. Group the filtered data by strategy
            strategies_in_subplot = defaultdict(list)
            for item in subplot_data:
                strategy = item['metadata']['strategy']
                if strategy in STRATEGY_STYLES:
                    strategies_in_subplot[strategy].append(item['metrics'])
                else:
                    # Debug: print unknown strategies
                    print(f"Debug: Unknown strategy '{strategy}' for {model_display_name}, {row_spec}")

            # 3. Plot each strategy's learning curve
            if not strategies_in_subplot:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=17, alpha=0.5)
                print(f"Debug: No data for {model_display_name}, {row_spec}. Found {len(subplot_data)} items but none matched strategy styles.")

            for strategy, metrics_list in strategies_in_subplot.items():
                all_runs = [m[metric_name] for m in metrics_list if metric_name in m and m[metric_name]]
                if not all_runs:
                    continue

                # Aggregate metrics across runs (mean and std)
                max_len = max(len(run) for run in all_runs)
                mean_curve = [np.mean([run[i] for run in all_runs if i < len(run)]) for i in range(max_len)]
                std_curve = [np.std([run[i] for run in all_runs if i < len(run)]) for i in range(max_len)]
                rounds = np.arange(1, len(mean_curve) + 1)
                
                style = STRATEGY_STYLES[strategy]
                # ax.plot(rounds, mean_curve, label=f"{style['label']} (n={len(all_runs)})", **{k: v for k, v in style.items() if k != 'label'})
                ax.plot(rounds, mean_curve, label=f"{style['label']}", **{k: v for k, v in style.items() if k != 'label'})
                ax.fill_between(rounds, np.array(mean_curve) - np.array(std_curve), np.array(mean_curve) + np.array(std_curve), color=style['color'], alpha=0.15)

            # --- 4. Formatting and Labeling ---
            # Set model titles only for the top row
            if row_idx == 0:
                ax.set_title(model_display_name, fontsize=20, pad=10)

            # Set y-axis labels only for the leftmost column
            if col_idx == 0:
                ax.set_ylabel(f"{row_spec['dataset']}\n({row_spec['judge_type']})", fontsize=17, labelpad=10)
            
            # Set x-axis labels only for the bottom row
            if row_idx == 3:
                ax.set_xlabel('Exploration Round', fontsize=17)
            
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='lower right', fontsize=17)

            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for suptitle

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"main_figure_4x4_{metric_name}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate the main 4x4 figure for the paper.")
    parser.add_argument("--metrics_dir", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/metrics", help="Directory containing all metrics JSON files.")
    parser.add_argument("--save_dir", type=str, default="./plots", help="Directory to save the final plot.")
    parser.add_argument("--filter", type=str, help="Filter criteria in format 'key1=value1,key2=value2'")
    parser.add_argument("--exclude", type=str, help="Exclude criteria in format 'key1=value1,key2=value2'", default="llm_agent_name=gpt-4.1-nano,strategy=simple_rewrite_holistic,judge_backbone=gpt-5")
    parser.add_argument("--report_metric", type=str, default="best_so_far", help="Metric to report", choices=["best_so_far", "pool_mean", "replacement_ratio", "lift_per_1k_tokens", "ci_width", "ucb_gap"])
    args = parser.parse_args()

    # Initialize MetricsLoader
    try:
        loader = MetricsLoader(args.metrics_dir)
        print(f"Found {len(loader.available_files)} total metrics files.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Apply filtering if provided
    loader.filter_and_exclude_by_string(args.filter, args.exclude)
    print(f"After filtering: {len(loader.available_files)} files remaining.")

    # Load all metrics
    all_loaded_metrics = loader.load_all_metrics()
    if not all_loaded_metrics:
        print("No valid metrics files found after filtering.")
        return

    print(f"Successfully loaded {len(all_loaded_metrics)} valid experiment files.")

    # Debug: Print available strategies, datasets, and judge types
    strategies = set()
    datasets = set()
    judge_types = set()
    judge_models = set()
    
    for loaded_metrics in all_loaded_metrics:
        strategies.add(loaded_metrics.metadata.strategy)
        datasets.add(loaded_metrics.metadata.dataset_name)
        judge_types.add(loaded_metrics.metadata.judge_type)
        judge_models.add(loaded_metrics.metadata.judge_backbone)
    
    print(f"Debug: Available strategies: {sorted(strategies)}")
    print(f"Debug: Available datasets: {sorted(datasets)}")
    print(f"Debug: Available judge types: {sorted(judge_types)}")
    print(f"Debug: Available judge models: {sorted(judge_models)}")

    # Convert to the format expected by plot_4x4_main_figure
    all_data = []
    for loaded_metrics in all_loaded_metrics:
        # Convert metadata to dictionary format expected by plotting function
        metadata_dict = {
            'strategy': loaded_metrics.metadata.strategy,
            'dataset': loaded_metrics.metadata.dataset_name,
            'judge_type': loaded_metrics.metadata.judge_type,
            'judge_model': loaded_metrics.metadata.judge_backbone,
            'llm_agent': loaded_metrics.metadata.llm_agent_name,
            'response_model': loaded_metrics.metadata.response_model_name,
            'budget': loaded_metrics.metadata.budget,
            'pool_size': loaded_metrics.metadata.pool_size,
        }
        
        # Convert aggregated metrics to the format expected by plotting function
        metrics_dict = {
            'best_so_far': loaded_metrics.aggregated_metrics.best_so_far,
            'pool_mean': loaded_metrics.aggregated_metrics.pool_mean,
            'replacement_ratio': loaded_metrics.aggregated_metrics.replacement_ratio,
            'lift_per_1k_tokens': loaded_metrics.aggregated_metrics.lift_per_1k_tokens,
            'ci_width': loaded_metrics.aggregated_metrics.ci_width,
            'ucb_gap': loaded_metrics.aggregated_metrics.ucb_gap,
        }
        
        all_data.append({'metadata': metadata_dict, 'metrics': metrics_dict})

    # Generate the specific 4x4 plot
    # plot_4x4_main_figure(all_data, args.report_metric, args.save_dir)
    plot_4x5_main_figure(all_data, args.report_metric, args.save_dir)


if __name__ == "__main__":
    main()