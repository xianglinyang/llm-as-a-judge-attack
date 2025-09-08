'''
This is used to analyze the win rate of the two methods.

## Comparing Methods

Use --compare to compare two methods **in the same experimental settings**:

```bash
python -m src.results_analysis.analysis.win_rate_analysis --compare random,simple_rewrite_improve
```

This will:
1. Find trajectories for both methods
2. Group them by experimental settings (dataset, judge, budget, etc.)
3. Only compare methods that have the same experimental setup
4. Calculate win rates per question across all matching settings

Example output:
```
Found 3 common experimental settings

Comparing ucb vs random across 3 experimental settings
Total common questions: 150
Wins/Ties/Losses (for ucb): 90/30/30
Win rate for ucb: 75.00%
Average score difference (ucb - random): 1.250

Breakdown by experimental setting:
  Setting 1: AlpacaEval | gpt-4 | pointwise | budget=20 | pool_size=3
    Questions: 50 | W/T/L: 30/10/10 | Win rate: 75.0%
  Setting 2: UltraFeedback | gpt-4 | pointwise | budget=20 | pool_size=3  
    Questions: 50 | W/T/L: 30/10/10 | Win rate: 75.0%
  Setting 3: AlpacaEval | gpt-3.5-turbo | pointwise | budget=10 | pool_size=2
    Questions: 50 | W/T/L: 30/10/10 | Win rate: 75.0%
```

'''
import os
import logging
import argparse

logger = logging.getLogger(__name__)

from src.results_analysis.results_loader.trajectory_loader import load_trajectory_directory, parse_filter_criteria, parse_exclude_criteria

# Group trajectories by experimental settings (excluding strategy)
def get_setting_key(metadata):
    return (
        metadata.dataset_name,
        metadata.judge_backbone, 
        metadata.judge_type,
        metadata.llm_agent_name,
        metadata.response_model_name,
        metadata.baseline_response_model_name,
        metadata.budget,
        metadata.pool_size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and inspect trajectory files")
    parser.add_argument("--directory", type=str,
                       help="Directory containing trajectory files",
                       default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories")
    parser.add_argument("--filter", type=str, 
                       help="Include only files matching criteria (format: 'key1=value1,key2=value2'). "
                            "Example: --filter 'strategy=ucb,dataset_name=AlpacaEval'")
    parser.add_argument("--exclude", type=str, 
                       help="Exclude files matching criteria (format: 'key1=value1,key2=value2'). "
                            "Example: --exclude 'strategy=random,judge_backbone=gpt-3.5'")
    parser.add_argument("--compare", type=str, required=True,
                       help="Compare two methods (e.g., --compare ucb,random). "
                            "This will load trajectories for both methods, match them by question, "
                            "calculate per-prompt win rates (win if higher score, tie if equal), "
                            "and print a summary including overall win rate for the first method vs second, "
                            "number of wins/ties/losses, and average score difference.")
    parser.add_argument("--show_breakdown", action="store_true",
                       help="Show breakdown by experimental setting")
    
    args = parser.parse_args()

    directory = args.directory
    filter_criteria = args.filter
    exclude_criteria = args.exclude

    # ----- Load trajectories -----

    # Parse filter and exclude criteria
    filter_criteria = parse_filter_criteria(args.filter)
    exclude_criteria = parse_exclude_criteria(args.exclude)
    
    if filter_criteria:
        print(f"Filtering trajectories with: {filter_criteria}")
    if exclude_criteria:
        print(f"Excluding trajectories with: {exclude_criteria}")
    
    # Load trajectories
    all_trajectories = load_trajectory_directory(directory)

    # ----- Compare methods -----
    assert args.compare is not None, "Please specify two methods to compare"
    
    method1, method2 = args.compare.split(',')
    
    # Separate trajectories by strategy
    traj1 = [t for t in all_trajectories if t.metadata.strategy == method1]
    traj2 = [t for t in all_trajectories if t.metadata.strategy == method2]
    
    if not traj1:
        print(f"No trajectories found for method '{method1}'")
        exit(1)
    if not traj2:
        print(f"No trajectories found for method '{method2}'")
        exit(1)
    
    
    # Group method1 trajectories by settings
    method1_by_setting = {}
    for traj in traj1:
        key = get_setting_key(traj.metadata)
        if key not in method1_by_setting:
            method1_by_setting[key] = []
        method1_by_setting[key].append(traj)
    
    # Group method2 trajectories by settings
    method2_by_setting = {}
    for traj in traj2:
        key = get_setting_key(traj.metadata)
        if key not in method2_by_setting:
            method2_by_setting[key] = []
        method2_by_setting[key].append(traj)
    
    # Find common settings
    common_settings = set(method1_by_setting.keys()) & set(method2_by_setting.keys())
    
    if not common_settings:
        print(f"No common experimental settings found between {method1} and {method2}")
        print(f"Available settings for {method1}: {len(method1_by_setting)}")
        print(f"Available settings for {method2}: {len(method2_by_setting)}")
        exit(1)
    
    print(f"Found {len(common_settings)} common experimental settings")
    
    # Aggregate scores across all common settings
    all_wins = 0
    all_ties = 0
    all_losses = 0
    all_score_diffs = []
    total_common_questions = 0
    
    for setting_key in common_settings:
        # Get trajectories for this setting
        setting_traj1 = method1_by_setting[setting_key]
        setting_traj2 = method2_by_setting[setting_key]
        
        # Aggregate questions and scores for this setting
        method1_scores = {}
        for traj in setting_traj1:
            for item in traj.trajectories:
                if item.question in method1_scores:
                    # Take max score for duplicates within same setting
                    method1_scores[item.question] = max(method1_scores[item.question], item.final_score)
                else:
                    method1_scores[item.question] = item.final_score
        
        method2_scores = {}
        for traj in setting_traj2:
            for item in traj.trajectories:
                if item.question in method2_scores:
                    method2_scores[item.question] = max(method2_scores[item.question], item.final_score)
                else:
                    method2_scores[item.question] = item.final_score
        
        # Find common questions for this setting
        common_questions = set(method1_scores.keys()) & set(method2_scores.keys())
        total_common_questions += len(common_questions)
        
        # Compare for this setting
        for q in common_questions:
            s1 = float(method1_scores[q])
            s2 = float(method2_scores[q])
            diff = s1 - s2
            all_score_diffs.append(diff)
            
            if s1 > s2:
                all_wins += 1
            elif s1 < s2:
                all_losses += 1
            else:
                all_ties += 1
    
    # Print summary
    print(f"\nComparing {method1} vs {method2} across {len(common_settings)} experimental settings")
    print(f"Total common questions: {total_common_questions}")
    print(f"Wins/Ties/Losses (for {method1}): {all_wins}/{all_ties}/{all_losses}")
    
    total_comparisons = all_wins + all_ties + all_losses
    if total_comparisons > 0:
        # unbeaten rate
        unbeaten_rate = ((all_wins + all_ties) / total_comparisons) * 100
        print(f"Unbeaten rate for {method1}: {unbeaten_rate:.2f}%")
    else:
        print("All comparisons are ties")
    
    avg_diff = sum(all_score_diffs) / len(all_score_diffs) if all_score_diffs else 0.0
    print(f"Average score difference ({method1} - {method2}): {avg_diff:.3f}")
    
    # Optional: Show breakdown by setting
    if len(common_settings) > 1 and args.show_breakdown:
        print(f"\nBreakdown by experimental setting:")
        for i, setting_key in enumerate(sorted(common_settings)):
            dataset, judge, judge_type, llm_agent, response_model, baseline_model, budget, pool_size = setting_key
            print(f"  Setting {i+1}: {dataset} | {judge} | {judge_type} | budget={budget} | pool_size={pool_size}")
            
            # Recalculate for this specific setting for breakdown
            setting_traj1 = method1_by_setting[setting_key]
            setting_traj2 = method2_by_setting[setting_key]
            
            method1_scores = {}
            for traj in setting_traj1:
                for item in traj.trajectories:
                    method1_scores[item.question] = max(method1_scores.get(item.question, 0), item.final_score)
            
            method2_scores = {}
            for traj in setting_traj2:
                for item in traj.trajectories:
                    method2_scores[item.question] = max(method2_scores.get(item.question, 0), item.final_score)
            
            common_q = set(method1_scores.keys()) & set(method2_scores.keys())
            wins = sum(1 for q in common_q if method1_scores[q] > method2_scores[q])
            ties = sum(1 for q in common_q if method1_scores[q] == method2_scores[q])
            losses = sum(1 for q in common_q if method1_scores[q] < method2_scores[q])
            
            if wins + losses > 0:
                unbeaten_rate = ((wins + ties) / (wins + losses + ties)) * 100
                print(f"    Questions: {len(common_q)} | W/T/L: {wins}/{ties}/{losses} | Unbeaten: {unbeaten_rate:.1f}%")
