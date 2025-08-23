import json
import os
import glob
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import re
import argparse

logger = logging.getLogger(__name__)

from src.results_analysis.trajectory_loader import TrajectoryLoader, load_trajectory_directory, parse_filter_criteria, parse_exclude_criteria, should_include_trajectory, should_exclude_trajectory

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
    parser.add_argument("--show_summary", action="store_true",
                       help="Show summary statistics")
    parser.add_argument("--compare", type=str, 
                       help="Compare two methods (e.g., --compare ucb,random). "
                            "This will load trajectories for both methods, match them by question, "
                            "calculate per-prompt win rates (win if higher score, tie if equal), "
                            "and print a summary including overall win rate for the first method vs second, "
                            "number of wins/ties/losses, and average score difference.")
    
    args = parser.parse_args()
    
    # Parse filter and exclude criteria
    filter_criteria = parse_filter_criteria(args.filter)
    exclude_criteria = parse_exclude_criteria(args.exclude)
    
    if filter_criteria:
        print(f"Filtering trajectories with: {filter_criteria}")
    if exclude_criteria:
        print(f"Excluding trajectories with: {exclude_criteria}")
    
    # Load trajectories
    trajectories = load_trajectory_directory(args.directory)
    
    # Apply advanced filter criteria (AND logic - all criteria must match)
    if filter_criteria:
        original_count = len(trajectories)
        trajectories = [traj for traj in trajectories if should_include_trajectory(traj, filter_criteria)]
        filtered_count = original_count - len(trajectories)
        print(f"Filter removed {filtered_count} files, {len(trajectories)} remaining")
    
    # Apply exclusion criteria (OR logic - any criteria match means exclude)
    if exclude_criteria:
        original_count = len(trajectories)
        trajectories = [traj for traj in trajectories if not should_exclude_trajectory(traj, exclude_criteria)]
        excluded_count = original_count - len(trajectories)
        print(f"Excluded {excluded_count} files, {len(trajectories)} remaining")
    
    print(f"Loaded {len(trajectories)} trajectory files")
    
    print("\n=== SUMMARY ===")
    for traj in trajectories:
        print(f"File: {os.path.basename(traj.metadata.file_path)}")
        print(f"  Strategy: {traj.metadata.strategy}")
        print(f"  Dataset: {traj.metadata.dataset_name}")
        print(f"  Questions: {len(traj.trajectories)}")
        print(f"  Mean final score: {sum(traj.get_final_scores()) / len(traj.trajectories):.3f}")
        print(f"  Mean improvement: {sum(traj.get_improvements()) / len(traj.trajectories):.3f}")
        print()

    if args.compare:
        method1, method2 = args.compare.split(',')
        
        # Load all trajectories first
        all_trajectories = load_trajectory_directory(args.directory)
        
        # Separate trajectories by strategy
        traj1 = [t for t in all_trajectories if t.metadata.strategy == method1]
        traj2 = [t for t in all_trajectories if t.metadata.strategy == method2]
        
        if not traj1:
            print(f"No trajectories found for method '{method1}'")
            exit(1)
        if not traj2:
            print(f"No trajectories found for method '{method2}'")
            exit(1)
        
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
        if len(common_settings) > 1:
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
