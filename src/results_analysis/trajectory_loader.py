#!/usr/bin/env python3
"""
Trajectory Loader for LLM-as-a-Judge Attack Exploration Results

This module provides functionality to load and parse trajectory files saved by
various exploration strategies (UCB, random, simple rewrite, etc.). 

The trajectory format follows the structure:
{
    "strategy": str,
    "judge_type": str,
    "dataset_name": str,
    "judge_backbone": str,
    "llm_agent_name": str,
    "response_model_name": str,
    "baseline_response_model_name": str,
    "budget": int,
    "pool_size": int,
    "eval_num": int,
    "timestamp": str,
    "time_taken": float,
    ... (other metadata) ...
    "trajectories": [
        {
            "question": str,
            "score": float,
            "answer": str,
            "explanation": str,
            "origin": str,
            "tokens": int,
            "category": str,  # added during saving
            "history": [
                (score: float, explanation: str, answer: str, strategy_or_origin: str),
                ...
            ]
        },
        ...
    ]
}
"""

import json
import os
import glob
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import re


logger = logging.getLogger(__name__)


@dataclass
class TrajectoryMetadata:
    """Metadata for a trajectory file."""
    strategy: str
    judge_type: str
    dataset_name: str
    judge_backbone: str
    llm_agent_name: str
    response_model_name: str
    baseline_response_model_name: Optional[str]
    budget: int
    pool_size: int
    eval_num: int
    timestamp: str
    time_taken: float
    file_path: str
    
    # Optional fields that may be present
    alpha: Optional[float] = None
    lambda_reg: Optional[float] = None
    reward_type: Optional[str] = None
    answer_position: Optional[str] = None
    template_name: Optional[str] = None  # for simple_rewrite strategies
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], file_path: str) -> 'TrajectoryMetadata':
        """Create metadata from loaded trajectory data."""
        return cls(
            strategy=data.get('strategy', data.get('test_mode', 'unknown')),
            judge_type=data.get('judge_type', 'unknown'),
            dataset_name=data.get('dataset_name', 'unknown'),
            judge_backbone=data.get('judge_backbone', data.get('judge_model_name', 'unknown')),
            llm_agent_name=data.get('llm_agent_name', 'unknown'),
            response_model_name=data.get('response_model_name', 'unknown'),
            baseline_response_model_name=data.get('baseline_response_model_name'),
            budget=data.get('budget', data.get('Budget', 0)),
            pool_size=data.get('pool_size', 0),
            eval_num=data.get('eval_num', 0),
            timestamp=data.get('timestamp', 'unknown'),
            time_taken=data.get('time_taken', 0.0),
            file_path=file_path,
            alpha=data.get('alpha'),
            lambda_reg=data.get('lambda_reg'),
            reward_type=data.get('reward_type'),
            answer_position=data.get('answer_position'),
            template_name=data.get('template_name')
        )


@dataclass 
class TrajectoryStep:
    """A single step in a trajectory."""
    score: float
    explanation: str
    answer: str
    strategy_or_origin: str
    
    @classmethod
    def from_tuple(cls, step_tuple: Tuple) -> 'TrajectoryStep':
        """Create from tuple format (score, explanation, answer, strategy)."""
        if len(step_tuple) >= 4:
            return cls(
                score=float(step_tuple[0]),
                explanation=str(step_tuple[1]),
                answer=str(step_tuple[2]),
                strategy_or_origin=str(step_tuple[3])
            )
        else:
            raise ValueError(f"Invalid step tuple format: {step_tuple}")


@dataclass
class TrajectoryItem:
    """A single trajectory for one question."""
    question: str
    final_score: float
    final_answer: str
    final_explanation: str
    origin: str
    tokens: int
    category: str
    history: List[TrajectoryStep]
    
    @property
    def initial_score(self) -> float:
        """Get the initial score from history."""
        return self.history[0].score if self.history else 0.0
    
    @property
    def initial_answer(self) -> str:
        """Get the initial answer from history."""
        return self.history[0].answer if self.history else ""
    
    @property
    def exploration_length(self) -> int:
        """Get the number of exploration steps (excluding initial)."""
        return len(self.history) - 1 if self.history else 0
    
    @property
    def improvement(self) -> float:
        """Get the score improvement from initial to final."""
        return self.final_score - self.initial_score
    
    @property
    def strategies_used(self) -> List[str]:
        """Get list of strategies used during exploration."""
        return [step.strategy_or_origin for step in self.history[1:]]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrajectoryItem':
        """Create from dictionary format."""
        # Parse history steps
        history = []
        for step_data in data.get('history', []):
            if isinstance(step_data, (list, tuple)):
                history.append(TrajectoryStep.from_tuple(step_data))
            else:
                # Handle alternative formats if needed
                logger.warning(f"Unexpected history step format: {step_data}")
        
        return cls(
            question=data.get('question', ''),
            final_score=float(data.get('score', 0.0)),
            final_answer=data.get('answer', ''),
            final_explanation=data.get('explanation', ''),
            origin=data.get('origin', 'unknown'),
            tokens=int(data.get('tokens', 0)),
            category=data.get('category', 'unknown'),
            history=history
        )


@dataclass
class LoadedTrajectory:
    """Complete loaded trajectory data."""
    metadata: TrajectoryMetadata
    trajectories: List[TrajectoryItem]
    
    def __len__(self) -> int:
        """Number of trajectories (questions)."""
        return len(self.trajectories)
    
    def get_trajectory_by_category(self, category: str) -> List[TrajectoryItem]:
        """Get all trajectories for a specific category."""
        return [t for t in self.trajectories if t.category == category]
    
    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        return list(set(t.category for t in self.trajectories))
    
    def get_final_scores(self) -> List[float]:
        """Get final scores for all trajectories."""
        return [t.final_score for t in self.trajectories]
    
    def get_initial_scores(self) -> List[float]:
        """Get initial scores for all trajectories.""" 
        return [t.initial_score for t in self.trajectories]
    
    def get_improvements(self) -> List[float]:
        """Get score improvements for all trajectories."""
        return [t.improvement for t in self.trajectories]
    
    def get_exploration_lengths(self) -> List[int]:
        """Get exploration lengths for all trajectories."""
        return [t.exploration_length for t in self.trajectories]


class TrajectoryLoader:
    """Main class for loading trajectory files."""
    
    def __init__(self, base_dir: str):
        """
        Initialize trajectory loader.
        
        Args:
            base_dir: Base directory containing trajectory files
        """
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise ValueError(f"Directory does not exist: {base_dir}")
    
    def find_trajectory_files(self, 
                             pattern: str = "*.json", 
                             recursive: bool = True,
                             exclude_patterns: Optional[List[str]] = None) -> List[str]:
        """
        Find trajectory files matching pattern.
        
        Args:
            pattern: File pattern to match
            recursive: Whether to search recursively
            exclude_patterns: Patterns to exclude
            
        Returns:
            List of file paths
        """
        if recursive:
            files = list(self.base_dir.rglob(pattern))
        else:
            files = list(self.base_dir.glob(pattern))
        
        file_paths = [str(f) for f in files]
        
        # Apply exclusion patterns
        if exclude_patterns:
            filtered_paths = []
            for path in file_paths:
                filename = os.path.basename(path).lower()
                should_exclude = any(
                    exclude_pattern.lower() in filename 
                    for exclude_pattern in exclude_patterns
                )
                if not should_exclude:
                    filtered_paths.append(path)
            file_paths = filtered_paths
        
        return sorted(file_paths)
    
    def load_trajectory_file(self, file_path: str) -> Optional[LoadedTrajectory]:
        """
        Load a single trajectory file.
        
        Args:
            file_path: Path to trajectory file
            
        Returns:
            LoadedTrajectory object or None if loading failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if this is a trajectory file (has 'trajectories' key)
            if 'trajectories' not in data:
                logger.warning(f"File {file_path} does not contain trajectory data")
                return None
            
            # Extract metadata
            metadata = TrajectoryMetadata.from_dict(data, file_path)
            
            # Parse trajectories
            trajectories = []
            for traj_data in data['trajectories']:
                try:
                    trajectory_item = TrajectoryItem.from_dict(traj_data)
                    trajectories.append(trajectory_item)
                except Exception as e:
                    logger.warning(f"Failed to parse trajectory in {file_path}: {e}")
                    continue
            
            if not trajectories:
                logger.warning(f"No valid trajectories found in {file_path}")
                return None
            
            return LoadedTrajectory(metadata=metadata, trajectories=trajectories)
            
        except Exception as e:
            logger.error(f"Failed to load trajectory file {file_path}: {e}")
            return None
    
    def load_trajectories(self, 
                         pattern: str = "*.json",
                         recursive: bool = True,
                         exclude_patterns: Optional[List[str]] = None) -> List[LoadedTrajectory]:
        """
        Load all trajectory files matching criteria.
        
        Args:
            pattern: File pattern to match
            recursive: Whether to search recursively  
            exclude_patterns: Patterns to exclude (e.g., ['warmup', 'init'])
            
        Returns:
            List of LoadedTrajectory objects
        """
        file_paths = self.find_trajectory_files(pattern, recursive, exclude_patterns)
        
        loaded_trajectories = []
        for file_path in file_paths:
            trajectory = self.load_trajectory_file(file_path)
            if trajectory is not None:
                loaded_trajectories.append(trajectory)
        
        logger.info(f"Successfully loaded {len(loaded_trajectories)} trajectory files out of {len(file_paths)} found")
        return loaded_trajectories
    
    def filter_trajectories(self, 
                           trajectories: List[LoadedTrajectory],
                           **filters) -> List[LoadedTrajectory]:
        """
        Filter trajectories by metadata criteria.
        
        Args:
            trajectories: List of LoadedTrajectory objects
            **filters: Keyword arguments for filtering (e.g., strategy='ucb', dataset_name='AlpacaEval')
            
        Returns:
            Filtered list of LoadedTrajectory objects
        """
        filtered = []
        for traj in trajectories:
            include = True
            for key, value in filters.items():
                metadata_value = getattr(traj.metadata, key, None)
                if metadata_value != value:
                    include = False
                    break
            if include:
                filtered.append(traj)
        
        return filtered
    
    def group_trajectories_by(self, 
                             trajectories: List[LoadedTrajectory],
                             group_by: str) -> Dict[str, List[LoadedTrajectory]]:
        """
        Group trajectories by a metadata field.
        
        Args:
            trajectories: List of LoadedTrajectory objects
            group_by: Metadata field to group by
            
        Returns:
            Dictionary mapping group values to trajectory lists
        """
        groups = {}
        for traj in trajectories:
            group_value = getattr(traj.metadata, group_by, 'unknown')
            if group_value not in groups:
                groups[group_value] = []
            groups[group_value].append(traj)
        
        return groups


def load_trajectory_directory(directory: str, 
                            exclude_patterns: Optional[List[str]] = None,
                            filter_criteria: Optional[Dict[str, List[str]]] = None,
                            exclude_criteria: Optional[Dict[str, List[str]]] = None,
                            **filters) -> List[LoadedTrajectory]:
    """
    Convenience function to load all trajectories from a directory.
    
    Args:
        directory: Directory containing trajectory files
        exclude_patterns: Patterns to exclude from loading (filename-based)
        filter_criteria: Metadata criteria for inclusion (e.g., {'strategy': ['ucb', 'random']})
        exclude_criteria: Metadata criteria for exclusion (e.g., {'dataset_name': ['AlpacaEval']})
        **filters: Additional filters to apply (legacy format)
        
    Returns:
        List of LoadedTrajectory objects
    """
    loader = TrajectoryLoader(directory)
    
    # Default exclude patterns
    if exclude_patterns is None:
        exclude_patterns = ['warmup', 'init_ucb', 'init_linucb', 'warmup_summary']
    
    trajectories = loader.load_trajectories(exclude_patterns=exclude_patterns)
    
    # Apply metadata-based filter criteria (AND logic - all must match)
    if filter_criteria:
        trajectories = [traj for traj in trajectories if should_include_trajectory(traj, filter_criteria)]
    
    # Apply metadata-based exclusion criteria (OR logic - any match means exclude)
    if exclude_criteria:
        trajectories = [traj for traj in trajectories if not should_exclude_trajectory(traj, exclude_criteria)]
    
    # Apply legacy filters (for backward compatibility)
    if filters:
        trajectories = loader.filter_trajectories(trajectories, **filters)
    
    return trajectories


def load_single_trajectory(file_path: str) -> Optional[LoadedTrajectory]:
    """
    Convenience function to load a single trajectory file.
    
    Args:
        file_path: Path to trajectory file
        
    Returns:
        LoadedTrajectory object or None
    """
    directory = os.path.dirname(file_path)
    loader = TrajectoryLoader(directory)
    return loader.load_trajectory_file(file_path)


def parse_criteria(criteria_string: str, criteria_type: str = "criteria") -> Dict[str, List[str]]:
    """
    Parse criteria from 'key1=value1,key2=value2' format.
    
    Supports multiple values for the same key:
    'dataset_name=AlpacaEval,dataset_name=UltraFeedback' -> {'dataset_name': ['AlpacaEval', 'UltraFeedback']}
    
    Args:
        criteria_string: String in format 'key1=value1,key2=value2'
        criteria_type: Type of criteria for error messages ('filter', 'exclude', 'criteria')
        
    Returns:
        Dictionary mapping keys to lists of values
    """
    if not criteria_string:
        return {}
    
    criteria = {}
    pairs = criteria_string.split(',')
    
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)  # Split only on first =
            key = key.strip()
            value = value.strip()
            
            if key in criteria:
                criteria[key].append(value)
            else:
                criteria[key] = [value]
        else:
            print(f"Warning: Invalid {criteria_type} format '{pair}'. Expected 'key=value'")
    
    return criteria


def parse_exclude_criteria(exclude_string: str) -> Dict[str, List[str]]:
    """
    Parse exclude criteria from 'key1=value1,key2=value2' format.
    
    Args:
        exclude_string: String in format 'key1=value1,key2=value2'
        
    Returns:
        Dictionary mapping keys to lists of values to exclude
    """
    return parse_criteria(exclude_string, "exclude")


def parse_filter_criteria(filter_string: str) -> Dict[str, List[str]]:
    """
    Parse filter criteria from 'key1=value1,key2=value2' format.
    
    Args:
        filter_string: String in format 'key1=value1,key2=value2'
        
    Returns:
        Dictionary mapping keys to lists of values to include
    """
    return parse_criteria(filter_string, "filter")


def _get_metadata_value(trajectory: LoadedTrajectory, key: str) -> Optional[str]:
    """
    Get metadata value from trajectory, handling special case mappings.
    
    Args:
        trajectory: LoadedTrajectory object
        key: Metadata key to retrieve
        
    Returns:
        Metadata value as string or None if not found
    """
    # Get the metadata value (handle different possible attribute names)
    metadata_value = getattr(trajectory.metadata, key, None)
    
    # Handle special case mappings
    if metadata_value is None:
        if key == 'dataset' and hasattr(trajectory.metadata, 'dataset_name'):
            metadata_value = trajectory.metadata.dataset_name
        elif key == 'judge' and hasattr(trajectory.metadata, 'judge_backbone'):
            metadata_value = trajectory.metadata.judge_backbone
    
    return str(metadata_value) if metadata_value is not None else None


def _matches_criteria_values(metadata_value_str: str, criteria_values: List[str]) -> bool:
    """
    Check if a metadata value matches any of the criteria values.
    
    Args:
        metadata_value_str: Metadata value as lowercase string
        criteria_values: List of values to match against
        
    Returns:
        True if any criteria value matches, False otherwise
    """
    for criteria_value in criteria_values:
        criteria_value_lower = criteria_value.lower()
        if metadata_value_str == criteria_value_lower or criteria_value_lower in metadata_value_str:
            return True
    return False


def should_exclude_trajectory(trajectory: LoadedTrajectory, exclude_criteria: Dict[str, List[str]]) -> bool:
    """
    Check if a trajectory should be excluded based on criteria.
    
    Args:
        trajectory: LoadedTrajectory object to check
        exclude_criteria: Dictionary of exclusion criteria
        
    Returns:
        True if trajectory should be excluded, False otherwise
    """
    if not exclude_criteria:
        return False
    
    for key, exclude_values in exclude_criteria.items():
        metadata_value = _get_metadata_value(trajectory, key)
        
        if metadata_value is not None:
            metadata_value_str = metadata_value.lower()
            if _matches_criteria_values(metadata_value_str, exclude_values):
                return True
    
    return False


def should_include_trajectory(trajectory: LoadedTrajectory, filter_criteria: Dict[str, List[str]]) -> bool:
    """
    Check if a trajectory should be included based on filter criteria.
    
    Args:
        trajectory: LoadedTrajectory object to check
        filter_criteria: Dictionary of filter criteria
        
    Returns:
        True if trajectory should be included, False otherwise
    """
    if not filter_criteria:
        return True  # Include all if no filter criteria
    
    # For inclusion, ALL criteria must be satisfied (AND logic across different keys)
    for key, filter_values in filter_criteria.items():
        metadata_value = _get_metadata_value(trajectory, key)
        
        if metadata_value is None:
            return False  # If metadata is missing, exclude
        
        metadata_value_str = metadata_value.lower()
        if not _matches_criteria_values(metadata_value_str, filter_values):
            return False  # If this criteria doesn't match, exclude
    
    return True  # All criteria matched


if __name__ == "__main__":
    # Example usage
    import argparse
    
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
    
    if args.show_summary:
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
