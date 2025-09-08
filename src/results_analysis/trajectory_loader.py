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

import argparse
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


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
    
    def exclude_trajectories(self, 
                             trajectories: List[LoadedTrajectory],
                             **filters) -> List[LoadedTrajectory]:
        """
        Exclude trajectories by metadata criteria.
        """
        return [traj for traj in trajectories if not should_exclude_trajectory(traj, filters)]
    
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
                            ) -> List[LoadedTrajectory]:
    """
    Convenience function to load all trajectories from a directory.
    
    Args:
        directory: Directory containing trajectory files
        exclude_patterns: Patterns to exclude from loading (filename-based)
        filter_criteria: Metadata criteria for inclusion (e.g., {'strategy': ['ucb', 'random']})
        exclude_criteria: Metadata criteria for exclusion (e.g., {'dataset_name': ['AlpacaEval']})
        
    Returns:
        List of LoadedTrajectory objects
    """
    loader = TrajectoryLoader(directory)
    
    # Default exclude patterns
    if exclude_patterns is None:
        exclude_patterns = ['warmup', 'init_ucb', 'init_linucb', 'warmup_summary']
    
    trajectories = loader.load_trajectories(exclude_patterns=exclude_patterns)
    original_count = len(trajectories)
    print(f"Loaded {original_count} trajectory files")

    # Apply advanced filter criteria (AND logic - all criteria must match)
    if filter_criteria:
        original_count = len(trajectories)
        print(f"Filtering trajectories with: {filter_criteria}")
        trajectories = [traj for traj in trajectories if should_include_trajectory(traj, filter_criteria)]
        filtered_count = original_count - len(trajectories)
        print(f"Filter removed {filtered_count} files, {len(trajectories)} remaining")
    
    # Apply exclusion criteria (OR logic - any criteria match means exclude)
    if exclude_criteria:
        original_count = len(trajectories)
        print(f"Excluding trajectories with: {exclude_criteria}")
        trajectories = [traj for traj in trajectories if not should_exclude_trajectory(traj, exclude_criteria)]
        excluded_count = original_count - len(trajectories)
        print(f"Excluded {excluded_count} files, {len(trajectories)} remaining")
    
    print(f"Loaded {len(trajectories)} trajectory files")
    
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


# def load_trajectories_from_directory(directory: str, filter_criteria: Optional[Dict[str, List[str]]] = None, exclude_criteria: Optional[Dict[str, List[str]]] = None, show_summary: bool = True) -> List[LoadedTrajectory]:
    
#     # Parse filter and exclude criteria
#     filter_criteria = parse_filter_criteria(filter_criteria)
#     exclude_criteria = parse_exclude_criteria(exclude_criteria)
    
#     if filter_criteria:
#         print(f"Filtering trajectories with: {filter_criteria}")
#     if exclude_criteria:
#         print(f"Excluding trajectories with: {exclude_criteria}")
    
#     # Load trajectories
#     trajectories = load_trajectory_directory(directory)
    
#     # Apply advanced filter criteria (AND logic - all criteria must match)
#     if filter_criteria:
#         original_count = len(trajectories)
#         trajectories = [traj for traj in trajectories if should_include_trajectory(traj, filter_criteria)]
#         filtered_count = original_count - len(trajectories)
#         print(f"Filter removed {filtered_count} files, {len(trajectories)} remaining")
    
#     # Apply exclusion criteria (OR logic - any criteria match means exclude)
#     if exclude_criteria:
#         original_count = len(trajectories)
#         trajectories = [traj for traj in trajectories if not should_exclude_trajectory(traj, exclude_criteria)]
#         excluded_count = original_count - len(trajectories)
#         print(f"Excluded {excluded_count} files, {len(trajectories)} remaining")
    
#     print(f"Loaded {len(trajectories)} trajectory files")


def show_summary(trajectories: List[LoadedTrajectory]):
    print("\n=== SUMMARY ===")
    for traj in trajectories:
        print(f"File: {os.path.basename(traj.metadata.file_path)}")
        print(f"  Strategy: {traj.metadata.strategy}")
        print(f"  Dataset: {traj.metadata.dataset_name}")
        print(f"  Questions: {len(traj.trajectories)}")
        print(f"  Mean final score: {sum(traj.get_final_scores()) / len(traj.trajectories):.3f}")
        print(f"  Mean improvement: {sum(traj.get_improvements()) / len(traj.trajectories):.3f}")
        print()


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
    
    args = parser.parse_args()

    directory = args.directory
    filter_criteria = args.filter
    exclude_criteria = args.exclude

    # Parse filter and exclude criteria
    filter_criteria_parsed = parse_filter_criteria(filter_criteria)
    exclude_criteria_parsed = parse_exclude_criteria(exclude_criteria)

    # load_trajectories_from_directory(directory, filter_criteria, exclude_criteria, show_summary)
    trajectories = load_trajectory_directory(directory, filter_criteria=filter_criteria_parsed, exclude_criteria=exclude_criteria_parsed)

    