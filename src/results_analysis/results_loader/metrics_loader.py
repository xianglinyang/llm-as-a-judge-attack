#!/usr/bin/env python3
"""
Metrics Loader for LLM-as-a-Judge Attack Exploration Results

This module provides functionality to load and parse exploration metrics files 
from various exploration strategies (UCB, random, simple rewrite, etc.). 

The metrics format follows the structure:
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
    "timestamp": str,
    "time_taken": float,
    ... (other metadata) ...
    "metrics": [
        {
            "best_so_far": [float, ...],
            "pool_mean": [float, ...],
            "replacement_ratio": [float, ...],
            "lift_per_1k_tokens": [float, ...],
            "ci_width": [float, ...],
            "ucb_gap": [float, ...]
        },
        ...
    ]
}

Usage:
    loader = MetricsLoader("/path/to/metrics")
    all_metrics = loader.load_all_metrics()
    ucb_metrics = loader.load_strategy_metrics("ucb")
    aggregated = loader.get_aggregated_metrics("ucb")
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from src.results_analysis.results_loader.utils import parse_exclude_criteria, parse_filter_criteria

logger = logging.getLogger(__name__)


@dataclass
class ExplorationMetadata:
    """Metadata for exploration metrics."""
    strategy: str
    judge_type: str
    dataset_name: str
    judge_backbone: str
    llm_agent_name: str
    response_model_name: str
    baseline_response_model_name: Optional[str]
    budget: int
    pool_size: int
    timestamp: str
    time_taken: float
    file_path: str
    
    # Optional fields that may be present
    alpha: Optional[float] = None
    lambda_reg: Optional[float] = None
    reward_type: Optional[str] = None
    answer_position: Optional[str] = None
    template_name: Optional[str] = None  # for simple_rewrite strategies
    eval_num: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], file_path: str) -> 'ExplorationMetadata':
        """Create metadata from loaded exploration data."""
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
            timestamp=data.get('timestamp', 'unknown'),
            time_taken=data.get('time_taken', 0.0),
            file_path=file_path,
            alpha=data.get('alpha'),
            lambda_reg=data.get('lambda_reg'),
            reward_type=data.get('reward_type'),
            answer_position=data.get('answer_position'),
            template_name=data.get('template_name'),
            eval_num=data.get('eval_num')
        )


@dataclass
class QuestionMetrics:
    """Metrics for a single question across exploration rounds."""
    best_so_far: List[float]
    pool_mean: List[float]
    replacement_ratio: List[float]
    lift_per_1k_tokens: List[float]
    ci_width: List[float]
    ucb_gap: List[float]
    
    @property
    def num_rounds(self) -> int:
        """Get the number of exploration rounds."""
        return len(self.best_so_far)
    
    @property
    def final_score(self) -> float:
        """Get the final best score."""
        return self.best_so_far[-1] if self.best_so_far else 0.0
    
    @property
    def improvement(self) -> float:
        """Get the total improvement from initial to final score."""
        if len(self.best_so_far) >= 2:
            return self.best_so_far[-1] - self.best_so_far[0]
        return 0.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuestionMetrics':
        """Create from dictionary format."""
        return cls(
            best_so_far=data.get('best_so_far', []),
            pool_mean=data.get('pool_mean', []),
            replacement_ratio=data.get('replacement_ratio', []),
            lift_per_1k_tokens=data.get('lift_per_1k_tokens', []),
            ci_width=data.get('ci_width', []),
            ucb_gap=data.get('ucb_gap', [])
        )


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple questions."""
    best_so_far: List[float]
    pool_mean: List[float]
    replacement_ratio: List[float]
    lift_per_1k_tokens: List[float]
    ci_width: List[float]
    ucb_gap: List[float]
    stability: List[float]  # Coefficient of variation across questions
    
    @property
    def num_rounds(self) -> int:
        """Get the number of exploration rounds."""
        return len(self.best_so_far)
    
    @property
    def final_score(self) -> float:
        """Get the final aggregated score."""
        return self.best_so_far[-1] if self.best_so_far else 0.0
    
    @property
    def total_improvement(self) -> float:
        """Get the total improvement from initial to final score."""
        if len(self.best_so_far) >= 2:
            return self.best_so_far[-1] - self.best_so_far[0]
        return 0.0


@dataclass
class LoadedExplorationMetrics:
    """Complete loaded exploration metrics data."""
    metadata: ExplorationMetadata
    question_metrics: List[QuestionMetrics]
    aggregated_metrics: AggregatedMetrics
    
    def __len__(self) -> int:
        """Number of questions."""
        return len(self.question_metrics)
    
    def get_metric_by_round(self, metric_name: str, round_idx: int) -> List[float]:
        """Get a specific metric for all questions at a specific round."""
        values = []
        for question_metrics in self.question_metrics:
            metric_values = getattr(question_metrics, metric_name, [])
            if round_idx < len(metric_values):
                values.append(metric_values[round_idx])
        return values
    
    def get_final_scores(self) -> List[float]:
        """Get final scores for all questions."""
        return [qm.final_score for qm in self.question_metrics]
    
    def get_improvements(self) -> List[float]:
        """Get improvements for all questions."""
        return [qm.improvement for qm in self.question_metrics]


class MetricsLoader:
    """Main class for loading exploration metrics files."""
    
    def __init__(self, metrics_dir: str = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/metrics"):
        """Initialize the metrics loader.
        
        Args:
            metrics_dir: Directory containing metrics files
        """
        self.metrics_dir = Path(metrics_dir)
        if not self.metrics_dir.exists():
            raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")
        
        self.available_files = self._discover_metrics_files()
        logger.info(f"Discovered {len(self.available_files)} metrics files")
    
    def _discover_metrics_files(self, patterns: List[str] = ["*.json"]) -> List[Path]:
        """Discover available metrics files."""
        files = []
        
        for pattern in patterns:
            files.extend(self.metrics_dir.glob(pattern))
        
        # Filter out warmup and initialization files
        filtered_files = []
        for file in files:
            filename = file.name.lower()
            # Default skip files
            if any(skip_term in filename for skip_term in ['init_ucb_warmup', 'init_linucb_warmup']):
                continue
            if 'warmup' in filename or 'init_ucb' in filename or 'init_linucb' in filename:
                continue
            filtered_files.append(file)
        
        return sorted(filtered_files)
    
    def _matches_criteria(self, metadata: ExplorationMetadata, criteria: Dict[str, List[str]]) -> bool:
        """Check if metadata matches the given criteria.
        
        Args:
            metadata: ExplorationMetadata object to check
            criteria: Dictionary of criteria to match against
            
        Returns:
            True if metadata matches all criteria, False otherwise
        """
        for key, values in criteria.items():
            metadata_value = getattr(metadata, key, None)
            if metadata_value is None:
                return False
            
            metadata_value_str = str(metadata_value).lower()
            
            # Check if any of the criterion values match
            matches_any = False
            for value in values:
                value_str = value.lower()
                if value_str in metadata_value_str or metadata_value_str == value_str:
                    matches_any = True
                    break
            
            if not matches_any:
                return False
        
        return True
    
    def _should_exclude(self, metadata: ExplorationMetadata, exclude_criteria: Dict[str, List[str]]) -> bool:
        """Check if metadata should be excluded based on exclude criteria.
        
        Args:
            metadata: ExplorationMetadata object to check
            exclude_criteria: Dictionary of criteria for exclusion
            
        Returns:
            True if metadata should be excluded, False otherwise
        """
        for key, values in exclude_criteria.items():
            metadata_value = getattr(metadata, key, None)
            if metadata_value is None:
                continue
            
            metadata_value_str = str(metadata_value).lower()
            
            # Check if any of the exclude values match
            for value in values:
                value_str = value.lower()
                if value_str in metadata_value_str or metadata_value_str == value_str:
                    return True
        
        return False
    
    def filter_metrics_files(self, criteria: Dict[str, List[str]]) -> List[Path]:
        """Filter metrics files by criteria.
        
        Args:
            criteria: Dictionary of filter criteria
                     Format: {'key': ['value1', 'value2']}
                     Only files matching ALL keys will be included
                     
        Returns:
            List of filtered file paths
        """
        if not criteria:
            return self.available_files.copy()
        
        filtered_files = []
        
        for file_path in self.available_files:
            try:
                # Load metadata to check criteria
                with open(file_path, 'r') as f:
                    data = json.load(f)
                metadata = ExplorationMetadata.from_dict(data, str(file_path))
                
                if self._matches_criteria(metadata, criteria):
                    filtered_files.append(file_path)
                    
            except Exception as e:
                logger.warning(f"Error checking criteria for {file_path}: {e}")
        
        # Update available files
        self.available_files = filtered_files
        logger.info(f"Filtered to {len(filtered_files)} files based on criteria")
        return filtered_files
    
    def exclude_metrics_files(self, exclude_criteria: Dict[str, List[str]]) -> List[Path]:
        """Exclude metrics files by criteria.
        
        Args:
            exclude_criteria: Dictionary of exclusion criteria
                            Format: {'key': ['value1', 'value2']}
                            Files matching ANY of the criteria will be excluded
                            
        Returns:
            List of remaining file paths after exclusion
        """
        if not exclude_criteria:
            return self.available_files.copy()
        
        filtered_files = []
        
        for file_path in self.available_files:
            try:
                # Load metadata to check exclusion criteria
                with open(file_path, 'r') as f:
                    data = json.load(f)
                metadata = ExplorationMetadata.from_dict(data, str(file_path))
                
                if not self._should_exclude(metadata, exclude_criteria):
                    filtered_files.append(file_path)
                    
            except Exception as e:
                logger.warning(f"Error checking exclusion criteria for {file_path}: {e}")
                # Include file if we can't check criteria
                filtered_files.append(file_path)
        
        # Update available files
        self.available_files = filtered_files
        logger.info(f"Excluded files, {len(filtered_files)} files remaining")
        return filtered_files
    
    def filter_and_exclude_metrics_files(self, filter_criteria: Dict[str, List[str]], exclude_criteria: Dict[str, List[str]]) -> List[Path]:
        """Filter and exclude metrics files by criteria.
        
        Args:
            filter_criteria: Dictionary of filter criteria (include only matching)
            exclude_criteria: Dictionary of exclusion criteria (exclude matching)
            
        Returns:
            List of file paths after filtering and exclusion
        """
        filtered_files = []
        
        for file_path in self.available_files:
            try:
                # Load metadata to check criteria
                with open(file_path, 'r') as f:
                    data = json.load(f)
                metadata = ExplorationMetadata.from_dict(data, str(file_path))
                
                # Check filter criteria (must match if provided)
                if filter_criteria and not self._matches_criteria(metadata, filter_criteria):
                    continue
                
                # Check exclude criteria (must not match if provided)
                if exclude_criteria and self._should_exclude(metadata, exclude_criteria):
                    continue
                
                filtered_files.append(file_path)
                    
            except Exception as e:
                logger.warning(f"Error checking criteria for {file_path}: {e}")
        
        # Update available files
        self.available_files = filtered_files
        logger.info(f"Applied filter and exclude criteria, {len(filtered_files)} files remaining")
        return filtered_files
    
    def filter_by_string(self, criteria_string: str) -> List[Path]:
        """Filter metrics files using string criteria format.
        
        Args:
            criteria_string: String in format 'key1=value1,key2=value2'
            
        Returns:
            List of filtered file paths
        """
        criteria = parse_filter_criteria(criteria_string)
        return self.filter_metrics_files(criteria)
    
    def exclude_by_string(self, exclude_string: str) -> List[Path]:
        """Exclude metrics files using string criteria format.
        
        Args:
            exclude_string: String in format 'key1=value1,key2=value2'
            
        Returns:
            List of remaining file paths after exclusion
        """
        exclude_criteria = parse_exclude_criteria(exclude_string)
        return self.exclude_metrics_files(exclude_criteria)
    
    def filter_and_exclude_by_string(self, filter_string: str = "", exclude_string: str = "") -> List[Path]:
        """Filter and exclude metrics files using string criteria format.
        
        Args:
            filter_string: String in format 'key1=value1,key2=value2' for filtering
            exclude_string: String in format 'key1=value1,key2=value2' for exclusion
            
        Returns:
            List of file paths after filtering and exclusion
        """
        filter_criteria = parse_filter_criteria(filter_string) if filter_string else {}
        exclude_criteria = parse_exclude_criteria(exclude_string) if exclude_string else {}
        return self.filter_and_exclude_metrics_files(filter_criteria, exclude_criteria)
    
    def reset_available_files(self) -> List[Path]:
        """Reset available files to the original discovered set."""
        self.available_files = self._discover_metrics_files()
        logger.info(f"Reset to {len(self.available_files)} original files")
        return self.available_files
    
    def list_available_files(self) -> List[str]:
        """Get list of available metrics file names."""
        return [f.name for f in self.available_files]
    
    def list_strategies(self) -> Set[str]:
        """Get set of available strategies."""
        strategies = set()
        for file_path in self.available_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                strategy = data.get('strategy', data.get('test_mode', 'unknown'))
                if 'simple_rewrite' in strategy.lower():
                    template_name = data.get('template_name', '')
                    if template_name:
                        strategy = f"simple_rewrite_{template_name}"
                strategies.add(strategy)
            except Exception as e:
                logger.warning(f"Error reading strategy from {file_path}: {e}")
        return strategies
    
    def list_datasets(self) -> Set[str]:
        """Get set of available datasets."""
        datasets = set()
        for file_path in self.available_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                datasets.add(data.get('dataset_name', 'unknown'))
            except Exception as e:
                logger.warning(f"Error reading dataset from {file_path}: {e}")
        return datasets
    
    def list_judge_types(self) -> Set[str]:
        """Get set of available judge types."""
        judge_types = set()
        for file_path in self.available_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                judge_types.add(data.get('judge_type', 'unknown'))
            except Exception as e:
                logger.warning(f"Error reading judge_type from {file_path}: {e}")
        return judge_types
    
    def load_metrics_file(self, file_path: Union[str, Path]) -> LoadedExplorationMetrics:
        """Load metrics from a single file."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not file_path.is_absolute():
            file_path = self.metrics_dir / file_path
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract metadata
            metadata = ExplorationMetadata.from_dict(data, str(file_path))
            
            # Extract metrics
            question_metrics = []
            metrics_data = data.get('metrics', [])
            
            if isinstance(metrics_data, list):
                # Multiple questions
                for question_data in metrics_data:
                    question_metrics.append(QuestionMetrics.from_dict(question_data))
            else:
                # Legacy format - single aggregated metrics
                question_metrics.append(QuestionMetrics.from_dict(metrics_data))
            
            # Calculate aggregated metrics
            aggregated_metrics = self._calculate_aggregated_metrics(question_metrics)
            
            return LoadedExplorationMetrics(
                metadata=metadata,
                question_metrics=question_metrics,
                aggregated_metrics=aggregated_metrics
            )
            
        except Exception as e:
            logger.error(f"Error loading metrics from {file_path}: {e}")
            raise
    
    def _calculate_aggregated_metrics(self, question_metrics: List[QuestionMetrics]) -> AggregatedMetrics:
        """Calculate aggregated metrics across questions."""
        if not question_metrics:
            return AggregatedMetrics([], [], [], [], [], [], [])
        
        # Find the maximum number of rounds across all questions
        max_rounds = max(qm.num_rounds for qm in question_metrics)
        
        # Initialize aggregated lists
        agg_best_so_far = []
        agg_pool_mean = []
        agg_replacement_ratio = []
        agg_lift_per_1k_tokens = []
        agg_ci_width = []
        agg_ucb_gap = []
        agg_stability = []
        
        # Aggregate for each round
        for round_idx in range(max_rounds):
            # Collect values for this round across all questions
            round_best_so_far = []
            round_pool_mean = []
            round_replacement_ratio = []
            round_lift_per_1k_tokens = []
            round_ci_width = []
            round_ucb_gap = []
            
            for qm in question_metrics:
                if round_idx < len(qm.best_so_far):
                    round_best_so_far.append(qm.best_so_far[round_idx])
                if round_idx < len(qm.pool_mean):
                    round_pool_mean.append(qm.pool_mean[round_idx])
                if round_idx < len(qm.replacement_ratio):
                    round_replacement_ratio.append(qm.replacement_ratio[round_idx])
                if round_idx < len(qm.lift_per_1k_tokens):
                    round_lift_per_1k_tokens.append(qm.lift_per_1k_tokens[round_idx])
                if round_idx < len(qm.ci_width):
                    round_ci_width.append(qm.ci_width[round_idx])
                if round_idx < len(qm.ucb_gap):
                    round_ucb_gap.append(qm.ucb_gap[round_idx])
            
            # Calculate means for this round
            agg_best_so_far.append(np.mean(round_best_so_far) if round_best_so_far else 0.0)
            agg_pool_mean.append(np.mean(round_pool_mean) if round_pool_mean else 0.0)
            agg_replacement_ratio.append(np.mean(round_replacement_ratio) if round_replacement_ratio else 0.0)
            agg_lift_per_1k_tokens.append(np.mean(round_lift_per_1k_tokens) if round_lift_per_1k_tokens else 0.0)
            agg_ci_width.append(np.mean(round_ci_width) if round_ci_width else 0.0)
            agg_ucb_gap.append(np.mean(round_ucb_gap) if round_ucb_gap else 0.0)
            
            # Calculate stability (coefficient of variation) for best_so_far
            if len(round_best_so_far) > 1:
                mean_val = np.mean(round_best_so_far)
                std_val = np.std(round_best_so_far)
                cv = std_val / mean_val if mean_val != 0 else 0
                agg_stability.append(cv)
            else:
                agg_stability.append(0.0)
        
        return AggregatedMetrics(
            best_so_far=agg_best_so_far,
            pool_mean=agg_pool_mean,
            replacement_ratio=agg_replacement_ratio,
            lift_per_1k_tokens=agg_lift_per_1k_tokens,
            ci_width=agg_ci_width,
            ucb_gap=agg_ucb_gap,
            stability=agg_stability
        )
    
    def load_strategy_metrics(self, strategy: str) -> List[LoadedExplorationMetrics]:
        """Load all metrics for a specific strategy."""
        strategy_metrics = []
        
        for file_path in self.available_files:
            try:
                metrics = self.load_metrics_file(file_path)
                if metrics.metadata.strategy == strategy:
                    strategy_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Error loading {file_path} for strategy {strategy}: {e}")
        
        return strategy_metrics
    
    def load_all_metrics(self, exclude_filters: Optional[Dict[str, List[str]]] = None) -> List[LoadedExplorationMetrics]:
        """Load all available metrics files.
        
        Args:
            exclude_filters: Dictionary of filters to exclude files
                           Format: {'key': ['value1', 'value2']}
        """
        all_metrics = []
        
        for file_path in self.available_files:
            try:
                metrics = self.load_metrics_file(file_path)
                
                # Apply exclude filters if provided
                if exclude_filters:
                    should_exclude = False
                    for key, values in exclude_filters.items():
                        metadata_value = getattr(metrics.metadata, key, None)
                        if metadata_value is not None:
                            metadata_value_str = str(metadata_value).lower()
                            for value in values:
                                if value.lower() in metadata_value_str:
                                    should_exclude = True
                                    break
                        if should_exclude:
                            break
                    
                    if should_exclude:
                        logger.debug(f"Excluding {file_path} due to filters")
                        continue
                
                all_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        logger.info(f"Successfully loaded {len(all_metrics)} metrics files")
        return all_metrics
    
    def get_metrics_by_criteria(self, strategy: Optional[str] = None, 
                               dataset: Optional[str] = None,
                               judge_type: Optional[str] = None,
                               judge_backbone: Optional[str] = None) -> List[LoadedExplorationMetrics]:
        """Get metrics filtered by various criteria."""
        filtered_metrics = []
        
        for file_path in self.available_files:
            try:
                metrics = self.load_metrics_file(file_path)
                
                # Apply filters
                if strategy and metrics.metadata.strategy != strategy:
                    continue
                if dataset and metrics.metadata.dataset_name != dataset:
                    continue
                if judge_type and metrics.metadata.judge_type != judge_type:
                    continue
                if judge_backbone and metrics.metadata.judge_backbone != judge_backbone:
                    continue
                
                filtered_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        return filtered_metrics
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics."""
        all_metrics = self.load_all_metrics()
        
        if not all_metrics:
            return {}
        
        # Collect statistics
        strategies = defaultdict(int)
        datasets = defaultdict(int)
        judge_types = defaultdict(int)
        judge_backbones = defaultdict(int)
        
        final_scores = []
        improvements = []
        num_questions = []
        
        for metrics in all_metrics:
            strategies[metrics.metadata.strategy] += 1
            datasets[metrics.metadata.dataset_name] += 1
            judge_types[metrics.metadata.judge_type] += 1
            judge_backbones[metrics.metadata.judge_backbone] += 1
            
            final_scores.append(metrics.aggregated_metrics.final_score)
            improvements.append(metrics.aggregated_metrics.total_improvement)
            num_questions.append(len(metrics.question_metrics))
        
        return {
            'total_files': len(all_metrics),
            'strategies': dict(strategies),
            'datasets': dict(datasets),
            'judge_types': dict(judge_types),
            'judge_backbones': dict(judge_backbones),
            'final_score_stats': {
                'mean': np.mean(final_scores),
                'std': np.std(final_scores),
                'min': np.min(final_scores),
                'max': np.max(final_scores)
            },
            'improvement_stats': {
                'mean': np.mean(improvements),
                'std': np.std(improvements),
                'min': np.min(improvements),
                'max': np.max(improvements)
            },
            'questions_per_file_stats': {
                'mean': np.mean(num_questions),
                'std': np.std(num_questions),
                'min': np.min(num_questions),
                'max': np.max(num_questions)
            }
        }
    
    def compare_strategies(self, strategies: List[str], 
                          dataset: Optional[str] = None,
                          judge_type: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Compare performance across different strategies."""
        comparison = {}
        
        for strategy in strategies:
            strategy_metrics = self.get_metrics_by_criteria(
                strategy=strategy, 
                dataset=dataset, 
                judge_type=judge_type
            )
            
            if strategy_metrics:
                final_scores = [m.aggregated_metrics.final_score for m in strategy_metrics]
                improvements = [m.aggregated_metrics.total_improvement for m in strategy_metrics]
                
                comparison[strategy] = {
                    'count': len(strategy_metrics),
                    'final_score_mean': np.mean(final_scores),
                    'final_score_std': np.std(final_scores),
                    'improvement_mean': np.mean(improvements),
                    'improvement_std': np.std(improvements)
                }
        
        return comparison


def main():
    """Example usage of the MetricsLoader."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize loader
    loader = MetricsLoader()
    
    # Print summary statistics
    print("=== METRICS LOADER SUMMARY ===")
    stats = loader.get_summary_statistics()
    
    print(f"Total files: {stats['total_files']}")
    print(f"Strategies: {list(stats['strategies'].keys())}")
    print(f"Datasets: {list(stats['datasets'].keys())}")
    print(f"Judge types: {list(stats['judge_types'].keys())}")
    
    print(f"\nFinal score stats:")
    print(f"  Mean: {stats['final_score_stats']['mean']:.3f}")
    print(f"  Std:  {stats['final_score_stats']['std']:.3f}")
    
    print(f"\nImprovement stats:")
    print(f"  Mean: {stats['improvement_stats']['mean']:.3f}")
    print(f"  Std:  {stats['improvement_stats']['std']:.3f}")
    
    # Example: Filter and exclude functionality
    print("\n=== FILTER AND EXCLUDE EXAMPLES ===")
    
    # Example 1: Filter by strategy using string format
    print(f"Original files: {len(loader.available_files)}")
    loader.filter_by_string("strategy=ucb")
    print(f"After filtering by strategy=ucb: {len(loader.available_files)}")
    
    # Reset and try exclusion
    loader.reset_available_files()
    loader.exclude_by_string("strategy=random")
    print(f"After excluding strategy=random: {len(loader.available_files)}")
    
    # Reset and try combined filtering
    loader.reset_available_files()
    loader.filter_and_exclude_by_string("judge_type=single", "strategy=random")
    print(f"After filtering judge_type=single and excluding strategy=random: {len(loader.available_files)}")
    
    # Reset for remaining examples
    loader.reset_available_files()
    
    # Example: Compare strategies
    print("\n=== STRATEGY COMPARISON ===")
    strategies = ['ucb', 'random', 'simple_rewrite_holistic']
    comparison = loader.compare_strategies(strategies)
    
    for strategy, metrics in comparison.items():
        print(f"\n{strategy}:")
        print(f"  Count: {metrics['count']}")
        print(f"  Final score: {metrics['final_score_mean']:.3f} ± {metrics['final_score_std']:.3f}")
        print(f"  Improvement: {metrics['improvement_mean']:.3f} ± {metrics['improvement_std']:.3f}")
    
    # Example: Load specific metrics
    print("\n=== EXAMPLE METRICS ===")
    ucb_metrics = loader.load_strategy_metrics('ucb')
    if ucb_metrics:
        example = ucb_metrics[0]
        print(f"Example UCB run:")
        print(f"  Dataset: {example.metadata.dataset_name}")
        print(f"  Judge: {example.metadata.judge_backbone}")
        print(f"  Questions: {len(example.question_metrics)}")
        print(f"  Final score: {example.aggregated_metrics.final_score:.3f}")
        print(f"  Total improvement: {example.aggregated_metrics.total_improvement:.3f}")


if __name__ == "__main__":
    main()
