#!/usr/bin/env python3
"""
Table Generator for LLM-as-a-Judge Attack Results

This module generates comparison tables showing ASR (Attack Success Rate) and 
SL (Score Lift) metrics across different question categories
and subjective/objective groupings.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import argparse
from collections import defaultdict

from src.results_analysis.results_loader.trajectory_loader import TrajectoryLoader, LoadedTrajectory
from src.results_analysis.results_loader.data_loader import DataLoader
from src.data.assign_category import CATEGORIES

# Define objective and subjective categories based on question types
OBJECTIVE_CATEGORIES = ["Objective"]  # Questions with clear, factual answers
SUBJECTIVE_CATEGORIES = ["Subjective"]  # Questions with opinion-based or creative answers

logger = logging.getLogger(__name__)

@dataclass 
class MetricResult:
    """Container for ASR and SL results."""
    asr: float
    sl: float
    delta_asr: Optional[float] = None
    delta_sl: Optional[float] = None

@dataclass
class AttackResult:
    """Results for a specific attack method."""
    name: str
    objective_asr: float
    subjective_asr: float
    objective_sl: float  
    subjective_sl: float
    category_results: Dict[str, MetricResult]

@dataclass
class QuestionTypeResult:
    """Results organized by question type from data_loader."""
    name: str
    question_type_results: Dict[str, MetricResult]  # Maps question_type to MetricResult
    category_results: Dict[str, MetricResult]       # Maps category to MetricResult
    instruction_samples: Dict[str, List[str]]       # Maps question_type to sample instructions

class CategoryComparisonAnalyzer:
    """Main class for generating attack comparison tables."""
    
    def __init__(self, trajectory_dir: str):
        """
        Initialize table generator.
        
        Args:
            trajectory_dir: Directory containing trajectory files
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.loader = TrajectoryLoader(str(trajectory_dir))
        
    def calculate_asr(self, initial_scores: np.ndarray, final_scores: np.ndarray) -> float:
        """
        Calculate Attack Success Rate (ASR).
        
        ASR = (number of improved responses) / (total responses)
        
        Args:
            initial_scores: Initial judge scores
            final_scores: Final judge scores after attack
            
        Returns:
            ASR as percentage
        """
        improvements = final_scores - initial_scores
        success_count = np.sum(improvements > 0)
        total_count = len(improvements)
        
        if total_count == 0:
            return 0.0
            
        return (success_count / total_count) * 100.0
        
    def calculate_sl(self, initial_scores: np.ndarray, final_scores: np.ndarray) -> float:
        """
        Calculate Score Lift.
        
        SL = average score improvement
        
        Args:
            initial_scores: Initial judge scores  
            final_scores: Final judge scores after attack
            
        Returns:
            SL (average score lift)
        """
        improvements = final_scores - initial_scores
        return float(np.mean(improvements))
        
    def get_category_metrics(self, trajectories: List[LoadedTrajectory], 
                           category: str) -> MetricResult:
        """
        Calculate metrics for a specific category.
        
        Args:
            trajectories: List of loaded trajectory data
            category: Category name to filter by
            
        Returns:
            MetricResult with ASR and SL
        """
        initial_scores = []
        final_scores = []
        
        for traj in trajectories:
            category_items = traj.get_trajectory_by_category(category)
            for item in category_items:
                initial_scores.append(item.initial_score)
                final_scores.append(item.final_score)
        
        if not initial_scores:
            logger.warning(f"No data found for category: {category}")
            return MetricResult(asr=0.0, sl=0.0)
            
        initial_scores = np.array(initial_scores)
        final_scores = np.array(final_scores)
        
        asr = self.calculate_asr(initial_scores, final_scores)
        sl = self.calculate_sl(initial_scores, final_scores)
        
        return MetricResult(asr=asr, sl=sl)
        
    def get_grouped_metrics(self, trajectories: List[LoadedTrajectory],
                          categories: List[str]) -> MetricResult:
        """
        Calculate metrics for a group of categories.
        
        Args:
            trajectories: List of loaded trajectory data
            categories: List of category names to include
            
        Returns:
            MetricResult with aggregated ASR and SL
        """
        initial_scores = []
        final_scores = []
        
        for traj in trajectories:
            for category in categories:
                category_items = traj.get_trajectory_by_category(category) 
                for item in category_items:
                    initial_scores.append(item.initial_score)
                    final_scores.append(item.final_score)
        
        if not initial_scores:
            logger.warning(f"No data found for categories: {categories}")
            return MetricResult(asr=0.0, sl=0.0)
            
        initial_scores = np.array(initial_scores)
        final_scores = np.array(final_scores)
        
        asr = self.calculate_asr(initial_scores, final_scores)
        sl = self.calculate_sl(initial_scores, final_scores)
        
        return MetricResult(asr=asr, sl=sl)
        
    def analyze_attack_method(self, attack_name: str, 
                            filter_criteria: Optional[Dict[str, Any]] = None) -> AttackResult:
        """
        Analyze results for a specific attack method.
        
        Args:
            attack_name: Name of the attack method
            filter_criteria: Additional filtering criteria for trajectory loading
            
        Returns:
            AttackResult with all metrics
        """
        # Load trajectories with filtering
        trajectories = self.loader.load_trajectories()
        
        if filter_criteria:
            trajectories = self.loader.filter_trajectories(trajectories, **filter_criteria)
            
        if not trajectories:
            logger.error(f"No trajectories found for {attack_name}")
            return AttackResult(
                name=attack_name,
                objective_asr=0.0, subjective_asr=0.0,
                objective_sl=0.0, subjective_sl=0.0,
                category_results={}
            )
        
        # Calculate metrics for each individual category
        category_results = {}
        for category in CATEGORIES:
            category_results[category] = self.get_category_metrics(trajectories, category)
        
        # Calculate grouped metrics
        objective_result = self.get_grouped_metrics(trajectories, OBJECTIVE_CATEGORIES)
        subjective_result = self.get_grouped_metrics(trajectories, SUBJECTIVE_CATEGORIES)
        
        return AttackResult(
            name=attack_name,
            objective_asr=objective_result.asr,
            subjective_asr=subjective_result.asr, 
            objective_sl=objective_result.sl,
            subjective_sl=subjective_result.sl,
            category_results=category_results
        )
        
    def compare_attacks(self, attack_configs: List[Dict[str, Any]]) -> List[AttackResult]:
        """
        Compare multiple attack methods.
        
        Args:
            attack_configs: List of attack configurations with 'name' and optional filter criteria
            
        Returns:
            List of AttackResult objects
        """
        results = []
        
        for config in attack_configs:
            attack_name = config['name']
            filter_criteria = config.get('filter_criteria', {})
            
            logger.info(f"Analyzing attack method: {attack_name}")
            result = self.analyze_attack_method(attack_name, filter_criteria)
            results.append(result)
            
        return results
        
    def generate_comparison_table(self, attack_configs: List[Dict[str, Any]], 
                                baseline_idx: int = 0) -> str:
        """
        Generate markdown comparison table.
        
        Args:
            attack_configs: List of attack configurations
            baseline_idx: Index of baseline method for delta calculation
            
        Returns:
            Markdown table string
        """
        results = self.compare_attacks(attack_configs)
        
        if not results:
            return "No results to display."
            
        # Calculate deltas relative to baseline
        baseline = results[baseline_idx]
        for i, result in enumerate(results):
            if i != baseline_idx:
                result.objective_asr_delta = result.objective_asr - baseline.objective_asr
                result.subjective_asr_delta = result.subjective_asr - baseline.subjective_asr  
                result.objective_sl_delta = result.objective_sl - baseline.objective_sl
                result.subjective_sl_delta = result.subjective_sl - baseline.subjective_sl
            else:
                result.objective_asr_delta = 0.0
                result.subjective_asr_delta = 0.0
                result.objective_sl_delta = 0.0  
                result.subjective_sl_delta = 0.0
        
        # Generate table
        table_lines = []
        table_lines.append("| Attack           | Objective ASR ↑ | Subjective ASR ↑ | Δ     | Objective SL ↑ | Subjective SL ↑ | Δ     |")
        table_lines.append("| ---------------- | --------------- | ---------------- | ----- | ---------------- | ----------------- | ----- |")
        
        for result in results:
            # Format percentages and deltas
            obj_asr = f"{result.objective_asr:.1f}%"
            subj_asr = f"{result.subjective_asr:.1f}%"
            asr_delta = f"{result.subjective_asr_delta:+.1f}" if hasattr(result, 'subjective_asr_delta') else "—"
            
            obj_sl = f"{result.objective_sl:.2f}"
            subj_sl = f"{result.subjective_sl:.2f}" 
            sl_delta = f"{result.subjective_sl_delta:+.2f}" if hasattr(result, 'subjective_sl_delta') else "—"
            
            # Bold formatting for best results (you can customize this logic)
            if result.name.lower().startswith('bite'):
                obj_asr = f"**{obj_asr}**"
                subj_asr = f"**{subj_asr}**"
                obj_sl = f"**{obj_sl}**"
                subj_sl = f"**{subj_sl}**"
                result_name = f"**{result.name}**"
            else:
                result_name = result.name
                
            table_lines.append(f"| {result_name:16} | {obj_asr:15} | {subj_asr:16} | {asr_delta:5} | {obj_sl:16} | {subj_sl:17} | {sl_delta:5} |")
        
        return "\n".join(table_lines)
        
    def generate_detailed_category_table(self, attack_configs: List[Dict[str, Any]]) -> str:
        """
        Generate detailed table showing all 7 categories.
        
        Args:
            attack_configs: List of attack configurations
            
        Returns:
            Markdown table string with all categories
        """
        results = self.compare_attacks(attack_configs)
        
        if not results:
            return "No results to display."
        
        # Create table header
        header_parts = ["| Attack"]
        for category in CATEGORIES:
            short_name = category.split()[0]  # Use first word as short name
            header_parts.extend([f"| {short_name} ASR ↑", f"| {short_name} SL ↑"])
        header_parts.append("|")
        
        separator_parts = ["| --------"]
        for _ in CATEGORIES:
            separator_parts.extend(["| -----------", "| ------------"])
        separator_parts.append("|")
        
        table_lines = [
            "".join(header_parts),
            "".join(separator_parts)
        ]
        
        # Add data rows
        for result in results:
            row_parts = [f"| {result.name}"]
            
            for category in CATEGORIES:
                if category in result.category_results:
                    cat_result = result.category_results[category]
                    asr_str = f"{cat_result.asr:.1f}%"
                    sl_str = f"{cat_result.sl:.2f}"
                else:
                    asr_str = "—"
                    sl_str = "—"
                    
                row_parts.extend([f"| {asr_str}", f"| {sl_str}"])
            row_parts.append("|")
            
            table_lines.append("".join(row_parts))
        
        return "\n".join(table_lines)
    
    def group_trajectories_by(self, trajectories: List[LoadedTrajectory], 
                             group_by: str) -> Dict[str, List[LoadedTrajectory]]:
        """
        Group trajectories by a metadata field.
        
        Args:
            trajectories: List of loaded trajectory data
            group_by: Metadata field to group by (e.g., 'strategy', 'judge_backbone', 'dataset_name')
            
        Returns:
            Dictionary mapping group values to trajectory lists
        """
        groups = {}
        
        for traj in trajectories:
            group_value = getattr(traj.metadata, group_by, None)
            if group_value is None:
                group_value = "Unknown"
                
            if group_value not in groups:
                groups[group_value] = []
            groups[group_value].append(traj)
            
        return groups
        
    def analyze_groups(self, group_by: str, 
                      filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, AttackResult]:
        """
        Analyze results grouped by a specific criteria.
        
        Args:
            group_by: Field to group by (e.g., 'strategy', 'judge_backbone', 'dataset_name')
            filter_criteria: Additional filtering criteria
            
        Returns:
            Dictionary mapping group names to AttackResult objects
        """
        # Load and filter trajectories
        trajectories = self.loader.load_trajectories()
        
        if filter_criteria:
            trajectories = self.loader.filter_trajectories(trajectories, **filter_criteria)
            
        if not trajectories:
            logger.warning("No trajectories found with given criteria")
            return {}
            
        # Group trajectories
        groups = self.group_trajectories_by(trajectories, group_by)
        
        # Analyze each group
        results = {}
        for group_name, group_trajs in groups.items():
            logger.info(f"Analyzing group: {group_name} ({len(group_trajs)} trajectories)")
            
            # Calculate metrics for individual categories
            category_results = {}
            for category in CATEGORIES:
                category_results[category] = self.get_category_metrics(group_trajs, category)
            
            # Calculate grouped metrics
            objective_result = self.get_grouped_metrics(group_trajs, OBJECTIVE_CATEGORIES)
            subjective_result = self.get_grouped_metrics(group_trajs, SUBJECTIVE_CATEGORIES)
            
            results[group_name] = AttackResult(
                name=group_name,
                objective_asr=objective_result.asr,
                subjective_asr=subjective_result.asr,
                objective_sl=objective_result.sl,
                subjective_sl=subjective_result.sl,
                category_results=category_results
            )
            
        return results
        
    def generate_grouped_comparison_table(self, group_by: str,
                                        filter_criteria: Optional[Dict[str, Any]] = None,
                                        baseline_group: Optional[str] = None) -> str:
        """
        Generate comparison table grouped by a specific criteria.
        
        Args:
            group_by: Field to group by (e.g., 'strategy', 'judge_backbone', 'dataset_name')
            filter_criteria: Additional filtering criteria
            baseline_group: Name of baseline group for delta calculation
            
        Returns:
            Markdown table string
        """
        results = self.analyze_groups(group_by, filter_criteria)
        
        if not results:
            return f"No results found for group_by='{group_by}'"
            
        # Convert to list format for existing table generation logic
        results_list = [result for result in results.values()]
        
        # Find baseline index if specified
        baseline_idx = 0
        if baseline_group and baseline_group in results:
            for i, result in enumerate(results_list):
                if result.name == baseline_group:
                    baseline_idx = i
                    break
        
        # Calculate deltas relative to baseline
        baseline = results_list[baseline_idx]
        for i, result in enumerate(results_list):
            if i != baseline_idx:
                result.objective_asr_delta = result.objective_asr - baseline.objective_asr
                result.subjective_asr_delta = result.subjective_asr - baseline.subjective_asr
                result.objective_sl_delta = result.objective_sl - baseline.objective_sl
                result.subjective_sl_delta = result.subjective_sl - baseline.subjective_sl
            else:
                result.objective_asr_delta = 0.0
                result.subjective_asr_delta = 0.0
                result.objective_sl_delta = 0.0
                result.subjective_sl_delta = 0.0
        
        # Generate table
        table_lines = []
        table_lines.append(f"| {group_by.title()}           | Objective ASR ↑ | Subjective ASR ↑ | Δ     | Objective SL ↑ | Subjective SL ↑ | Δ     |")
        table_lines.append("| ---------------- | --------------- | ---------------- | ----- | ---------------- | ----------------- | ----- |")
        
        for result in results_list:
            # Format percentages and deltas
            obj_asr = f"{result.objective_asr:.1f}%"
            subj_asr = f"{result.subjective_asr:.1f}%"
            asr_delta = f"{result.subjective_asr_delta:+.1f}" if hasattr(result, 'subjective_asr_delta') else "—"
            
            obj_sl = f"{result.objective_sl:.2f}"
            subj_sl = f"{result.subjective_sl:.2f}"
            sl_delta = f"{result.subjective_sl_delta:+.2f}" if hasattr(result, 'subjective_sl_delta') else "—"
            
            # Bold formatting for best results (you can customize this logic)
            if result.name.lower() in ['ucb', 'bite']:
                obj_asr = f"**{obj_asr}**"
                subj_asr = f"**{subj_asr}**"
                obj_sl = f"**{obj_sl}**"
                subj_sl = f"**{subj_sl}**"
                result_name = f"**{result.name}**"
            else:
                result_name = result.name
                
            table_lines.append(f"| {result_name:16} | {obj_asr:15} | {subj_asr:16} | {asr_delta:5} | {obj_sl:16} | {subj_sl:17} | {sl_delta:5} |")
        
        return "\n".join(table_lines)
        
    def generate_multi_group_analysis(self, group_configs: List[Dict[str, Any]]) -> str:
        """
        Generate analysis tables for multiple grouping criteria.
        
        Args:
            group_configs: List of group configurations with 'group_by', optional 'filter_criteria', and 'baseline_group'
            
        Returns:
            Combined markdown string with all group analyses
        """
        output_sections = []
        
        for config in group_configs:
            group_by = config['group_by']
            filter_criteria = config.get('filter_criteria', {})
            baseline_group = config.get('baseline_group', None)
            title = config.get('title', f"Analysis by {group_by.replace('_', ' ').title()}")
            
            output_sections.append(f"## {title}")
            output_sections.append("")
            
            table = self.generate_grouped_comparison_table(
                group_by=group_by,
                filter_criteria=filter_criteria,
                baseline_group=baseline_group
            )
            
            output_sections.append(table)
            output_sections.append("")
            
        return "\n".join(output_sections)

class QuestionTypeAnalyzer:
    """Enhanced analyzer that integrates trajectory results with question type data from data_loader."""
    
    def __init__(self, trajectory_dir: str, data_dir: str = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"):
        """
        Initialize question type analyzer.
        
        Args:
            trajectory_dir: Directory containing trajectory files
            data_dir: Directory containing dataset files with question types
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.trajectory_loader = TrajectoryLoader(str(trajectory_dir))
        self.data_loader = DataLoader(data_dir)
        
        # Cache for loaded datasets
        self._dataset_cache = {}
    
    def _get_dataset(self, dataset_name: str):
        """Get dataset with caching."""
        if dataset_name not in self._dataset_cache:
            try:
                self._dataset_cache[dataset_name] = self.data_loader.load_dataset(dataset_name)
            except Exception as e:
                logger.warning(f"Could not load dataset {dataset_name}: {e}")
                self._dataset_cache[dataset_name] = None
        return self._dataset_cache[dataset_name]
    
    def _create_instruction_to_metadata_map(self, dataset_name: str) -> Dict[str, Dict[str, str]]:
        """Create a mapping from instruction text to metadata (question_type, category)."""
        dataset = self._get_dataset(dataset_name)
        if not dataset:
            return {}
        
        instruction_map = {}
        for instruction_meta in dataset.info.instructions:
            instruction_map[instruction_meta.instruction] = {
                'question_type': instruction_meta.question_type or 'Unknown',
                'category': instruction_meta.category,
                'dataset': instruction_meta.dataset
            }
        
        return instruction_map
    
    def analyze_question_types(self, attack_name: str, 
                              filter_criteria: Optional[Dict[str, Any]] = None) -> QuestionTypeResult:
        """
        Analyze results by question type using data_loader integration.
        
        Args:
            attack_name: Name of the attack method
            filter_criteria: Additional filtering criteria for trajectory loading
            
        Returns:
            QuestionTypeResult with metrics organized by question type
        """
        # Load trajectories
        trajectories = self.trajectory_loader.load_trajectories()
        
        if filter_criteria:
            trajectories = self.trajectory_loader.filter_trajectories(trajectories, **filter_criteria)
            
        if not trajectories:
            logger.error(f"No trajectories found for {attack_name}")
            return QuestionTypeResult(
                name=attack_name,
                question_type_results={},
                category_results={},
                instruction_samples={}
            )
        
        # Organize data by dataset and create instruction mappings
        dataset_mappings = {}
        for traj in trajectories:
            dataset_name = traj.metadata.dataset_name
            if dataset_name not in dataset_mappings:
                dataset_mappings[dataset_name] = self._create_instruction_to_metadata_map(dataset_name)
        
        # Group trajectory items by question type and category
        question_type_data = defaultdict(lambda: {'initial_scores': [], 'final_scores': [], 'instructions': []})
        category_data = defaultdict(lambda: {'initial_scores': [], 'final_scores': [], 'instructions': []})
        
        for traj in trajectories:
            dataset_name = traj.metadata.dataset_name
            instruction_map = dataset_mappings.get(dataset_name, {})
            
            for item in traj.trajectories:
                instruction = item.question
                
                # Get metadata from data_loader
                metadata = instruction_map.get(instruction, {
                    'question_type': 'Unknown',
                    'category': item.category,  # Fallback to trajectory category
                    'dataset': dataset_name
                })
                
                question_type = metadata['question_type']
                category = metadata['category']
                
                # Add to question type grouping
                question_type_data[question_type]['initial_scores'].append(item.initial_score)
                question_type_data[question_type]['final_scores'].append(item.final_score)
                question_type_data[question_type]['instructions'].append(instruction)
                
                # Add to category grouping
                category_data[category]['initial_scores'].append(item.initial_score)
                category_data[category]['final_scores'].append(item.final_score)
                category_data[category]['instructions'].append(instruction)
        
        # Calculate metrics for each question type
        question_type_results = {}
        instruction_samples = {}
        
        for question_type, data in question_type_data.items():
            if data['initial_scores']:
                initial_scores = np.array(data['initial_scores'])
                final_scores = np.array(data['final_scores'])
                
                asr = self._calculate_asr(initial_scores, final_scores)
                sl = self._calculate_sl(initial_scores, final_scores)
                
                question_type_results[question_type] = MetricResult(asr=asr, sl=sl)
                
                # Sample instructions for this question type (up to 3 examples)
                instruction_samples[question_type] = data['instructions'][:3]
        
        # Calculate metrics for each category
        category_results = {}
        for category, data in category_data.items():
            if data['initial_scores']:
                initial_scores = np.array(data['initial_scores'])
                final_scores = np.array(data['final_scores'])
                
                asr = self._calculate_asr(initial_scores, final_scores)
                sl = self._calculate_sl(initial_scores, final_scores)
                
                category_results[category] = MetricResult(asr=asr, sl=sl)
        
        return QuestionTypeResult(
            name=attack_name,
            question_type_results=question_type_results,
            category_results=category_results,
            instruction_samples=instruction_samples
        )
    
    def _calculate_asr(self, initial_scores: np.ndarray, final_scores: np.ndarray) -> float:
        """Calculate Attack Success Rate (ASR)."""
        improvements = final_scores - initial_scores
        success_count = np.sum(improvements > 0)
        total_count = len(improvements)
        
        if total_count == 0:
            return 0.0
            
        return (success_count / total_count) * 100.0
    
    def _calculate_sl(self, initial_scores: np.ndarray, final_scores: np.ndarray) -> float:
        """Calculate Score Lift."""
        improvements = final_scores - initial_scores
        return float(np.mean(improvements))
    
    def compare_question_types(self, attack_configs: List[Dict[str, Any]]) -> List[QuestionTypeResult]:
        """
        Compare multiple attack methods by question type.
        
        Args:
            attack_configs: List of attack configurations with 'name' and optional filter criteria
            
        Returns:
            List of QuestionTypeResult objects
        """
        results = []
        
        for config in attack_configs:
            attack_name = config['name']
            filter_criteria = config.get('filter_criteria', {})
            
            logger.info(f"Analyzing question types for attack method: {attack_name}")
            result = self.analyze_question_types(attack_name, filter_criteria)
            results.append(result)
            
        return results
    
    def generate_question_type_table(self, attack_configs: List[Dict[str, Any]]) -> str:
        """
        Generate markdown table showing results by question type.
        
        Args:
            attack_configs: List of attack configurations
            
        Returns:
            Markdown table string organized by question type
        """
        results = self.compare_question_types(attack_configs)
        
        if not results:
            return "No results to display."
        
        # Get all unique question types across all results
        all_question_types = set()
        for result in results:
            all_question_types.update(result.question_type_results.keys())
        
        all_question_types = sorted(list(all_question_types))
        
        if not all_question_types:
            return "No question type data found."
        
        # Create table header
        header_parts = ["| Attack"]
        for qt in all_question_types:
            short_name = qt[:10] if len(qt) > 10 else qt  # Truncate long names
            header_parts.extend([f"| {short_name} ASR ↑", f"| {short_name} SL ↑"])
        header_parts.append("|")
        
        separator_parts = ["| --------"]
        for _ in all_question_types:
            separator_parts.extend(["| -----------", "| -----------"])
        separator_parts.append("|")
        
        table_lines = [
            "".join(header_parts),
            "".join(separator_parts)
        ]
        
        # Add data rows
        for result in results:
            row_parts = [f"| {result.name}"]
            
            for qt in all_question_types:
                if qt in result.question_type_results:
                    qt_result = result.question_type_results[qt]
                    asr_str = f"{qt_result.asr:.1f}%"
                    sl_str = f"{qt_result.sl:.2f}"
                else:
                    asr_str = "—"
                    sl_str = "—"
                    
                row_parts.extend([f"| {asr_str}", f"| {sl_str}"])
            row_parts.append("|")
            
            table_lines.append("".join(row_parts))
        
        return "\n".join(table_lines)
    
    def generate_instruction_samples_report(self, attack_configs: List[Dict[str, Any]]) -> str:
        """
        Generate a report showing sample instructions for each question type.
        
        Args:
            attack_configs: List of attack configurations
            
        Returns:
            Markdown report with instruction samples
        """
        results = self.compare_question_types(attack_configs)
        
        if not results:
            return "No results to display."
        
        # Use the first result for instruction samples (they should be the same across attacks)
        first_result = results[0]
        
        report_lines = ["# Question Type Analysis with Instruction Samples\n"]
        
        for question_type, samples in first_result.instruction_samples.items():
            if samples:
                report_lines.append(f"## {question_type}")
                report_lines.append("")
                
                # Show metrics for this question type across all attacks
                report_lines.append("### Performance Metrics")
                report_lines.append("")
                report_lines.append("| Attack | ASR ↑ | SL ↑ |")
                report_lines.append("| ------ | ----- | ---- |")
                
                for result in results:
                    if question_type in result.question_type_results:
                        qt_result = result.question_type_results[question_type]
                        asr_str = f"{qt_result.asr:.1f}%"
                        sl_str = f"{qt_result.sl:.2f}"
                    else:
                        asr_str = "—"
                        sl_str = "—"
                    
                    report_lines.append(f"| {result.name} | {asr_str} | {sl_str} |")
                
                report_lines.append("")
                
                # Show sample instructions
                report_lines.append("### Sample Instructions")
                report_lines.append("")
                
                for i, instruction in enumerate(samples, 1):
                    # Truncate very long instructions
                    display_instruction = instruction[:200] + "..." if len(instruction) > 200 else instruction
                    report_lines.append(f"{i}. {display_instruction}")
                
                report_lines.append("")
        
        return "\n".join(report_lines)


# Convenience functions for easy usage
def generate_attack_comparison_table(trajectory_dir: str, 
                                   attack_configs: List[Dict[str, Any]],
                                   baseline_idx: int = 0) -> str:
    """
    Convenience function to generate attack comparison table.
    
    Args:
        trajectory_dir: Directory containing trajectory files
        attack_configs: List of attack configurations with 'name' and optional 'filter_criteria'
        baseline_idx: Index of baseline method for delta calculation
        
    Returns:
        Markdown table string
    """
    generator = CategoryComparisonAnalyzer(trajectory_dir)
    return generator.generate_comparison_table(attack_configs, baseline_idx)


def generate_category_analysis_table(trajectory_dir: str,
                                   attack_configs: List[Dict[str, Any]]) -> str:
    """
    Convenience function to generate detailed category analysis table.
    
    Args:
        trajectory_dir: Directory containing trajectory files  
        attack_configs: List of attack configurations
        
    Returns:
        Markdown table string with all categories
    """
    generator = CategoryComparisonAnalyzer(trajectory_dir)
    return generator.generate_detailed_category_table(attack_configs)


def generate_grouped_table(trajectory_dir: str,
                          group_by: str,
                          filter_criteria: Optional[Dict[str, Any]] = None,
                          baseline_group: Optional[str] = None) -> str:
    """
    Convenience function to generate grouped comparison table.
    
    Args:
        trajectory_dir: Directory containing trajectory files
        group_by: Field to group by (e.g., 'strategy', 'judge_backbone', 'dataset_name')
        filter_criteria: Additional filtering criteria
        baseline_group: Name of baseline group for delta calculation
        
    Returns:
        Markdown table string grouped by specified criteria
    """
    generator = CategoryComparisonAnalyzer(trajectory_dir)
    return generator.generate_grouped_comparison_table(group_by, filter_criteria, baseline_group)


def generate_multi_group_analysis(trajectory_dir: str,
                                 group_configs: List[Dict[str, Any]]) -> str:
    """
    Convenience function to generate analysis for multiple grouping criteria.
    
    Args:
        trajectory_dir: Directory containing trajectory files
        group_configs: List of group configurations with 'group_by', optional 'filter_criteria', and 'baseline_group'
        
    Returns:
        Combined markdown string with all group analyses
    """
    generator = CategoryComparisonAnalyzer(trajectory_dir)
    return generator.generate_multi_group_analysis(group_configs)


# New convenience functions for question type analysis
def generate_question_type_analysis_table(trajectory_dir: str,
                                         attack_configs: List[Dict[str, Any]],
                                         data_dir: str = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data") -> str:
    """
    Convenience function to generate question type analysis table.
    
    Args:
        trajectory_dir: Directory containing trajectory files
        attack_configs: List of attack configurations with 'name' and optional 'filter_criteria'
        data_dir: Directory containing dataset files with question types
        
    Returns:
        Markdown table string organized by question type
    """
    analyzer = QuestionTypeAnalyzer(trajectory_dir, data_dir)
    return analyzer.generate_question_type_table(attack_configs)


def generate_instruction_samples_report(trajectory_dir: str,
                                       attack_configs: List[Dict[str, Any]],
                                       data_dir: str = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data") -> str:
    """
    Convenience function to generate instruction samples report.
    
    Args:
        trajectory_dir: Directory containing trajectory files
        attack_configs: List of attack configurations with 'name' and optional 'filter_criteria'
        data_dir: Directory containing dataset files with question types
        
    Returns:
        Markdown report with instruction samples organized by question type
    """
    analyzer = QuestionTypeAnalyzer(trajectory_dir, data_dir)
    return analyzer.generate_instruction_samples_report(attack_configs)

def main():
    """Main function to generate comparison tables."""
    parser = argparse.ArgumentParser(description='Generate attack comparison tables')
    parser.add_argument('--trajectory_dir', type=str, default='/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories',
                      help='Directory containing trajectory files')
    parser.add_argument('--output_file', type=str, default=None,
                      help='Output file to save tables (optional)')
    parser.add_argument('--dataset_name', type=str, default='AlpacaEval',
                      help='Dataset to filter by')
    parser.add_argument('--detailed', action='store_true',
                      help='Also generate detailed category breakdown table')
    parser.add_argument('--group_by', type=str, default=None,
                      help='Group results by field (e.g., strategy, judge_backbone, dataset_name)')
    parser.add_argument('--baseline_group', type=str, default=None,
                      help='Baseline group for delta calculation (when using --group_by)')
    parser.add_argument('--multi_group', action='store_true',
                      help='Generate multi-group analysis with predefined configurations')
    parser.add_argument('--judge_type', type=str, default='pointwise', choices=['pointwise', 'alpaca_eval', "arena_hard_auto", "mlr_bench"],
                      help='Judge type to filter by')
    parser.add_argument('--question_type_analysis', action='store_true',
                      help='Generate question type analysis using data_loader integration')
    parser.add_argument('--instruction_samples', action='store_true',
                      help='Generate instruction samples report')
    parser.add_argument('--data_dir', type=str, default='/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data',
                      help='Directory containing dataset files with question types')
    
    args = parser.parse_args()
    
    attack_configs = [
        {
            "name": "Holistic Rewrite",
            "filter_criteria": {
                "strategy": "simple_rewrite_improve",
                "dataset_name": args.dataset_name,
                "judge_type": args.judge_type
            }
        },
        {
            "name": "Random",
            "filter_criteria": {
                "strategy": "random", 
                "dataset_name": args.dataset_name,
                "judge_type": args.judge_type
            }
        },
                {
            "name": "BITE (ours)",
            "filter_criteria": {
                "strategy": "ucb",
                "dataset_name": args.dataset_name,
                "judge_type": args.judge_type
            }
        }
    ]
    
    print("Generating attack comparison table...")
    print(f"Trajectory directory: {args.trajectory_dir}")
    print(f"Dataset: {args.dataset_name}")
    print("=" * 80)
    
    try:
        output_content = []
        
        # Group by analysis
        if args.group_by:
            print(f"\n📊 GROUP BY {args.group_by.upper()}")
            print("=" * 50)
            
            filter_criteria = {
                "judge_backbone": args.judge_backbone,
                "dataset_name": args.dataset_name
            }
            
            grouped_table = generate_grouped_table(
                trajectory_dir=args.trajectory_dir,
                group_by=args.group_by,
                filter_criteria=filter_criteria,
                baseline_group=args.baseline_group
            )
            print(grouped_table)
            output_content.append(f"# Group by {args.group_by.title()}\n\n{grouped_table}\n\n")
            
            # Generate detailed category table if requested
            if args.detailed:
                print("\n\n📋 DETAILED CATEGORY ANALYSIS")
                print("=" * 50)
                detailed_table = generate_category_analysis_table(
                    trajectory_dir=args.trajectory_dir,
                    attack_configs=attack_configs
                )
                print(detailed_table)
                output_content.append(f"## Detailed Category Analysis\n\n{detailed_table}\n\n")
            
        # Question type analysis
        elif args.question_type_analysis:
            print("\n📊 QUESTION TYPE ANALYSIS")
            print("=" * 50)
            
            question_type_table = generate_question_type_analysis_table(
                trajectory_dir=args.trajectory_dir,
                attack_configs=attack_configs,
                data_dir=args.data_dir
            )
            print(question_type_table)
            output_content.append(f"# Question Type Analysis\n\n{question_type_table}\n\n")
            
            # Generate instruction samples report if requested
            if args.instruction_samples:
                print("\n\n📋 INSTRUCTION SAMPLES REPORT")
                print("=" * 50)
                
                samples_report = generate_instruction_samples_report(
                    trajectory_dir=args.trajectory_dir,
                    attack_configs=attack_configs,
                    data_dir=args.data_dir
                )
                print(samples_report)
                output_content.append(f"{samples_report}\n\n")
            
            # Generate detailed category table if requested
            if args.detailed:
                print("\n\n📋 DETAILED CATEGORY ANALYSIS")
                print("=" * 50)
                detailed_table = generate_category_analysis_table(
                    trajectory_dir=args.trajectory_dir,
                    attack_configs=attack_configs
                )
                print(detailed_table)
                output_content.append(f"## Detailed Category Analysis\n\n{detailed_table}\n\n")
        
        # Multi-group analysis
        elif args.multi_group:
            print("\n📊 MULTI-GROUP ANALYSIS")
            print("=" * 50)
            
            group_configs = [
                {
                    "group_by": "judge_backbone", 
                    "filter_criteria": {"dataset_name": args.dataset_name},
                    "title": f"Judge Model Comparison ({args.dataset_name})"
                }
            ]
            
            multi_analysis = generate_multi_group_analysis(args.trajectory_dir, group_configs)
            print(multi_analysis)
            output_content.append(f"# Multi-Group Analysis\n\n{multi_analysis}\n\n")
            
            # Generate detailed category table if requested
            if args.detailed:
                print("\n\n📋 DETAILED CATEGORY ANALYSIS")
                print("=" * 50)
                detailed_table = generate_category_analysis_table(
                    trajectory_dir=args.trajectory_dir,
                    attack_configs=attack_configs
                )
                print(detailed_table)
                output_content.append(f"## Detailed Category Analysis\n\n{detailed_table}\n\n")
            
        # Standard attack comparison
        else:
            # Generate main comparison table
            comparison_table = generate_attack_comparison_table(
                trajectory_dir=args.trajectory_dir,
                attack_configs=attack_configs,
                baseline_idx=1  # Use Random as baseline (index 1)
            )
            
            print("\n📊 ATTACK COMPARISON TABLE")
            print("=" * 50)
            print(comparison_table)
            output_content.append(f"# Attack Comparison Results\n\n## Main Comparison Table\n\n{comparison_table}\n\n")
            
            # Generate detailed category table if requested
            if args.detailed:
                print("\n\n📋 DETAILED CATEGORY ANALYSIS")
                print("=" * 50)
                detailed_table = generate_category_analysis_table(
                    trajectory_dir=args.trajectory_dir,
                    attack_configs=attack_configs
                )
                print(detailed_table)
                output_content.append(f"## Detailed Category Analysis\n\n{detailed_table}\n\n")
        
        # Save to file if specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write("".join(output_content))
                    
            print(f"\n💾 Results saved to: {args.output_file}")
        
        print("\n✅ Table generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error generating tables: {str(e)}")
        logging.error(f"Table generation failed: {str(e)}", exc_info=True)
        sys.exit(1)

# Example usage
def demo():
    trajectory_dir = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories"
    data_dir = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"
    
    attack_configs = [
        {
            "name": "Holistic Rewrite", 
            "filter_criteria": {"strategy": "simple_rewrite_improve"}
        },
        {
            "name": "Random",
            "filter_criteria": {"strategy": "random"}
        },
        {
            "name": "BITE (ours)", 
            "filter_criteria": {"strategy": "ucb"}
        }
    ]
    
    # Generate comparison table
    print("=== Attack Comparison Table ===")
    table = generate_attack_comparison_table(trajectory_dir, attack_configs)
    print(table)
    
    print("\n=== Detailed Category Analysis ===") 
    detailed_table = generate_category_analysis_table(trajectory_dir, attack_configs)
    print(detailed_table)
    
    # NEW: Question Type Analysis using data_loader
    print("\n=== Question Type Analysis (Enhanced with data_loader) ===")
    question_type_table = generate_question_type_analysis_table(trajectory_dir, attack_configs, data_dir)
    print(question_type_table)
    
    print("\n=== Instruction Samples Report ===")
    samples_report = generate_instruction_samples_report(trajectory_dir, attack_configs, data_dir)
    print(samples_report)
    
    # Group by examples
    print("\n=== Group by Strategy ===")
    strategy_table = generate_grouped_table(
        trajectory_dir=trajectory_dir,
        group_by="strategy",
        filter_criteria={"judge_backbone": "gpt-4"},
        baseline_group="Holistic Rewrite"
    )
    print(strategy_table)
    
    print("\n=== Group by Judge Model ===")
    judge_table = generate_grouped_table(
        trajectory_dir=trajectory_dir,
        group_by="judge_backbone",
        filter_criteria={"dataset_name": "AlpacaEval"}
    )
    print(judge_table)
    
    print("\n=== Multi-Group Analysis ===")
    group_configs = [
        {
            "group_by": "strategy",
            "filter_criteria": {"judge_backbone": "gpt-4"},
            "baseline_group": "random",
            "title": "Strategy Comparison (GPT-4 Judge)"
        },
        {
            "group_by": "dataset_name", 
            "filter_criteria": {"strategy": "ucb"},
            "title": "Dataset Performance (BITE Method)"
        },
        {
            "group_by": "budget",
            "filter_criteria": {"strategy": "ucb", "judge_backbone": "gpt-4"},
            "baseline_group": "10",
            "title": "Budget Analysis (BITE + GPT-4)"
        }
    ]
    
    multi_analysis = generate_multi_group_analysis(trajectory_dir, group_configs)
    print(multi_analysis)

if __name__ == "__main__":
    main()
