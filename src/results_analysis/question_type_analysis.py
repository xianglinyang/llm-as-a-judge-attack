#!/usr/bin/env python3
"""
Table Generator for LLM-as-a-Judge Attack Results

This module generates comparison tables showing ASR (Attack Success Rate) and 
SL (Score Lift) metrics across different question categories
and subjective/objective groupings.

Based on the category mapping:
- Subjective: categories 1,4,5,6 (Computer Science, Business, Writing, Social)  
- Objective: categories 2,3 (Mathematics, Science)
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import argparse

from src.results_analysis.trajectory_loader import TrajectoryLoader, LoadedTrajectory
from src.data.assign_category import CATEGORIES

logger = logging.getLogger(__name__)

# Category mapping based on user requirements
SUBJECTIVE_CATEGORIES = [
    "Computer Science & Programming",  # 1
    "Business & Finance",              # 4  
    "Writing & Communication",         # 5
    "Social & Daily Life"              # 6
]

OBJECTIVE_CATEGORIES = [
    "Mathematics & Statistics",        # 2
    "Science & Engineering"            # 3
]

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

def main():
    """Main function to generate comparison tables."""
    parser = argparse.ArgumentParser(description='Generate attack comparison tables')
    parser.add_argument('--trajectory_dir', type=str, required=True,
                      help='Directory containing trajectory files')
    parser.add_argument('--output_file', type=str, default=None,
                      help='Output file to save tables (optional)')
    parser.add_argument('--judge_backbone', type=str, default='gpt-4',
                      help='Judge model to filter by')
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
    
    args = parser.parse_args()
    
    attack_configs = [
        {
            "name": "BITE (ours)",
            "filter_criteria": {
                "strategy": "ucb",
                "judge_backbone": args.judge_backbone,
                "dataset_name": args.dataset_name
            }
        },
        {
            "name": "Random",
            "filter_criteria": {
                "strategy": "random", 
                "judge_backbone": args.judge_backbone,
                "dataset_name": args.dataset_name
            }
        },
        {
            "name": "Simple Rewrite",
            "filter_criteria": {
                "strategy": "simple_rewrite",
                "judge_backbone": args.judge_backbone,
                "dataset_name": args.dataset_name
            }
        }
    ]
    
    print("Generating attack comparison table...")
    print(f"Trajectory directory: {args.trajectory_dir}")
    print(f"Judge backbone: {args.judge_backbone}")
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
            
        # Multi-group analysis
        elif args.multi_group:
            print("\n📊 MULTI-GROUP ANALYSIS")
            print("=" * 50)
            
            group_configs = [
                {
                    "group_by": "strategy",
                    "filter_criteria": {"judge_backbone": args.judge_backbone, "dataset_name": args.dataset_name},
                    "baseline_group": "random",
                    "title": f"Strategy Comparison ({args.judge_backbone} Judge, {args.dataset_name})"
                },
                {
                    "group_by": "judge_backbone", 
                    "filter_criteria": {"dataset_name": args.dataset_name},
                    "title": f"Judge Model Comparison ({args.dataset_name})"
                },
                {
                    "group_by": "dataset_name",
                    "filter_criteria": {"judge_backbone": args.judge_backbone},
                    "title": f"Dataset Comparison ({args.judge_backbone} Judge)"
                }
            ]
            
            multi_analysis = generate_multi_group_analysis(args.trajectory_dir, group_configs)
            print(multi_analysis)
            output_content.append(f"# Multi-Group Analysis\n\n{multi_analysis}\n\n")
            
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
if __name__ == "__main__":
    # Example configuration
    trajectory_dir = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories"
    
    attack_configs = [
        {
            "name": "Holistic",
            "filter_criteria": {"strategy": "simple_rewrite_holistic"}
        },
        {
            "name": "Improved", 
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
    
    # Group by examples
    print("\n=== Group by Strategy ===")
    strategy_table = generate_grouped_table(
        trajectory_dir=trajectory_dir,
        group_by="strategy",
        filter_criteria={"judge_backbone": "gpt-4"},
        baseline_group="random"
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
