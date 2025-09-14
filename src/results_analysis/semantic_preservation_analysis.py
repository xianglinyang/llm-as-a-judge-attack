#!/usr/bin/env python3
"""
Semantic Preservation Analysis for LLM-as-a-Judge Attack Results

This module analyzes how well attack methods preserve semantic meaning between
original and attacked responses using sentence transformer embeddings.

Method:
1. Load trajectory data for different attack methods
2. Extract original (first) and final (last) responses for each question
3. Calculate semantic similarity using sentence-transformers embeddings
4. Aggregate results by attack method, dataset, and question categories
5. Generate comparison tables and visualizations
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import argparse
import warnings
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


from src.results_analysis.results_loader.trajectory_loader import TrajectoryLoader, LoadedTrajectory
from src.results_analysis.results_loader.data_loader import DataLoader

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class SemanticPreservationResult:
    """Results for semantic preservation analysis."""
    attack_method: str
    dataset_name: str
    judge_type: str
    mean_similarity: float
    std_similarity: float
    median_similarity: float
    min_similarity: float
    max_similarity: float
    num_samples: int
    category_results: Dict[str, float]  # Category -> mean similarity
    question_type_results: Dict[str, float]  # Question type -> mean similarity

@dataclass
class SemanticComparisonResult:
    """Comparison results across multiple attack methods."""
    results: List[SemanticPreservationResult]
    baseline_method: Optional[str] = None

class SemanticPreservationAnalyzer:
    """Main class for semantic preservation analysis."""
    
    def __init__(self, trajectory_dir: str, data_dir: str = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"):
        """
        Initialize semantic preservation analyzer.
        
        Args:
            trajectory_dir: Directory containing trajectory files
            data_dir: Directory containing dataset files with metadata
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.trajectory_loader = TrajectoryLoader(str(trajectory_dir))
        self.data_loader = DataLoader(data_dir)
        
        # Initialize sentence transformer model
        self._init_sentence_transformer()
        
        # Cache for loaded datasets
        self._dataset_cache = {}
    
    def _init_sentence_transformer(self):
        """Initialize sentence transformer model for semantic similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import torch
            
            # Check for GPU availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Use a high-quality multilingual model for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            self.cosine_similarity = cosine_similarity
            logger.info(f"Initialized sentence transformer model: all-MiniLM-L6-v2 on {device}")
            
            # Enable half precision if using GPU for better performance
            if device == 'cuda':
                logger.info("GPU detected - using optimized inference settings")
            
        except ImportError as e:
            logger.error("Required packages not installed. Please install: pip install sentence-transformers scikit-learn torch")
            raise ImportError("sentence-transformers, scikit-learn, and torch are required for semantic analysis") from e
    
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
        """Create a mapping from instruction text to metadata."""
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
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using sentence transformers.
        
        Args:
            text1: First text (original response)
            text2: Second text (attacked response)
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1.strip() or not text2.strip():
            return 0.0
        
        try:
            # Generate embeddings
            embeddings = self.sentence_model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity_matrix = self.cosine_similarity([embeddings[0]], [embeddings[1]])
            similarity = similarity_matrix[0][0]
            
            # Ensure similarity is between 0 and 1
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def calculate_batch_semantic_similarities(self, text_pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[float]:
        """
        Calculate semantic similarities for multiple text pairs using batch inference.
        
        Args:
            text_pairs: List of (text1, text2) tuples
            batch_size: Number of pairs to process in each batch
            
        Returns:
            List of similarity scores between 0 and 1
        """
        if not text_pairs:
            return []
        
        similarities = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(text_pairs), batch_size):
            batch_pairs = text_pairs[i:i + batch_size]
            
            # Prepare texts for batch encoding
            batch_text1 = []
            batch_text2 = []
            valid_indices = []
            
            for idx, (text1, text2) in enumerate(batch_pairs):
                if text1.strip() and text2.strip():
                    batch_text1.append(text1)
                    batch_text2.append(text2)
                    valid_indices.append(idx)
                else:
                    # Handle empty texts by adding placeholder that will get 0.0 similarity
                    batch_text1.append("empty")
                    batch_text2.append("empty")
                    valid_indices.append(idx)
            
            try:
                # Batch encode all texts at once
                all_texts = batch_text1 + batch_text2
                embeddings = self.sentence_model.encode(all_texts, batch_size=batch_size, show_progress_bar=False)
                
                # Split embeddings back into text1 and text2 groups
                mid_point = len(batch_text1)
                embeddings1 = embeddings[:mid_point]
                embeddings2 = embeddings[mid_point:]
                
                # Calculate cosine similarities for this batch
                batch_similarities = self.cosine_similarity(embeddings1, embeddings2)
                
                # Extract diagonal (pairwise similarities)
                for idx, (original_text1, original_text2) in enumerate(batch_pairs):
                    if not original_text1.strip() or not original_text2.strip():
                        similarities.append(0.0)
                    else:
                        similarity = batch_similarities[idx][idx]
                        # Ensure similarity is between 0 and 1
                        similarities.append(max(0.0, min(1.0, similarity)))
                
                # Progress logging
                if i % (batch_size * 10) == 0:
                    logger.info(f"Processed {min(i + batch_size, len(text_pairs))}/{len(text_pairs)} text pairs...")
                    
            except Exception as e:
                logger.warning(f"Error in batch similarity calculation: {e}")
                # Fallback to individual calculation for this batch
                for text1, text2 in batch_pairs:
                    similarities.append(self.calculate_semantic_similarity(text1, text2))
        
        return similarities
    
    def extract_response_pairs(self, trajectories: List[LoadedTrajectory]) -> List[Tuple[str, str, str, str, str]]:
        """
        Extract (original_response, final_response, question, category, question_type) pairs.
        
        Args:
            trajectories: List of loaded trajectory data
            
        Returns:
            List of tuples: (original_response, final_response, question, category, question_type)
        """
        response_pairs = []
        one_shot_response_pairs = []
        
        for traj in trajectories:
            dataset_name = traj.metadata.dataset_name
            instruction_map = self._create_instruction_to_metadata_map(dataset_name)
            
            for trajectory_item in traj.trajectories:
                if len(trajectory_item.history) >= 2:
                    # Get original (first) and final (last) responses
                    # Extract the actual answer text from TrajectoryStep objects
                    original_response = trajectory_item.history[0].answer
                    final_response = trajectory_item.history[-1].answer
                    
                    question = trajectory_item.question
                    category = trajectory_item.category
                    
                    # Get question type from data_loader if available
                    metadata = instruction_map.get(question, {})
                    question_type = metadata.get('question_type', 'Unknown')
                    
                    response_pairs.append((
                        original_response,
                        final_response, 
                        question,
                        category,
                        question_type
                    ))
                    one_shot_response_pairs.append((
                        original_response,
                        trajectory_item.history[1].answer, 
                        question,
                        category,
                        question_type
                    ))
        
        return response_pairs, one_shot_response_pairs
    
    def analyze_attack_method(self, attack_name: str, 
                            filter_criteria: Optional[Dict[str, Any]] = None,
                            batch_size: int = 64) -> SemanticPreservationResult:
        """
        Analyze semantic preservation for a specific attack method.
        
        Args:
            attack_name: Name of the attack method
            filter_criteria: Additional filtering criteria for trajectory loading
            
        Returns:
            SemanticPreservationResult with analysis results
        """
        # Load trajectories
        trajectories = self.trajectory_loader.load_trajectories()
        
        if filter_criteria:
            trajectories = self.trajectory_loader.filter_trajectories(trajectories, **filter_criteria)
            
        if not trajectories:
            logger.error(f"No trajectories found for {attack_name}")
            return SemanticPreservationResult(
                attack_method=attack_name,
                dataset_name=filter_criteria.get('dataset_name', 'Unknown'),
                judge_type=filter_criteria.get('judge_type', 'Unknown'),
                mean_similarity=0.0,
                std_similarity=0.0,
                median_similarity=0.0,
                min_similarity=0.0,
                max_similarity=0.0,
                num_samples=0,
                category_results={},
                question_type_results={}
            )
        
        # Extract response pairs
        response_pairs, one_shot_response_pairs = self.extract_response_pairs(trajectories)
        
        if not response_pairs:
            logger.warning(f"No response pairs found for {attack_name}")
            return SemanticPreservationResult(
                attack_method=attack_name,
                dataset_name=filter_criteria.get('dataset_name', 'Unknown'),
                judge_type=filter_criteria.get('judge_type', 'Unknown'),
                mean_similarity=0.0,
                std_similarity=0.0,
                median_similarity=0.0,
                min_similarity=0.0,
                max_similarity=0.0,
                num_samples=0,
                category_results={},
                question_type_results={}
            )
        
        logger.info(f"Calculating semantic similarities for {len(response_pairs)} response pairs using batch inference...")
        
        # Prepare text pairs for batch processing
        text_pairs = [(original, final) for original, final, _, _, _ in response_pairs]
        one_shot_text_pairs = [(original, final) for original, final, _, _, _ in one_shot_response_pairs]
        
        # Calculate similarities using batch inference
        similarities = self.calculate_batch_semantic_similarities(text_pairs, batch_size=batch_size)
        one_shot_similarities = self.calculate_batch_semantic_similarities(one_shot_text_pairs, batch_size=batch_size)
        
        # Group similarities by category and question type
        category_similarities = defaultdict(list)
        question_type_similarities = defaultdict(list)
        one_shot_category_similarities = defaultdict(list)
        one_shot_question_type_similarities = defaultdict(list)
        
        for similarity, one_shot_similarity, (_, _, question, category, question_type) in zip(similarities, one_shot_similarities, response_pairs):
            category_similarities[category].append(similarity)
            question_type_similarities[question_type].append(similarity)
            one_shot_category_similarities[category].append(one_shot_similarity)
            one_shot_question_type_similarities[question_type].append(one_shot_similarity)
        
        # Calculate aggregate statistics
        similarities = np.array(similarities)
        one_shot_similarities = np.array(one_shot_similarities)
        
        # Calculate category averages
        category_results = {}
        for category, sims in category_similarities.items():
            if sims:
                category_results[category] = float(np.mean(sims))
        
        # Calculate one-shot category averages
        one_shot_category_results = {}
        for category, sims in one_shot_category_similarities.items():
            if sims:
                one_shot_category_results[category] = float(np.mean(sims))
        
        # Calculate question type averages
        question_type_results = {}
        for question_type, sims in question_type_similarities.items():
            if sims:
                question_type_results[question_type] = float(np.mean(sims))
        
        # Calculate one-shot question type averages
        one_shot_question_type_results = {}
        for question_type, one_shot_sims in one_shot_question_type_similarities.items():
            if one_shot_sims:
                one_shot_question_type_results[question_type] = float(np.mean(one_shot_sims))
        
        return SemanticPreservationResult(
            attack_method=attack_name,
            dataset_name=filter_criteria.get('dataset_name', 'Unknown'),
            judge_type=filter_criteria.get('judge_type', 'Unknown'),
            mean_similarity=float(np.mean(similarities)),
            std_similarity=float(np.std(similarities)),
            median_similarity=float(np.median(similarities)),
            min_similarity=float(np.min(similarities)),
            max_similarity=float(np.max(similarities)),
            num_samples=len(similarities),
            category_results=category_results,
            question_type_results=question_type_results
        ), SemanticPreservationResult(
            attack_method=attack_name+"-One-shot",
            dataset_name=filter_criteria.get('dataset_name', 'Unknown'),
            judge_type=filter_criteria.get('judge_type', 'Unknown'),
            mean_similarity=float(np.mean(one_shot_similarities)),
            std_similarity=float(np.std(one_shot_similarities)),
            median_similarity=float(np.median(one_shot_similarities)),
            min_similarity=float(np.min(one_shot_similarities)),
            max_similarity=float(np.max(one_shot_similarities)),
            num_samples=len(one_shot_similarities),
            category_results=one_shot_category_results,
            question_type_results=one_shot_question_type_results
        )
    
    def compare_attack_methods(self, attack_configs: List[Dict[str, Any]], batch_size: int = 64) -> SemanticComparisonResult:
        """
        Compare semantic preservation across multiple attack methods.
        
        Args:
            attack_configs: List of attack configurations with 'name' and optional filter criteria
            
        Returns:
            SemanticComparisonResult with comparison results
        """
        results = []
        one_shot_results = []
        
        for config in attack_configs:
            attack_name = config['name']
            filter_criteria = config.get('filter_criteria', {})
            
            logger.info(f"Analyzing semantic preservation for: {attack_name}")
            result, one_shot_result = self.analyze_attack_method(attack_name, filter_criteria, batch_size)
            results.append(result)
            one_shot_results.append(one_shot_result)
        
        # merge the results
        results.extend(one_shot_results) 
        return SemanticComparisonResult(results=results)
    
    def generate_summary_table(self, comparison_result: SemanticComparisonResult) -> str:
        """
        Generate markdown summary table for semantic preservation results.
        
        Args:
            comparison_result: Comparison results across attack methods
            
        Returns:
            Markdown table string
        """
        if not comparison_result.results:
            return "No results to display."
        
        table_lines = []
        table_lines.append("| Attack Method | Dataset | Judge Type | Mean Similarity ↑ | Std Dev | Median | Min | Max | Samples |")
        table_lines.append("| ------------- | ------- | ---------- | ----------------- | ------- | ------ | --- | --- | ------- |")
        
        for result in comparison_result.results:
            table_lines.append(
                f"| {result.attack_method:13} | {result.dataset_name:7} | {result.judge_type:10} | "
                f"{result.mean_similarity:.3f} | {result.std_similarity:.3f} | {result.median_similarity:.3f} | "
                f"{result.min_similarity:.3f} | {result.max_similarity:.3f} | {result.num_samples:7} |"
            )
        
        return "\n".join(table_lines)
    
    def generate_category_breakdown_table(self, comparison_result: SemanticComparisonResult) -> str:
        """
        Generate detailed category breakdown table.
        
        Args:
            comparison_result: Comparison results across attack methods
            
        Returns:
            Markdown table string with category breakdowns
        """
        if not comparison_result.results:
            return "No results to display."
        
        # Get all categories across all results
        all_categories = set()
        for result in comparison_result.results:
            all_categories.update(result.category_results.keys())
        
        all_categories = sorted(list(all_categories))
        
        if not all_categories:
            return "No category data found."
        
        # Create table header
        header_parts = ["| Attack Method"]
        for category in all_categories:
            short_name = category.split()[0] if category else "Unknown"  # Use first word
            header_parts.append(f"| {short_name} ↑")
        header_parts.append("|")
        
        separator_parts = ["| -------------"]
        for _ in all_categories:
            separator_parts.append("| -------")
        separator_parts.append("|")
        
        table_lines = [
            "".join(header_parts),
            "".join(separator_parts)
        ]
        
        # Add data rows
        for result in comparison_result.results:
            row_parts = [f"| {result.attack_method}"]
            
            for category in all_categories:
                if category in result.category_results:
                    similarity = result.category_results[category]
                    row_parts.append(f"| {similarity:.3f}")
                else:
                    row_parts.append("| —")
            
            row_parts.append("|")
            table_lines.append("".join(row_parts))
        
        return "\n".join(table_lines)
    
    def generate_question_type_breakdown_table(self, comparison_result: SemanticComparisonResult) -> str:
        """
        Generate question type breakdown table.
        
        Args:
            comparison_result: Comparison results across attack methods
            
        Returns:
            Markdown table string with question type breakdowns
        """
        if not comparison_result.results:
            return "No results to display."
        
        # Get all question types across all results
        all_question_types = set()
        for result in comparison_result.results:
            all_question_types.update(result.question_type_results.keys())
        
        all_question_types = sorted(list(all_question_types))
        
        if not all_question_types:
            return "No question type data found."
        
        table_lines = []
        table_lines.append("| Attack Method | " + " | ".join([f"{qt} ↑" for qt in all_question_types]) + " |")
        table_lines.append("| ------------- | " + " | ".join(["-------" for _ in all_question_types]) + " |")
        
        # Add data rows
        for result in comparison_result.results:
            row_parts = [result.attack_method]
            
            for question_type in all_question_types:
                if question_type in result.question_type_results:
                    similarity = result.question_type_results[question_type]
                    row_parts.append(f"{similarity:.3f}")
                else:
                    row_parts.append("—")
            
            table_lines.append("| " + " | ".join(row_parts) + " |")
        
        return "\n".join(table_lines)
    
    def generate_comprehensive_report(self, comparison_result: SemanticComparisonResult) -> str:
        """
        Generate comprehensive markdown report with all breakdowns.
        
        Args:
            comparison_result: Comparison results across attack methods
            
        Returns:
            Complete markdown report
        """
        if not comparison_result.results:
            return "No results to display."
        
        report_lines = ["# Semantic Preservation Analysis Report\n"]
        
        # Summary statistics
        report_lines.append("## Summary Statistics")
        report_lines.append("")
        report_lines.append("Higher scores indicate better semantic preservation (1.0 = perfect preservation, 0.0 = complete divergence).")
        report_lines.append("")
        
        summary_table = self.generate_summary_table(comparison_result)
        report_lines.append(summary_table)
        report_lines.append("")
        
        # Question type breakdown
        report_lines.append("## Semantic Preservation by Question Type")
        report_lines.append("")
        report_lines.append("Breakdown of semantic preservation scores by question type (Objective vs Subjective):")
        report_lines.append("")
        
        question_type_table = self.generate_question_type_breakdown_table(comparison_result)
        report_lines.append(question_type_table)
        report_lines.append("")
        
        # Category breakdown
        report_lines.append("## Semantic Preservation by Category")
        report_lines.append("")
        report_lines.append("Breakdown of semantic preservation scores across different question categories:")
        report_lines.append("")
        
        category_table = self.generate_category_breakdown_table(comparison_result)
        report_lines.append(category_table)
        report_lines.append("")
        
        # Key insights
        report_lines.append("## Key Insights")
        report_lines.append("")
        
        # Find best and worst performing methods
        best_method = max(comparison_result.results, key=lambda x: x.mean_similarity)
        worst_method = min(comparison_result.results, key=lambda x: x.mean_similarity)
        
        report_lines.append(f"- **Best semantic preservation**: {best_method.attack_method} (mean similarity: {best_method.mean_similarity:.3f})")
        report_lines.append(f"- **Worst semantic preservation**: {worst_method.attack_method} (mean similarity: {worst_method.mean_similarity:.3f})")
        
        # Calculate overall statistics
        all_similarities = [result.mean_similarity for result in comparison_result.results]
        overall_mean = np.mean(all_similarities)
        overall_std = np.std(all_similarities)
        
        report_lines.append(f"- **Overall mean similarity**: {overall_mean:.3f} ± {overall_std:.3f}")
        
        # Question type insights
        report_lines.append("")
        report_lines.append("### Question Type Analysis")
        
        # Analyze objective vs subjective patterns
        objective_similarities = []
        subjective_similarities = []
        
        for result in comparison_result.results:
            if 'Objective' in result.question_type_results:
                objective_similarities.append(result.question_type_results['Objective'])
            if 'Subjective' in result.question_type_results:
                subjective_similarities.append(result.question_type_results['Subjective'])
        
        if objective_similarities and subjective_similarities:
            obj_mean = np.mean(objective_similarities)
            subj_mean = np.mean(subjective_similarities)
            
            if obj_mean > subj_mean:
                report_lines.append(f"- **Objective questions** show better semantic preservation (mean: {obj_mean:.3f}) than **subjective questions** (mean: {subj_mean:.3f})")
            else:
                report_lines.append(f"- **Subjective questions** show better semantic preservation (mean: {subj_mean:.3f}) than **objective questions** (mean: {obj_mean:.3f})")
        
        # Sample size information
        total_samples = sum(result.num_samples for result in comparison_result.results)
        report_lines.append(f"- **Total response pairs analyzed**: {total_samples:,}")
        report_lines.append("")
        
        return "\n".join(report_lines)


def main():
    """Main function for semantic preservation analysis."""
    parser = argparse.ArgumentParser(description='Analyze semantic preservation of attack methods')
    parser.add_argument('--trajectory_dir', type=str, default='/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories',
                       help='Directory containing trajectory files')
    parser.add_argument('--data_dir', type=str, default='/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data',
                       help='Directory containing dataset files with metadata')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file to save analysis results (optional)')
    parser.add_argument('--dataset_name', type=str, default='AlpacaEval',
                       help='Dataset to filter by')
    parser.add_argument('--judge_type', type=str, default='pointwise', 
                       choices=['pointwise', 'alpaca_eval', 'arena_hard_auto', 'mlr_bench'],
                       help='Judge type to filter by')
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed category breakdowns')
    parser.add_argument('--summary_only', action='store_true',
                       help='Generate only summary table (faster)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for embedding inference (default: 64)')
    parser.add_argument('--question_types', action='store_true',
                       help='Include question type breakdown (Objective vs Subjective)')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Generate comprehensive report with all breakdowns and insights')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Define attack configurations
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
    
    print("🔬 SEMANTIC PRESERVATION ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Judge Type: {args.judge_type}")
    print(f"Trajectory Directory: {args.trajectory_dir}")
    print("=" * 80)
    
    try:
        # Initialize analyzer
        analyzer = SemanticPreservationAnalyzer(args.trajectory_dir, args.data_dir)
        
        # Perform analysis
        print(f"\n🚀 Starting semantic preservation analysis (batch_size={args.batch_size})...")
        comparison_result = analyzer.compare_attack_methods(attack_configs, batch_size=args.batch_size)
        
        # Generate results based on options
        if args.comprehensive:
            print("\n📋 COMPREHENSIVE ANALYSIS")
            print("=" * 50)
            comprehensive_report = analyzer.generate_comprehensive_report(comparison_result)
            print(comprehensive_report)
            
            output_content = comprehensive_report
            
        elif args.summary_only:
            print("\n📊 SUMMARY TABLE")
            print("=" * 50)
            summary_table = analyzer.generate_summary_table(comparison_result)
            print(summary_table)
            
            output_content = f"# Semantic Preservation Analysis\n\n## Summary\n\n{summary_table}\n"
            
        else:
            # Default: summary + optional breakdowns
            print("\n📊 SUMMARY TABLE")
            print("=" * 50)
            summary_table = analyzer.generate_summary_table(comparison_result)
            print(summary_table)
            
            output_content = f"# Semantic Preservation Analysis\n\n## Summary\n\n{summary_table}\n"
            
            # Question type breakdown
            if args.question_types:
                print("\n\n📋 QUESTION TYPE BREAKDOWN")
                print("=" * 50)
                question_type_table = analyzer.generate_question_type_breakdown_table(comparison_result)
                print(question_type_table)
                
                output_content += f"\n## Question Type Breakdown\n\n{question_type_table}\n"
            
            # Category breakdown
            if args.detailed:
                print("\n\n📋 CATEGORY BREAKDOWN")
                print("=" * 50)
                category_table = analyzer.generate_category_breakdown_table(comparison_result)
                print(category_table)
                
                output_content += f"\n## Category Breakdown\n\n{category_table}\n"
        
        # Save to file if specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(output_content)
            print(f"\n💾 Results saved to: {args.output_file}")
        
        print("\n✅ Semantic preservation analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        logging.error(f"Analysis failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()