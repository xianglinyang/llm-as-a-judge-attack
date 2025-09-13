'''
Transfer Analysis for LLM-as-a-Judge Attacks

This module implements transfer analysis to evaluate how well attacks optimized 
for one judge model transfer to other judge models.

Model-level transfer workflow:
1. Run BITE end-to-end on Judge A until it converges on a high-performing answer.
2. Without any further adaptation, evaluate those final answers on Judge B.

Metrics:
- Score Lift Transfer Rate (SLTR): SLTR = Score Lift on Target Judge / Score Lift on Source Judge
- Attack Success Rate (ASR) on target vs. source.
- Transfer Effectiveness: How well attacks generalize across judges
'''
import os
import sys
import numpy as np
import asyncio
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from src.results_analysis.results_loader.trajectory_loader import (
    TrajectoryLoader, LoadedTrajectory
)
from src.llm_evaluator import load_judge_model, get_judge_type
from src.data.data_utils import load_dataset
from src.llm_zoo.api_zoo import get_model_name
from src.logging_utils import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class TransferResult:
    """Container for transfer analysis results."""
    source_judge: str
    target_judge: str
    dataset_name: str
    source_asr: float
    target_asr: float
    transfer_asr: float  # ASR when source responses evaluated on target judge
    source_sltr: float
    target_sltr: float
    transfer_sltr: float  # SLTR when source responses evaluated on target judge
    transfer_effectiveness: float  # target_sltr / source_sltr
    num_questions: int


def get_sltr(initial_source_score: np.ndarray, final_source_score: np.ndarray, 
             initial_target_score: np.ndarray, final_target_score: np.ndarray) -> float:
    """Calculate Score Lift Transfer Rate."""
    src_lift = np.sum(final_source_score - initial_source_score)
    tgt_lift = np.sum(final_target_score - initial_target_score)
    return tgt_lift / src_lift if src_lift != 0 else np.nan


def get_asr(initial_source_score: np.ndarray, final_source_score: np.ndarray, 
            initial_target_score: np.ndarray, final_target_score: np.ndarray) -> float:
    """Calculate Attack Success Rate."""
    src_lift = final_source_score - initial_source_score
    tgt_lift = final_target_score - initial_target_score
    
    idxs = np.where(src_lift > 0)[0]
    if len(idxs) == 0:
        return np.nan
    
    success = np.sum(tgt_lift[idxs] > 0)
    return success / len(idxs)


def calculate_basic_asr(initial_scores: np.ndarray, final_scores: np.ndarray) -> float:
    """Calculate basic ASR for a single judge."""
    improvements = final_scores - initial_scores
    success_count = np.sum(improvements > 0)
    return (success_count / len(improvements)) * 100.0 if len(improvements) > 0 else 0.0


def calculate_basic_sltr(initial_scores: np.ndarray, final_scores: np.ndarray) -> float:
    """Calculate basic SLTR (average score lift) for a single judge."""
    improvements = final_scores - initial_scores
    return float(np.mean(improvements))


class TransferAnalyzer:
    """Main class for transfer analysis."""
    
    def __init__(self, trajectory_dir: str, data_dir: str=None):
        """
        Initialize transfer analyzer.
        
        Args:
            trajectory_dir: Directory containing trajectory files
            data_dir: Directory containing data files
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.data_dir = Path(data_dir) if data_dir else None
        self.trajectory_dir = Path(trajectory_dir)
        self.loader = TrajectoryLoader(str(trajectory_dir))
        
    def extract_responses_and_questions(self, trajectories: List[LoadedTrajectory]) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract questions, initial responses, and final responses from trajectories.
        
        Args:
            trajectories: List of trajectory data
            
        Returns:
            Tuple of (questions, initial_responses, final_responses)
        """
        questions = []
        initial_responses = []
        final_responses = []
        initial_scores = []
        final_scores = []
        
        for traj in trajectories:
            for item in traj.trajectories:
                questions.append(item.question)
                initial_responses.append(item.initial_answer)
                final_responses.append(item.final_answer)
                initial_scores.append(item.initial_score)
                final_scores.append(item.final_score)
                
        return questions, initial_responses, final_responses, initial_scores, final_scores
        
    async def evaluate_responses_with_judge(self, 
                                            questions: List[str], 
                                            responses: List[str], 
                                            judge_model_implementation_name: str, 
                                            judge_type: str,
                                            answer_position: Optional[str] = None,
                                            baseline_model_name: Optional[str] = None,
                                            dataset_name: Optional[str] = None,
                                            ) -> Tuple[List[float], List[str]]:
        
        judge_type_enum = get_judge_type(judge_type)
        judge_model = load_judge_model(judge_type_enum, judge_model_implementation_name)

        # load baseline response if applicable
        if baseline_model_name:
            # new_dataset: list [
            # {
            #     "instruction": str,
            #     "output": str,
            #     "category": str,
            # },
            baseline_dataset = load_dataset(self.data_dir, dataset_name, baseline_model_name)
            question2output_mapping = {item["instruction"]: item["output"] for item in baseline_dataset}
            baseline_responses = [question2output_mapping[question] for question in questions]
            if answer_position == "first":
                scores, explanations = await judge_model.batch_get_score(questions, responses, baseline_responses)
            elif answer_position == "second":
                scores, explanations = await judge_model.batch_get_score(questions, baseline_responses, responses)
                scores = [-score for score in scores]
            else:
                raise ValueError(f"Invalid answer position: {answer_position}")
        else:
            scores, explanations = await judge_model.batch_get_score(questions, responses)
        
        return scores, explanations

        
    def load_trajectories_for_judge(self, source_judge: str, 
                                  filter_criteria: Optional[Dict[str, Any]] = None) -> List[LoadedTrajectory]:
        """
        Load trajectories for a specific source judge.
        
        Args:
            source_judge: Source judge model name
            filter_criteria: Additional filtering criteria
            
        Returns:
            List of trajectories from the source judge
        """
        trajectories = self.loader.load_trajectories()
        
        # Filter by source judge
        judge_criteria = {"judge_backbone": source_judge}
        if filter_criteria:
            judge_criteria.update(filter_criteria)
            
        filtered_trajectories = self.loader.filter_trajectories(trajectories, **judge_criteria)
        
        logger.info(f"Loaded {len(filtered_trajectories)} trajectories for source judge: {source_judge}")
        return filtered_trajectories
        
    async def analyze_transfer(self, source_judge: str, target_judge: str,
                             strategy: str = "ucb", 
                             dataset_name: Optional[str] = None,
                             judge_type: str = "pointwise",
                             answer_position: Optional[str] = None,
                             baseline_model_name: Optional[str] = None) -> TransferResult:
        """
        Analyze transfer performance between two judges.
        
        Args:
            source_judge: Source judge model name
            target_judge: Target judge model name  
            strategy: Attack strategy to analyze
            dataset_name: Dataset to filter by (optional)
            judge_type: Type of judge evaluation
            
        Returns:
            TransferResult containing all metrics
        """
        logger.info(f"Analyzing transfer from {get_model_name(source_judge)} to {get_model_name(target_judge)}")
        
        # Build filter criteria
        filter_criteria = {"strategy": strategy}
        if dataset_name:
            filter_criteria["dataset_name"] = dataset_name
            
        # Load source judge trajectories
        source_trajectories = self.load_trajectories_for_judge(get_model_name(source_judge), filter_criteria)
        
        if not source_trajectories:
            raise ValueError(f"No trajectories found for source judge {get_model_name(source_judge)}")
            
        # Extract questions and responses
        questions, initial_responses, final_responses, source_initial_scores, source_final_scores = self.extract_responses_and_questions(source_trajectories)
        
        if not questions:
            raise ValueError("No questions found in trajectories")
            
        logger.info(f"Evaluating {len(questions)} question-response pairs")
        
        # Get target judge scores
        logger.info("Evaluating with target judge...")
        target_initial_scores, _ = await self.evaluate_responses_with_judge(
            questions, initial_responses, target_judge, judge_type, answer_position, get_model_name(baseline_model_name)
        )
        target_final_scores, _ = await self.evaluate_responses_with_judge(
            questions, final_responses, target_judge, judge_type, answer_position, get_model_name(baseline_model_name)
        )
        
        # Convert to numpy arrays
        source_initial_scores = np.array(source_initial_scores)
        source_final_scores = np.array(source_final_scores)
        target_initial_scores = np.array(target_initial_scores)
        target_final_scores = np.array(target_final_scores)
        
        # Calculate metrics
        source_asr = calculate_basic_asr(source_initial_scores, source_final_scores)
        target_asr = calculate_basic_asr(target_initial_scores, target_final_scores)
        transfer_asr = get_asr(source_initial_scores, source_final_scores, 
                              target_initial_scores, target_final_scores) * 100.0
        
        source_sltr = calculate_basic_sltr(source_initial_scores, source_final_scores)
        target_sltr = calculate_basic_sltr(target_initial_scores, target_final_scores)
        transfer_sltr = get_sltr(source_initial_scores, source_final_scores,
                                target_initial_scores, target_final_scores)
        
        transfer_effectiveness = transfer_sltr / source_sltr if source_sltr != 0 else np.nan
        
        
        return TransferResult(
            source_judge=get_model_name(source_judge),
            target_judge=get_model_name(target_judge),
            dataset_name=dataset_name,
            source_asr=source_asr,
            target_asr=target_asr,
            transfer_asr=transfer_asr,
            source_sltr=source_sltr,
            target_sltr=target_sltr,
            transfer_sltr=transfer_sltr,
            transfer_effectiveness=transfer_effectiveness,
            num_questions=len(questions),
        )
                
        
    def generate_transfer_report(self, results: List[TransferResult]) -> str:
        """
        Generate a formatted transfer analysis report.
        
        Args:
            results: List of transfer results
            
        Returns:
            Formatted markdown report
        """
        if not results:
            return "No transfer results to report."
            
        report_lines = []
        report_lines.append("# Transfer Analysis Report")
        report_lines.append("")
        
        # Summary table
        report_lines.append("## Transfer Performance Summary")
        report_lines.append("")
        report_lines.append("| Source Judge | Target Judge | Dataset | Source ASR | Target ASR | Transfer ASR | Transfer Effectiveness | Questions |")
        report_lines.append("| ------------ | ------------ | ---------- | ---------- | ------------ | --------------------- | --------- |")
        
        for result in results:
            transfer_eff = f"{result.transfer_effectiveness:.2f}" if not np.isnan(result.transfer_effectiveness) else "N/A"
            report_lines.append(
                f"| {result.source_judge} | {result.target_judge} | "
                f"{result.dataset_name} | "
                f"{result.source_asr:.1f}% | {result.target_asr:.1f}% | "
                f"{result.transfer_asr:.1f}% | {transfer_eff} | {result.num_questions} |"
            )
            
        report_lines.append("")
        
        # Detailed analysis for each transfer
        for result in results:
            report_lines.append(f"## {result.source_judge} → {result.target_judge}")
            report_lines.append("")
            report_lines.append(f"- **Dataset**: {result.dataset_name}")
            report_lines.append(f"- **Source ASR**: {result.source_asr:.1f}%")
            report_lines.append(f"- **Target ASR**: {result.target_asr:.1f}%")
            report_lines.append(f"- **Transfer ASR**: {result.transfer_asr:.1f}%")
            report_lines.append(f"- **Source SLTR**: {result.source_sltr:.3f}")
            report_lines.append(f"- **Target SLTR**: {result.target_sltr:.3f}")
            report_lines.append(f"- **Transfer SLTR**: {result.transfer_sltr:.3f}")
            
            transfer_eff = result.transfer_effectiveness
            if not np.isnan(transfer_eff):
                report_lines.append(f"- **Transfer Effectiveness**: {transfer_eff:.2f} ({transfer_eff*100:.1f}%)")
            else:
                report_lines.append(f"- **Transfer Effectiveness**: N/A")
                
            report_lines.append(f"- **Questions Analyzed**: {result.num_questions}")
            report_lines.append("")
                
        return "\n".join(report_lines)


# Convenience functions
async def analyze_judge_transfer(trajectory_dir: str, source_judge: str, target_judge: str,
                               strategy: str = "ucb", dataset_name: Optional[str] = None) -> TransferResult:
    """
    Convenience function to analyze transfer between two judges.
    
    Args:
        trajectory_dir: Directory containing trajectory files
        source_judge: Source judge model name
        target_judge: Target judge model name
        strategy: Attack strategy to analyze
        dataset_name: Dataset to filter by
        
    Returns:
        TransferResult object
    """
    analyzer = TransferAnalyzer(trajectory_dir)
    return await analyzer.analyze_transfer(source_judge, target_judge, strategy, dataset_name)

async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Transfer Analysis for LLM-as-a-Judge Attacks")
    parser.add_argument("--trajectory_dir", type=str,
                        default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories",
                        help="Directory containing trajectory files")
    parser.add_argument("--source_judge", type=str, required=True,
                       help="Source judge model implementation name")
    parser.add_argument("--target_judge", type=str, required=True,
                       help="Target judge model implementation name")
    parser.add_argument("--strategy", type=str, default="ucb",
                       help="Attack strategy to analyze (default: ucb)")
    parser.add_argument("--dataset_name", type=str, default="AlpacaEval",
                       help="Dataset to filter by (default: AlpacaEval)")
    parser.add_argument("--judge_type", type=str, default="pointwise",
                       help="Judge type (default: pointwise)")
    parser.add_argument("--output_dir", type=str, default="./reports",
                       help="Output directory to save report (optional)")
    # for pairwise transfer analysis
    parser.add_argument("--answer_position", type=str, default=None,
                       help="Answer position to analyze (default: None)")
    parser.add_argument("--baseline_model_name", type=str, default=None,
                       help="Baseline model implementation name (default: None)")
    
    args = parser.parse_args()
    setup_logging(task_name="transfer_analysis")

    source_judge_name = get_model_name(args.source_judge)
    target_judge_name = get_model_name(args.target_judge)

    source_judge_implementation_name = args.source_judge
    target_judge_implementation_name = args.target_judge
    baseline_model_implementation_name = args.baseline_model_name

    analyzer = TransferAnalyzer(args.trajectory_dir)

    # Analyze single transfer
    print(f"🔄 Analyzing transfer: {source_judge_name} → {target_judge_name}")
    print(f"Strategy: {args.strategy}")
    if args.dataset_name:
        print(f"Dataset: {args.dataset_name}")
    print("=" * 80)
    
    result = await analyzer.analyze_transfer(
        source_judge=source_judge_implementation_name,
        target_judge=target_judge_implementation_name,
        strategy=args.strategy,
        dataset_name=args.dataset_name,
        judge_type=args.judge_type,
        answer_position=args.answer_position,
        baseline_model_name=baseline_model_implementation_name
    )
    results = [result]

    # Generate and display report
    report = analyzer.generate_transfer_report(results)
    print("\n📊 TRANSFER ANALYSIS REPORT")
    print("=" * 50)
    print(report)
    
    # Save to file if specified
    if args.output_dir:
        # create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        file_name = f"{source_judge_name}_to_{target_judge_name}.md"
        file_path = os.path.join(args.output_dir, file_name)
        with open(file_path, 'a') as f:
            f.write(report)
        print(f"\n💾 Report saved to: {file_path}")
    
    print("\n✅ Transfer analysis completed successfully!")
        

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

