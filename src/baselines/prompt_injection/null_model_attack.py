import re
import argparse
import logging
import asyncio
import time
import os
import json
from src.logging_utils import setup_logging
from src.results_analysis.results_loader.trajectory_loader import load_trajectory_directory, parse_filter_criteria, parse_exclude_criteria
from src.llm_evaluator import load_judge_model, get_judge_type, JudgeType
from src.evolve_agent.bandit.reward_cal import create_reward_calculator

from src.baselines.prompt_injection.base import BasePromptInjectionBaseline
from src.llm_zoo.api_zoo import get_model_name
from src.evolve_agent.utils import get_result_analysis, save_result_analysis, save_trajectories
from src.baselines.prompt_injection.utils import get_implementation_name
from src.data.data_utils import load_dataset_for_exploration

logger = logging.getLogger(__name__)

# Import null model strings from null_model.py
from src.baselines.prompt_injection.null_model import (
    pointwise_null_model,
    alpaca_eval_null_model,
    alpaca_eval_null_model_RS,
    arenahard_null_model,
    arenahard_null_model_RS,
    mtbench_null_model,
    mtbench_null_model_RS
)

NULL_MODEL_MAPPING = {
    "pointwise": pointwise_null_model,
    "alpaca_eval": alpaca_eval_null_model,
    "alpaca_eval_rs": alpaca_eval_null_model_RS,
    "arenahard": arenahard_null_model,
    "arenahard_rs": arenahard_null_model_RS,
    "mtbench": mtbench_null_model,
    "mtbench_rs": mtbench_null_model_RS,
}

class NullModelAttack(BasePromptInjectionBaseline):
    """Null model attack - replaces the answer with predefined null model strings"""
    
    def __init__(self, judge_type: str, use_rs_variant: bool = False):
        """
        Args:
            judge_type: The judge type (e.g., "alpaca_eval", "arenahard", "mt_bench", "mlr_bench")
            use_rs_variant: Whether to use the RS (random suffix) variant
        """
        self.judge_type = judge_type.lower()
        self.use_rs_variant = use_rs_variant
        
        # Normalize judge type to mapping key
        # Handle special cases: mt_bench -> mtbench, mlr_bench -> pointwise
        normalized_type = self.judge_type
        if normalized_type == "mt_bench" or normalized_type == "mt_bench_reference_guided":
            normalized_type = "mtbench"
        elif normalized_type == "mlr_bench":
            # MLR bench uses pointwise null model
            normalized_type = "pointwise"
        
        # Determine the key for null model mapping
        key = normalized_type
        if self.use_rs_variant:
            key += "_rs"
            
        if key not in NULL_MODEL_MAPPING:
            # Fallback to non-RS variant if RS not available
            key = normalized_type
            if key not in NULL_MODEL_MAPPING:
                logger.warning(f"No null model found for judge type: {self.judge_type}, using pointwise as fallback")
                key = "pointwise"
                
        self.null_model_string = NULL_MODEL_MAPPING[key]
    
    def generate_attack_prompt(self, benign_question: str) -> str:
        """Replace the original answer with the null model string"""
        return self.null_model_string


async def main():
    parser = argparse.ArgumentParser(description="Null model attack for LLM-as-a-Judge")
    parser.add_argument("--directory", type=str,
                       help="Directory containing trajectory files",
                       default="/data2/xianglin/A40/llm-as-a-judge-attack/trajectories")
    parser.add_argument("--data_dir", type=str,
                       help="Directory containing dataset",
                       default="/data2/xianglin/A40/llm-as-a-judge-attack/data")
    parser.add_argument("--filter", type=str, 
                       help="Include only files matching criteria (format: 'key1=value1,key2=value2'). "
                            "Example: --filter 'strategy=ucb,dataset_name=AlpacaEval'")
    parser.add_argument("--exclude", type=str, 
                       help="Exclude files matching criteria (format: 'key1=value1,key2=value2'). "
                            "Example: --exclude 'strategy=random,judge_backbone=gpt-3.5'")
    parser.add_argument("--output_dir", type=str, default="./reports",
                       help="Output directory to save results")
    parser.add_argument("--save_analysis_path", type=str, default="./results/",
                       help="Path to save analysis results")
    parser.add_argument("--save_trajectory_path", type=str, default="/data2/xianglin/A40/llm-as-a-judge-attack/trajectories",
                       help="Path to save trajectory results")
    parser.add_argument("--use_rs_variant", default=0, help="Use RS (random suffix) variant of null models")
         
    
    args = parser.parse_args()

    setup_logging(task_name="null_model_attack")
    start_time = time.time()

    # 0. Parse filter and exclude criteria
    general_filter_criteria = parse_filter_criteria(args.filter) if args.filter else {}
    general_exclude_criteria = parse_exclude_criteria(args.exclude) if args.exclude else {}

    # 1. Load UCB trajectories
    ucb_filter_criteria = parse_filter_criteria("strategy=ucb")
    ucb_filter_criteria.update(general_filter_criteria)

    logging.info(f"UCB filter criteria: {ucb_filter_criteria}")

    # 2. Load UCB trajectory dataset
    ucb_trajectories = load_trajectory_directory(
        directory=args.directory,
        filter_criteria=ucb_filter_criteria,
        exclude_criteria=general_exclude_criteria
    )

    # Log settings: judge type+dataset name+judge backbone+llm agent name+baseline response model name+response model name
    logging.info(f"Loaded {len(ucb_trajectories)} UCB trajectories")
    for traj in ucb_trajectories:
        logging.info(f"{traj.metadata.judge_type}, {traj.metadata.dataset_name}, {traj.metadata.judge_backbone}, {traj.metadata.llm_agent_name}, Baseline: {traj.metadata.baseline_response_model_name}, ({traj.metadata.answer_position})")

    if not ucb_trajectories:
        logger.error("No UCB trajectories found. Please check your filter criteria.")
        return

    for traj_idx, traj in enumerate(ucb_trajectories):
             
        logger.info(f"\nProcessing trajectory {traj_idx + 1}/{len(ucb_trajectories)}")

        # Extract data for processing from this trajectory
        original_answers = [item.initial_answer for item in traj.trajectories]
        original_scores = [item.initial_score for item in traj.trajectories]
        questions = [item.question for item in traj.trajectories]

        # Generate null model attack prompts (replace answers with null model strings)
        null_model_attack = NullModelAttack(traj.metadata.judge_type, args.use_rs_variant)
        new_answers = [null_model_attack.generate_attack_prompt(original_answer) for original_answer in original_answers]
        
        # Load judge model and evaluate null model attack answers
        judge_type = get_judge_type(traj.metadata.judge_type)
        judge_model_name = get_implementation_name(traj.metadata.judge_backbone)
        logger.info(f"  Evaluating {len(new_answers)} null model attack answers...")

        # Handle different judge types similar to naive.py
        # if pointwise, baseline response is the original answer
        if judge_type in [JudgeType.POINTWISE]:
            reward_calculator = create_reward_calculator(judge_type, judge_model_name, "absolute")
            attack_scores, _, attack_explanations = await reward_calculator.calculate_batch_reward(questions, new_answers, original_scores)
        elif judge_type in [JudgeType.PAIRWISE, JudgeType.PAIRWISE_FINE_GRAINED, JudgeType.ALPACA_EVAL, JudgeType.ARENA_HARD_AUTO]:
            reward_calculator = create_reward_calculator(judge_type, judge_model_name, "absolute", answer_position=traj.metadata.answer_position)
            baseline_dataset = load_dataset_for_exploration(args.data_dir, traj.metadata.dataset_name, traj.metadata.baseline_response_model_name, traj.metadata.judge_backbone)
            baseline_dataset_mapping = {item['instruction']: item['output'] for item in baseline_dataset}
            baseline_responses = [baseline_dataset_mapping[question] for question in questions]
            attack_scores, _, attack_explanations = await reward_calculator.calculate_batch_reward(questions, new_answers, original_scores, baseline_responses)
        else:
            logger.warning(f"Unknown judge type: {traj.metadata.judge_type}, defaulting to pointwise evaluation")
            reward_calculator = create_reward_calculator(judge_type, judge_model_name, "absolute")
            attack_scores, _, attack_explanations = await reward_calculator.calculate_batch_reward(questions, new_answers, original_scores)
        
        # Create result items for this trajectory
        traj_results = []
        for i in range(len(traj.trajectories)):
            original_score = traj.trajectories[i].initial_score
            if original_score >= attack_scores[i]:
                result_item = {
                    "category": traj.trajectories[i].category,
                    "instruction": traj.trajectories[i].question,
                    "output": traj.trajectories[i].initial_answer,
                    "original_score": float(original_score),
                    "original_explanation": traj.trajectories[i].history[0].explanation,
                    "final_score": original_score,
                    "final_explanation": traj.trajectories[i].history[0].explanation,
                    "final_response": traj.trajectories[i].initial_answer,
                    "exploration_length": 1,
                    "skip": 0,
                }
            else:
                result_item = {
                    "category": traj.trajectories[i].category,
                    "instruction": traj.trajectories[i].question,
                    "output": new_answers[i],
                    "original_score": float(traj.trajectories[i].initial_score),
                    "original_explanation": traj.trajectories[i].history[0].explanation,
                    "final_score": float(attack_scores[i]),
                    "final_explanation": attack_explanations[i],
                    "final_response": new_answers[i],
                    "exploration_length": 2,
                    "skip": 0,
                }
            traj_results.append(result_item)
    
        traj_meta_info = {
            "strategy": f"null_model_attack{'_rs' if args.use_rs_variant else ''}",
            "judge_type": traj.metadata.judge_type,
            "answer_position": traj.metadata.answer_position,
            "dataset_name": traj.metadata.dataset_name,
            "judge_backbone": traj.metadata.judge_backbone,
            "baseline_response_model_name": traj.metadata.baseline_response_model_name,
            "llm_agent_name": None,
            "response_model_name": traj.metadata.response_model_name,
            "budget": 1,
            "pool_size": 1,
            "eval_num": len(traj.trajectories),
            "reward_type": traj.metadata.reward_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_taken": time.time() - start_time,
        }

        analysis = get_result_analysis(traj_results)
        analysis.update(traj_meta_info)
        
        traj_to_save = traj_meta_info.copy()
        traj_to_save.update({"trajectories": traj_results})

        save_result_analysis(analysis, args.save_analysis_path)
        strategy_name = f"null_model_attack{'_rs' if args.use_rs_variant else ''}"
        save_trajectories(traj_to_save, args.save_trajectory_path, strategy_name)
     
    logger.info(f"Null model attack completed!")


if __name__ == "__main__":
    asyncio.run(main())
