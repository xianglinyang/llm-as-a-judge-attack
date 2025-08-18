import numpy as np
import logging
import argparse
import time
import asyncio
import os

from src.evolve_agent.bandit.UCB import ContextualLinUCBAgent
from src.llm_evaluator import JudgeType
from src.text_encoder import MiniLMTextEncoder
from src.llm_zoo import load_model
from src.logging_utils import setup_logging
from src.evolve_agent.utils import (prepare_dataset_for_exploration, 
                                    exclude_perfect_response, 
                                    sample_and_filter_data, 
                                    save_result_analysis, 
                                    save_metrics)
from src.llm_zoo.api_zoo import get_model_name
from src.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def get_warmup_model_path(args):
    """Create organized directory structure for warmup models."""
    
    # Clean model names (remove slashes, special chars)
    judge_clean = get_model_name(args.judge_model_name)
    response_clean = get_model_name(args.response_model_name)
    baseline_clean = get_model_name(args.baseline_response_model_name)
    
    # Hierarchical organization
    path_parts = [
        args.save_model_path,
        "warmup_models",
        f"judge_{judge_clean}",
        f"response_{response_clean}", 
        f"baseline_{baseline_clean}",
        f"reward_{args.reward_type}",
        f"alpha_{args.alpha}_lambda_{args.lambda_reg}",
        f"ci_{args.ci_width_threshold}_patience_{args.patience}",
        time.strftime("%Y%m%d_%H%M%S")  # timestamp for uniqueness
    ]
    
    return os.path.join(*path_parts)


async def main(args):
    llm_agent = load_model(args.llm_agent_name)
    embedding_model = MiniLMTextEncoder()
    judge_type = JudgeType(args.judge_type)
    judge_model_backbone = args.judge_model_name
    save_path = get_warmup_model_path(args)

    # === dataset prep (exactly like your explore main) ===
    (
        question_list,
        init_response_list,
        category_list,
        original_score_list,
        original_explanation_list,
        baseline_response_list,
    ) = await prepare_dataset_for_exploration(
        args.data_dir,
        args.dataset_name,
        args.response_model_name,
        judge_type,
        judge_model_backbone,
        args.baseline_response_model_name,
        args.answer_position,
    )

    test_results, selected_idxs = exclude_perfect_response(
        judge_type,
        question_list,
        init_response_list,
        category_list,
        original_score_list,
        original_explanation_list,
        baseline_response_list,
    )
    logger.info(f"Skipped {len(test_results)} samples")
    logger.info(f"Dataset for warm-up: {len(selected_idxs)} samples...")

    (
        eval_num,
        selected_idxs,
        question_list,
        init_response_list,
        original_score_list,
        original_explanation_list,
        category_list,
        baseline_response_list,
    ) = sample_and_filter_data(
        selected_idxs,
        args.eval_num,
        question_list,
        init_response_list,
        original_score_list,
        original_explanation_list,
        category_list,
        baseline_response_list,
    )

    # === agent init (same signature you use elsewhere) ===
    logger.info(f"Initializing the agent for warm-up...")
    agent = ContextualLinUCBAgent(
        args.n_features,
        llm_agent,
        embedding_model,
        judge_type,
        judge_model_backbone,
        args.reward_type,
        args.alpha,
        args.lambda_reg,
        args.answer_position,
    )
    logger.info("Agent initialized.")
    logger.info("-" * 100)

    start_time = time.time()

    # === run global warm-up (ONE model shared across questions) ===
    logger.info(
        f"Running global warm-up with burnin_passes={args.burnin_passes}, "
        f"ucb_passes={args.ucb_passes}, epsilon={args.epsilon}, alpha={args.alpha}, "
        f"lambda_reg={args.lambda_reg}..."
    )


    summary = await agent.warmup(
        question_list=question_list,
        init_response_list=init_response_list,
        original_score_list=original_score_list,
        original_explanation_list=original_explanation_list,
        baseline_response_list=baseline_response_list,
        burnin_passes=args.burnin_passes,
        ucb_passes=args.ucb_passes,
        epsilon_schedule=[args.epsilon] * max(1, args.ucb_passes),
        enable_ci_early_stop=True,
        ci_width_threshold=args.ci_width_threshold,
        patience=args.patience,
        save_path=save_path,
    )

    end_time = time.time()
    logger.info("Warm-up finished.")
    logger.info("-" * 100)

    # === assemble analysis meta, save ===
    analysis = {
        "strategy": "UCB-Warmup",
        "phase": summary.get("phase", "complete"),
        "rounds": summary.get("rounds", 0),
        "median_ci": summary.get("median_ci", []),
        "median_gap": summary.get("median_gap", []),
        "judge_type": args.judge_type,
        "answer_position": args.answer_position,
        "dataset_name": args.dataset_name,
        "judge_backbone": get_model_name(args.judge_model_name),
        "baseline_response_model_name": get_model_name(args.baseline_response_model_name),
        "llm_agent_name": get_model_name(args.llm_agent_name),
        "response_model_name": get_model_name(args.response_model_name),
        "lambda_reg": args.lambda_reg,
        "n_features": args.n_features,
        "n_arms": args.n_arms,
        "burnin_passes": args.burnin_passes,
        "ucb_passes": args.ucb_passes,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "ci_width_threshold": args.ci_width_threshold,
        "patience": args.patience,
        "eval_num": eval_num,
        "reward_type": args.reward_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_taken": end_time - start_time,
        "saved_model_path": save_path,
    }

    # Save warm-up analysis + metrics (use same helpers as exploration for consistency)
    save_result_analysis(analysis, args.save_analysis_path)
    save_metrics({"warmup_summary": analysis}, args.save_metrics_path, f"init_ucb_warmup")

    logger.info(
        f"Total time taken: {end_time - start_time:.2f}s for warm-up with "
        f"{args.burnin_passes} burn-in passes and {args.ucb_passes} UCB passes "
        f"on {eval_num} samples. Model saved to: {save_path}"
    )


if __name__ == "__main__":
    setup_logging(task_name="UCB-Warmup")

    parser = argparse.ArgumentParser()
    # same core args as your explore script
    parser.add_argument("--judge_model_name", type=str, default="qwen/qwen3-235b-a22b-2507")
    parser.add_argument("--judge_type", type=str, default="pointwise",
                        choices=["pointwise", "pairwise", "pairwise_fine_grained", "alpaca_eval", "arena_hard_auto", "mt_bench", "mlr_bench"])
    parser.add_argument("--answer_position", type=str, default=None,
                        choices=["first", "second", None])
    parser.add_argument("--baseline_response_model_name", type=str, default=None)
    parser.add_argument("--llm_agent_name", type=str, default="gpt-4.1-nano")
    parser.add_argument("--response_model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--dataset_name", type=str, default="UltraFeedback")
    parser.add_argument("--reward_type", type=str, default="relative", choices=["relative", "absolute"])
    parser.add_argument("--n_features", type=int, default=384)
    parser.add_argument("--n_arms", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data")
    parser.add_argument("--eval_num", type=int, default=3)

    # warm-up specific
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--burnin_passes", type=int, default=1)
    parser.add_argument("--ucb_passes", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--ci_width_threshold", type=float, default=0.12)
    parser.add_argument("--patience", type=int, default=2)

    # saving/paths
    parser.add_argument("--save_model_path", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/models/")
    parser.add_argument("--save_analysis_path", type=str, default="results/")
    parser.add_argument("--save_metrics_path", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/metrics/")

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    finally:
        logger.info("Warm-up application finished.")
