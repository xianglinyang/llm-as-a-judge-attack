import logging
import random
import argparse
import time
import asyncio
import os

from src.evolve_agent import EvolveAgent
from src.llm_zoo import BaseLLM, load_model
from src.llm_evaluator import JudgeType
from src.logging_utils import setup_logging
from src.evolve_agent.utils import (prepare_dataset_for_exploration, 
                                    exclude_perfect_response,
                                    extract_result_from_trajectories,
                                    sample_and_filter_data,
                                    get_result_analysis, 
                                    save_result_analysis, 
                                    save_trajectories,
                                    save_metrics)
from src.evolve_agent.rewrite_template import get_template
from src.evolve_agent.utils import (_estimate_tokens, _select_from_pool_uniform, _best_item, _worst_index, 
                                    _batch_estimate_tokens, _get_pool_metrics, _find_pool_extremes)
from src.llm_zoo.api_zoo import get_model_name
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class SimpleRewriteAgent(EvolveAgent):
    '''
    Evolve the response by modifying the style and tone of the response.
    Supports both pointwise and pairwise evaluation.
    '''
    def __init__(self, llm_agent: BaseLLM, judge_type: JudgeType, judge_model_backbone: str, reward_type: str = "relative", answer_position: str = None, template_name: str = "holistic", domain: str = None):
        super().__init__(llm_agent, judge_type, judge_model_backbone, reward_type, answer_position)
        self.template_name = template_name
        self.domain = domain
        self.template = get_template(template_name, domain)
        self.evolve_strategy = f"simple_rewrite_{template_name}"

    def explore(self,
        question: str,
        init_response: str,
        original_score: float,
        original_explanation: str,
        budget: int = 5,
        pool_size: int = 2,
        baseline_response: Optional[str] = None,
        # extras / knobs:
        replacement_margin: float = 0.0,          # require s_new > s_worst + margin to enter pool
        record_rows: Optional[List[Dict[str, Any]]] = None,  # if provided, we append per-round logs
        # convergence parameters:
        convergence_patience: int = 3,            # stop if no improvement for N rounds
        min_improvement_threshold: float = 0.001, # minimum improvement to count as progress
        **kwargs
        ) -> tuple[Dict[str, Any], Dict[str, List[float]]]:
        """
        Pool-based evolution loop (shared across baselines and LinUCB):
        - Keep a pool of size <= pool_size
        - Each round: sample one answer from the pool, generate a candidate, judge it
        - If better than current worst (by margin), insert and evict worst
        - Log online metrics for plotting

        Returns a tuple containing:
        1. best_item dict:
        {
            "question": str,
            "score": float,
            "answer": str,
            "explanation": str,
            "origin": "init" | "evolved",
            "tokens": int,
            "history": List[ (score, explanation, answer, meta) ]  # per-path history
        }
        2. metrics_to_record dict: {
            "best_so_far": List[float],
            "pool_mean": List[float],
            "replacement_ratio": List[float],
            "lift_per_1k_tokens": List[float]
        }
        """
        # 0) Validate judge requirements
        self.validate_judge_requirements(baseline_response)

        # 1) Metrics containers
        metrics_to_record = {
            "best_so_far": [],
            "pool_mean": [],
            "replacement_ratio": [],
            "lift_per_1k_tokens": []
        }
        total_tokens = 0
        replacements = 0

        # 2) Initialize pool (each item tracks its own mini history)
        pool: List[Dict[str, Any]] = [{
            "question": question,
            "score": float(original_score),
            "answer": init_response,
            "explanation": original_explanation,
            "origin": "init",
            "tokens": _estimate_tokens(init_response),
            "history": [(original_score, original_explanation, init_response, self.template_name)]
        }]

        best0 = float(original_score)  # baseline for lift computation
        
        # Early stopping tracking
        no_improvement_count = 0
        last_best_score = best0

        # 3) Main loop with early stopping
        for t in range(1, budget + 1):
            # (a) Choose a seed from pool
            seed_idx = _select_from_pool_uniform(pool)
            seed = pool[seed_idx]
            current_answer = seed["answer"]

            # (b) Generate new response
            try:
                prompt = self.template.format(base_answer=current_answer)
                new_response = self.llm_agent.invoke(prompt)
            except Exception as e:
                # If LLM call fails, skip this round but still log
                new_response = None

            # (c) Judge new response
            try:
                _, new_score, new_explanation = self.get_reward(
                    question, new_response, original_score, baseline_response
                )
            except Exception:
                # On judge failure, skip update
                new_score, new_explanation = seed["score"], seed["explanation"]

            # (d) Pool replacement policy
            if new_response is None:
                # Skip this round if LLM failed
                continue
            new_tokens = _estimate_tokens(new_response)
            total_tokens += new_tokens

            if len(pool) < pool_size:
                pool.append({
                    "question": question,
                    "score": float(new_score),
                    "answer": new_response,
                    "explanation": new_explanation,
                    "origin": self.template_name,
                    "tokens": new_tokens,
                    "history": seed["history"] + [(new_score, new_explanation, new_response, self.template_name)]
                })
                replacements += 1
            else:
                worst_i = _worst_index(pool)
                if new_score > pool[worst_i]["score"] + replacement_margin:
                    pool[worst_i] = {
                        "question": question,
                        "score": float(new_score),
                        "answer": new_response,
                        "explanation": new_explanation,
                        "origin": self.template_name,
                        "tokens": new_tokens,
                        "history": seed["history"] + [(new_score, new_explanation, new_response, self.template_name)]
                    }
                    replacements += 1

            # (e) Metrics per round - optimized
            best_now, pool_mean = _get_pool_metrics(pool)
            metrics_to_record["best_so_far"].append(best_now)
            metrics_to_record["pool_mean"].append(pool_mean)
            metrics_to_record["replacement_ratio"].append(replacements / t)
            metrics_to_record["lift_per_1k_tokens"].append((best_now - best0) / max(1, total_tokens / 1000))
            
            # Early stopping check
            if best_now > last_best_score + min_improvement_threshold:
                last_best_score = best_now
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= convergence_patience:
                logger.info(f"Early stopping at round {t}: no improvement for {convergence_patience} rounds")
                break

        # 4) Return best item + metrics (so caller can save logs / plot)
        best_item = _best_item(pool)
        return best_item, metrics_to_record

    async def batch_explore(
        self,
        question_list: list[str],
        init_response_list: list[str],
        original_score_list: list[float],
        original_explanation_list: list[str],
        budget: int = 5,
        pool_size: int = 2,
        baseline_response_list: Optional[list[str]] = None,
        replacement_margin: float = 0.0,
        # convergence parameters:
        convergence_patience: int = 3,            # stop if no improvement for N rounds
        min_improvement_threshold: float = 0.001, # minimum improvement to count as progress
    ) -> tuple[list[dict], list[dict]]:
        """
        Batched pool-based evolution loop (mirrors `explore`, but per-question in parallel).

        Returns a tuple containing:
        1. A list of `best_item` dicts (one per question), each of the form:
        {
            "question": str,
            "score": float,
            "answer": str,
            "explanation": str,
            "origin": "init" | <template_name>,
            "tokens": int,
            "history": List[(score, explanation, answer, meta)]
        }
        2. A list of metrics dicts (one per question), each of the form:
        {
            "best_so_far": List[float],
            "pool_mean": List[float],
            "replacement_ratio": List[float],
            "lift_per_1k_tokens": List[float]
        }
        """

        # ---- 0) Validate judge requirements
        self.validate_judge_requirements(baseline_response=baseline_response_list)

        n = len(question_list)
        assert len(init_response_list) == n
        assert len(original_score_list) == n
        assert len(original_explanation_list) == n
        if baseline_response_list is not None:
            assert len(baseline_response_list) == n

        # ---- 1) Per-question state
        pools: list[list[dict]] = []
        metrics_list = []
        totals_tokens = [0 for _ in range(n)]
        replacements = [0 for _ in range(n)]
        best0_list = [float(s) for s in original_score_list]
        
        # Early stopping tracking per question
        no_improvement_counts = [0 for _ in range(n)]
        last_best_scores = [float(s) for s in original_score_list]

        for q, init_ans, s0, e0 in zip(question_list, init_response_list, original_score_list, original_explanation_list):
            item = {
                "question": q,
                "score": float(s0),
                "answer": init_ans,
                "explanation": e0,
                "origin": "init",
                "tokens": _estimate_tokens(init_ans),
                "history": [(s0, e0, init_ans, self.template_name)]
            }
            pools.append([item])
            metrics_list.append({
                "best_so_far": [],
                "pool_mean": [],
                "replacement_ratio": [],
                "lift_per_1k_tokens": []
            })

        # ---- 2) Main loop (T rounds) with early stopping
        converged_questions = set()
        
        for t in range(1, budget + 1):
            # Skip if all questions have converged
            if len(converged_questions) == n:
                logger.info(f"All questions converged at round {t-1}")
                break
            # (a) Choose one seed per question (uniform)
            seeds = []
            curr_answers = []
            for pool in pools:
                si = random.randrange(len(pool))
                seed = pool[si]
                seeds.append(seed)
                curr_answers.append(seed["answer"])

            # (b) Generate new responses for the whole batch
            prompts = [self.template.format(base_answer=a) for a in curr_answers]
            try:
                new_responses = await self.llm_agent.batch_invoke(prompts)
            except Exception:
                # If the LLM call fails for the whole batch, fallback to None candidates
                new_responses = [None] * n

            # (c) Judge the new responses (batched)
            try:
                # Judge against the *new* responses (aligns with single `explore`)
                _, new_scores, new_explanations = await self.get_batch_reward(
                    question_list,
                    new_responses,
                    original_score_list,
                    baseline_response_list
                )
            except Exception:
                # On judge failure, keep seed's score/expl
                new_scores = [seeds[i]["score"] for i in range(n)]
                new_explanations = [seeds[i]["explanation"] for i in range(n)]

            # (d) Update pools with replacement policy - optimized
            # Batch estimate tokens for all valid responses
            valid_responses = [resp for resp in new_responses if resp is not None]
            if valid_responses:
                batch_tokens = _batch_estimate_tokens(valid_responses)
                token_idx = 0
            
            for i in range(n):
                pool = pools[i]
                seed = seeds[i]
                new_resp = new_responses[i]
                
                # Skip this question if LLM failed
                if new_resp is None:
                    continue
                
                # Use pre-computed token count
                new_tokens = batch_tokens[token_idx]
                token_idx += 1
                totals_tokens[i] += new_tokens
                
                new_s = float(new_scores[i])
                new_e = new_explanations[i]

                # Check if we need to replace before creating the item
                needs_replacement = len(pool) < pool_size
                if not needs_replacement:
                    wi = _worst_index(pool)
                    needs_replacement = new_s > pool[wi]["score"] + replacement_margin

                if needs_replacement:
                    new_item = {
                        "question": question_list[i],
                        "score": new_s,
                        "answer": new_resp,
                        "explanation": new_e,
                        "origin": self.template_name,
                        "tokens": new_tokens,
                        "history": seed["history"] + [(new_s, new_e, new_resp, self.template_name)]
                    }
                    
                    if len(pool) < pool_size:
                        pool.append(new_item)
                    else:
                        pool[wi] = new_item
                    
                    replacements[i] += 1

            # (e) Per-round metrics (per question) - optimized with cached calculations and convergence check
            for i in range(n):
                if i in converged_questions:
                    continue
                    
                pool = pools[i]
                best_now, pool_mean = _get_pool_metrics(pool)
                metrics = metrics_list[i]
                metrics["best_so_far"].append(best_now)
                metrics["pool_mean"].append(pool_mean)
                metrics["replacement_ratio"].append(replacements[i] / t)
                denom = max(1, totals_tokens[i] / 1000)
                metrics["lift_per_1k_tokens"].append((best_now - last_best_scores[i]) / denom)
                
                # Early stopping check per question
                if best_now > last_best_scores[i] + min_improvement_threshold:
                    last_best_scores[i] = best_now
                    no_improvement_counts[i] = 0
                else:
                    no_improvement_counts[i] += 1
                    
                if no_improvement_counts[i] >= convergence_patience:
                    converged_questions.add(i)
                    logger.info(f"Question {i} converged at round {t}: no improvement for {convergence_patience} rounds")

        # ---- 3) Collect best items + attach metrics
        best_items: list[dict] = []
        for i in range(n):
            best_item = _best_item(pools[i])
            best_items.append(best_item)

        return best_items, metrics_list


async def main(args):
    # ---- 0) Config + Agent
    try:
        judge_type = JudgeType(args.judge_type)  # ensure arg matches enum value
    except Exception:
        raise ValueError(f"Unknown judge_type: {args.judge_type}. Valid: {[e.value for e in JudgeType]}")

    llm_agent = load_model(args.llm_agent_name)
    logger.info("Initializing the agent...")
    agent = SimpleRewriteAgent(
        llm_agent,
        judge_type,
        args.judge_backbone,
        template_name=args.template_name,
        domain=args.domain
    )
    logger.info(f"Agent initialized with config: {agent.get_agent_info()}")
    logger.info("-" * 100)

    # ---- 1) Dataset prep
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
        args.judge_backbone,
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
    logger.info(f"Skipped {len(test_results)} samples (already perfect).")
    logger.info(f"Remaining candidates: {len(selected_idxs)}")

    eval_num, selected_idxs, question_list, init_response_list, original_score_list, \
    original_explanation_list, category_list, baseline_response_list = sample_and_filter_data(
        selected_idxs,
        args.eval_num,
        question_list,
        init_response_list,
        original_score_list,
        original_explanation_list,
        category_list,
        baseline_response_list,
    )

    # ---- 2) Exploration
    start_time = time.time()
    try:
        trajectories, metrics_list = await agent.batch_explore(
            question_list=question_list,
            init_response_list=init_response_list,
            original_score_list=original_score_list,
            original_explanation_list=original_explanation_list,
            budget=args.Budget,
            pool_size=args.pool_size,
            baseline_response_list=baseline_response_list,
        )
    except Exception as e:
        logger.exception("batch_explore failed.")
        raise

    # ---- 3) Evaluate evolved answers
    new_test_results = extract_result_from_trajectories(
        question_list,
        init_response_list,
        category_list,
        original_score_list,
        original_explanation_list,
        baseline_response_list,
        trajectories,
    )
    test_results.extend(new_test_results)

    end_time = time.time()

    # ---- 4) Analysis + Metadata
    analysis = get_result_analysis(test_results)
    meta_info = {
        "strategy": agent.evolve_strategy,
        "template_name": args.template_name,
        "domain": args.domain,
        "judge_type": args.judge_type,
        "dataset_name": args.dataset_name,
        "judge_backbone": get_model_name(args.judge_backbone),
        "answer_position": args.answer_position,
        "baseline_response_model_name": get_model_name(args.baseline_response_model_name),
        "llm_agent_name": get_model_name(args.llm_agent_name),
        "response_model_name": get_model_name(args.response_model_name),
        "budget": args.Budget,
        "pool_size": args.pool_size,
        "eval_num": eval_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_taken": end_time - start_time,
    }
    analysis.update(meta_info)

    trajectories_to_save = meta_info.copy()
    trajectories_to_save.update({"trajectories": trajectories})

    metrics_to_save = meta_info.copy()
    metrics_to_save.update({"metrics": metrics_list})

    # ---- 5) Persist results (ensure dirs)
    os.makedirs(os.path.dirname(args.save_analysis_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_trajectory_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_metrics_path), exist_ok=True)

    save_result_analysis(analysis, args.save_analysis_path)
    save_trajectories(trajectories_to_save, args.save_trajectory_path, f"simple_rewrite_{args.template_name}")
    save_metrics(metrics_to_save, args.save_metrics_path, f"simple_rewrite_{args.template_name}")

    logger.info(
        "Total time taken: %.2f s | budget=%s | pool_size=%d | eval_num=%d",
        end_time - start_time,
        meta_info["budget"],
        args.pool_size,
        eval_num,
    )




if __name__ == "__main__":
    # 1. Setup Logging
    setup_logging(task_name="Direct Prompting Evolve Agent")

    # 2. Setup and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--Budget", type=int, default=2)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--judge_backbone", type=str, default="gemini-2.0-flash")
    parser.add_argument("--judge_type", type=str, default="pointwise", 
                       choices=["pointwise", "pairwise", "pairwise_fine_grained", "alpaca_eval", "arena_hard_auto", "mt_bench", "mlr_bench"])
    parser.add_argument("--answer_position", type=str, default="first", choices=["first", "second"], help="The position of the answer in the pairwise comparison")
    parser.add_argument("--response_model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--baseline_response_model_name", type=str, default=None)
    parser.add_argument("--llm_agent_name", type=str, default="gpt-4.1-nano")
    parser.add_argument("--dataset_name", type=str, default="ArenaHard")
    parser.add_argument("--template_name", type=str, default="holistic")
    parser.add_argument("--domain", type=str, default="general")
    parser.add_argument("--data_dir", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data")
    parser.add_argument("--eval_num", type=int, default=2)
    parser.add_argument("--save_analysis_path", type=str, default="results/")
    parser.add_argument("--save_trajectory_path", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/")
    parser.add_argument("--save_metrics_path", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/metrics/")

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    # 3. Run the main application logic, passing in the configuration.
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    finally:
        logger.info("Application finished.")