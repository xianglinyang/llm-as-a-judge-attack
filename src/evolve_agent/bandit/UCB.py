import os
import numpy as np
import logging
import argparse
import time
import asyncio
import random
from typing import List, Optional, Tuple, Dict, Any

from src.evolve_agent.bandit.base import ContextualLinBanditAgent
from src.llm_evaluator import JudgeType
from src.text_encoder import TextEncoder, MiniLMTextEncoder
from src.llm_zoo import BaseLLM, load_model
from src.logging_utils import setup_logging
from src.evolve_agent.utils import (prepare_dataset_for_exploration, 
                                    exclude_perfect_response, 
                                    sample_and_filter_data, 
                                    extract_result_from_trajectories, 
                                    get_result_analysis, 
                                    save_result_analysis, 
                                    save_trajectories, 
                                    save_metrics)
from src.llm_zoo.api_zoo import get_model_name
from src.evolve_agent.utils import _worst_index, _best_item, _estimate_tokens, _select_from_pool_uniform
from src.evolve_agent.bias_strategies import Bias_types
    

logger = logging.getLogger(__name__)

class ContextualLinUCBAgent(ContextualLinBanditAgent):
    def __init__(self, n_features: int, llm_agent: BaseLLM, embedding_model: TextEncoder, judge_type: JudgeType, judge_model_backbone: str, reward_type: str = "relative", alpha: float = 1.0, lambda_reg: float = 1.0, answer_position: str = "first"):
        """
        Initializes the LinUCB agent.

        Args:
            n_features (int): Dimension of context features (d).
            llm_agent (BaseLLM): LLM agent to generate the response.
            embedding_model (TextEncoder): Embedding model to encode the context.
            judge_type (JudgeType): Type of judge evaluation (pointwise, pairwise, etc.)
            judge_model_backbone (str): Backbone model for the judge
            reward_type (str): Type of reward to use ("relative" or "absolute").
            alpha (float): Exploration parameter. Controls the width of the confidence interval.
                           Higher alpha means more exploration.
            lambda_reg (float): Regularization parameter for Ridge Regression.
                                This is the 'lambda' in (X^T X + lambda*I)^-1 X^T y.
                                Corresponds to initializing A_a with lambda_reg * I.
            answer_position (str): Position of the answer in pairwise comparison ("first" or "second").
        """
        super().__init__(n_features, llm_agent, embedding_model, judge_type, judge_model_backbone, reward_type, lambda_reg, answer_position)
        self.alpha = alpha
    
    def _linucb_arm_stats(self, q_idx: int, x_col: np.ndarray):
        """
        Returns per-arm: theta_hat^T x, s = sqrt(x^T A^{-1} x), ucb = mean + alpha*s.
        Uses cached As_inv and thetas for performance.
        Shapes assumed:
        - self.As_inv[q_idx][a]: (d,d) - cached inverse
        - self.thetas[q_idx][a]: (d,1) - cached theta
        - x_col: (d,1)
        """
        means, ss, ucbs = [], [], []
        for a in range(self.n_arms):
            # Use cached structures for performance
            A_inv = self.As_inv[q_idx][a]
            theta_hat = self.thetas[q_idx][a]
            
            mean = float(theta_hat.T @ x_col)                # scalar
            # s = sqrt(x^T A^{-1} x)
            s2 = float(x_col.T @ A_inv @ x_col)
            s = float(np.sqrt(max(1e-18, s2)))  # avoid numerical issues
            ucb = mean + self.alpha * s
            means.append(mean); ss.append(s); ucbs.append(ucb)
        return means, ss, ucbs

    def _compute_gap_metrics(self, context_x, model_idx):
        """
        Compute CI width and UCB gap for the current context.
        
        Args:
            context_x (np.array): Context vector (n_features, 1)
            model_idx (int): Model index
        
        Returns:
            tuple: (ci_width, ucb_gap)
        """
        means, ss, ucbs = self._linucb_arm_stats(model_idx, context_x)
        
        # Find chosen arm (highest UCB)
        chosen_arm = int(np.argmax(ucbs))
        ci_width = 2.0 * self.alpha * ss[chosen_arm]
        
        # Compute UCB gap (best - second best)
        if len(ucbs) >= 2:
            sorted_ucbs = sorted(ucbs, reverse=True)
            ucb_gap = float(sorted_ucbs[0] - sorted_ucbs[1])
        else:
            ucb_gap = 0.0
            
        return ci_width, ucb_gap

    def predict(self, context_x, model_idx):
        """
        Predicts the UCB score for each arm given the context.

        Args:
            context_x (np.array): A (n_features x 1) column vector representing the context.

        Returns:
            np.array: A (n_arms x 1) vector of UCB scores for each arm.
        """
        if context_x.shape != (self.n_features, 1):
            raise ValueError(f"Context_x must be a column vector of shape ({self.n_features}, 1)")

        # Vectorized computation across all arms
        p_ta_values = np.zeros((self.n_arms, 1))
        x_flat = context_x.flatten()  # (d,) for efficient computation
        
        for arm_idx in range(self.n_arms):
            # Use cached structures for performance
            A_inv = self.As_inv[model_idx][arm_idx]
            theta = self.thetas[model_idx][arm_idx].flatten()  # (d,)
            
            # Expected reward: x^T @ theta (vectorized)
            expected_reward = np.dot(x_flat, theta)
            
            # Uncertainty: x^T @ A_inv @ x (optimized)
            # Compute A_inv @ x first, then x^T @ result
            Ainv_x = A_inv @ context_x  # (d, d) @ (d, 1) = (d, 1)
            uncertainty_term = np.dot(x_flat, Ainv_x.flatten())
            exploration_bonus = self.alpha * np.sqrt(max(uncertainty_term, 1e-18))
            
            p_ta_values[arm_idx, 0] = expected_reward + exploration_bonus

        return p_ta_values
    
    def batch_predict(self, context_x_list, model_idxs):
        """
        Predicts the UCB score for each arm given the context.
        Args:
            context_x_list (np.array): A (n_samples, n_features, 1) tensor representing the context.
            model_idxs (List[int]): List of model indices for each sample.
        Returns:
            np.array: A (n_samples, n_arms, 1) tensor of UCB scores for each arm.
        """
        if context_x_list.shape[1:] != (self.n_features, 1):
            raise ValueError(f"Context_x must be a tensor of shape (n_samples, {self.n_features}, 1)")
        
        n_samples = context_x_list.shape[0]
        p_ta_values = np.zeros((n_samples, self.n_arms, 1))

        # Reshape contexts for vectorized operations: (n_samples, n_features)
        X = context_x_list.squeeze(-1)  # (n_samples, n_features)
        
        # Group samples by model_idx for efficient batch processing
        unique_models = list(set(model_idxs))
        
        for model_idx in unique_models:
            # Find all samples using this model
            sample_mask = np.array(model_idxs) == model_idx
            sample_indices = np.where(sample_mask)[0]
            
            if len(sample_indices) == 0:
                continue
                
            X_model = X[sample_indices]  # (n_model_samples, n_features)
            
            # Vectorized computation for all arms at once
            for arm_idx in range(self.n_arms):
                # Use cached structures
                A_inv = self.As_inv[model_idx][arm_idx]  # (d, d)
                theta = self.thetas[model_idx][arm_idx].flatten()  # (d,)
                
                # Expected rewards: X @ theta (vectorized)
                expected_rewards = X_model @ theta  # (n_model_samples,)
                
                # Uncertainty: sqrt(X @ A_inv @ X.T) (vectorized)
                # Compute X @ A_inv first: (n_model_samples, d) @ (d, d) = (n_model_samples, d)
                X_Ainv = X_model @ A_inv
                # Then element-wise: sum(X_Ainv * X_model, axis=1) = (n_model_samples,)
                uncertainty_terms = np.sum(X_Ainv * X_model, axis=1)
                exploration_bonus = self.alpha * np.sqrt(np.maximum(uncertainty_terms, 1e-18))
                
                # Combine and store
                ucb_scores = expected_rewards + exploration_bonus
                p_ta_values[sample_indices, arm_idx, 0] = ucb_scores

        return p_ta_values
    
     
    def choose_arm(self, context_x, model_idx):
        """
        Chooses an arm based on the highest UCB score.

        Args:
            context_x (np.array): A (n_features x 1) column vector representing the context.

        Returns:
            int: The index of the chosen arm.
        """
        ucb_scores = self.predict(context_x, model_idx)
        chosen_arm_idx = np.argmax(ucb_scores)
        return chosen_arm_idx
    
    def batch_choose_arm(self, context_x_list, model_idxs):
        """
        Chooses an arm based on the highest UCB score.
        Args:
            context_x_list (np.array): A (n_samples, n_features, 1) tensor representing the context.
        Returns:
            np.array: A (n_samples,) tensor of the chosen arm index for each sample.
        """
        ucb_scores = self.batch_predict(context_x_list, model_idxs)
        chosen_arm_idx = np.argmax(ucb_scores, axis=1)
        return chosen_arm_idx
    
    def explore(
        self,
        question: str,
        init_response: str,
        original_score: float,
        original_explanation: str,
        pool_size: int,
        Budget: int,
        cold_start: bool,
        baseline_response: str = None,
        replacement_margin: float = 0.0,
    ):
        """
        Single-question exploration using LinUCB with pool replacement.

        Returns:
        best_item: {
            "question": str,
            "score": float,
            "answer": str,
            "explanation": str,
            "origin": "init" | <bias_strategy_name>,
            "tokens": int,
            "history": List[(score, explanation, answer, meta_dict)]
        }
        metrics: {
            "best_so_far": List[float],
            "pool_mean": List[float],
            "replacement_ratio": List[float],
            "lift_per_1k_tokens": List[float],
        }
        """
        # ---- 0) Setup
        self.init_policy_model(1)  # model_idx==0
        self.validate_judge_requirements(baseline_response=baseline_response)

        if Budget < self.n_arms:
            raise ValueError(f"Budget must be greater than the number of arms: {self.n_arms}")

        # ---- 1) Init pool + metrics
        init_item = {
            "question": question,
            "score": float(original_score),
            "answer": init_response,
            "explanation": original_explanation,
            "origin": "init",
            "tokens": _estimate_tokens(init_response),
            "history": [(float(original_score), original_explanation, init_response, "init")],
        }
        pool = [init_item]

        metrics = {
            "best_so_far": [],
            "pool_mean": [],
            "replacement_ratio": [],
            "lift_per_1k_tokens": [],
            "ci_width": [],
            "ucb_gap": [],
        }
        replacements = 0
        totals_tokens = 0
        best0 = float(original_score)

        # ---- 2) Optional cold start (probe each arm once from the init response)
        curr_step = 0
        if cold_start:
            x = self.get_context_x(question, init_response)
            for arm_idx in range(self.n_arms):
                strategy = Bias_types[arm_idx]
                new_response = self.bias_modifier.principle_guided_mutation(init_response, strategy)

                reward, new_score, new_expl = self.get_reward(
                    question, new_response, original_score, baseline_response
                )
                self.update(arm_idx, x, reward, 0)

                new_item = {
                    "question": question,
                    "score": float(new_score),
                    "answer": new_response,
                    "explanation": new_expl,
                    "origin": strategy,
                    "tokens": _estimate_tokens(new_response),
                    "history": init_item["history"] + [(float(new_score), new_expl, new_response, strategy)],
                }
                totals_tokens += new_item["tokens"]

                if len(pool) < pool_size:
                    pool.append(new_item)
                    replacements += 1
                else:
                    wi = _worst_index(pool)
                    if new_item["score"] > pool[wi]["score"] + replacement_margin:
                        pool[wi] = new_item
                        replacements += 1

                # metrics after this probe
                best_now = _best_item(pool)["score"]
                pool_mean = sum(xi["score"] for xi in pool) / len(pool)
                metrics["best_so_far"].append(best_now)
                metrics["pool_mean"].append(pool_mean)
                metrics["replacement_ratio"].append(replacements / max(1, (curr_step + 1)))
                denom = max(1, totals_tokens / 1000)
                metrics["lift_per_1k_tokens"].append((best_now - best0) / denom)
                
                # UCB-specific metrics
                ci_width, ucb_gap = self._compute_gap_metrics(x, 0)
                metrics["ci_width"].append(ci_width)
                metrics["ucb_gap"].append(ucb_gap)

                logger.info(f"[ColdStart arm {arm_idx}] "
                            f"orig={original_score:.3f} → new={new_score:.3f} "
                            f"arm={strategy}")

                curr_step += 1

        # ---- 3) Main loop
        for t in range(curr_step, Budget):
            # pick a seed uniformly
            si = random.randrange(len(pool))
            seed = pool[si]
            curr_answer = seed["answer"]
            curr_score = seed["score"]
            curr_expl = seed["explanation"]

            # context from current answer
            x = self.get_context_x(question, curr_answer)

            # choose arm
            chosen_arm = int(self.choose_arm(x, 0))
            strategy = Bias_types[chosen_arm]

            # mutate
            new_response = self.bias_modifier.principle_guided_mutation(curr_answer, strategy)

            # reward
            reward, new_score, new_expl = self.get_reward(
                question, new_response, curr_score, baseline_response
            )

            # policy update
            self.update(chosen_arm, x, reward, 0)

            # build new item
            new_item = {
                "question": question,
                "score": float(new_score),
                "answer": new_response,
                "explanation": new_expl,
                "origin": strategy,
                "tokens": _estimate_tokens(new_response),
                "history": seed["history"] + [(float(new_score), new_expl, new_response, strategy)],
            }
            totals_tokens += new_item["tokens"]

            # pool replacement
            if len(pool) < pool_size:
                pool.append(new_item)
                replacements += 1
            else:
                wi = _worst_index(pool)
                if new_item["score"] > pool[wi]["score"] + replacement_margin:
                    pool[wi] = new_item
                    replacements += 1

            # metrics
            rounds = (t + 1)
            best_now = _best_item(pool)["score"]
            pool_mean = sum(xi["score"] for xi in pool) / len(pool)
            metrics["best_so_far"].append(best_now)
            metrics["pool_mean"].append(pool_mean)
            metrics["replacement_ratio"].append(replacements / max(1, rounds))
            denom = max(1, totals_tokens / 1000)
            metrics["lift_per_1k_tokens"].append((best_now - best0) / denom)

            # UCB-specific metrics (use current context)
            ci_width, ucb_gap = self._compute_gap_metrics(x, 0)
            metrics["ci_width"].append(ci_width)
            metrics["ucb_gap"].append(ucb_gap)

            logger.info(f"[Iter {t}] seed={curr_score:.3f} → new={new_score:.3f} "
                        f"arm={strategy}")

        # ---- 4) Return best item + metrics
        best_item = _best_item(pool)
        return best_item, metrics
    
    async def batch_explore(
        self,
        question_list: List[str],
        init_response_list: List[str],
        original_score_list: List[float],
        original_explanation_list: List[str],
        budget: int = 5,
        pool_size: int = 2,
        baseline_response_list: Optional[List[str]] = None,
        replacement_margin: float = 0.0,
        cold_start: bool = True,
        init_model_path: Optional[str] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Batched pool-based exploration using LinUCB (API-compatible with `simple_rewrite.batch_explore`).

        Returns:
        best_items: list of dict (one per question)
            {
            "question": str,
            "score": float,
            "answer": str,
            "explanation": str,
            "origin": "init" | <bias_strategy_name>,
            "tokens": int,
            "history": List[(score, explanation, answer, meta_dict)]
            }
        metrics_list: list of dict (one per question)
            {
            "best_so_far": List[float],
            "pool_mean": List[float],
            "replacement_ratio": List[float],
            "lift_per_1k_tokens": List[float],
            }
        """
        # ---- 0) Validate judge requirements
        # Keep the same kw as your other explorer for consistency
        self.validate_judge_requirements(baseline_response=baseline_response_list)

        n = len(question_list)
        assert len(init_response_list) == n
        assert len(original_score_list) == n
        assert len(original_explanation_list) == n
        if baseline_response_list is not None:
            assert len(baseline_response_list) == n
        
        # ---- 1) Policy model init / load
        if init_model_path is not None:
            self.load_policy_model(init_model_path)
            if len(self.As) != n:
                if len(self.As) == 1:
                    # replicate single model across questions (including cached structures)
                    # Use list comprehensions for better performance
                    self.As = [[A.copy() for A in self.As[0]] for _ in range(n)]
                    self.bs = [[b.copy() for b in self.bs[0]] for _ in range(n)]
                    self.As_inv = [[A_inv.copy() for A_inv in self.As_inv[0]] for _ in range(n)]
                    self.thetas = [[theta.copy() for theta in self.thetas[0]] for _ in range(n)]
                    # Also replicate update counters if they exist
                    if hasattr(self, '_updates_count'):
                        self._updates_count = [self._updates_count[0].copy() for _ in range(n)]
                else:
                    raise ValueError(
                        f"Loaded {len(self.As)} models but have {n} questions."
                    )
        else:
            self.init_policy_model(n)

        if budget < self.n_arms:
            # Keep behavior consistent with your original constraint
            raise ValueError(f"Budget must be >= number of arms: {self.n_arms}")
        
        # ---- 2) Per-question state (pool + metrics)
        pools: List[List[Dict]] = []
        metrics_list: List[Dict[str, List[float]]] = []
        totals_tokens = [0 for _ in range(n)]
        replacements = [0 for _ in range(n)]
        best0_list = [float(s) for s in original_score_list]

        for q, init_ans, s0, e0 in zip(question_list, init_response_list, original_score_list, original_explanation_list):
            item = {
                "question": q,
                "score": float(s0),
                "answer": init_ans,
                "explanation": e0,
                "origin": "init",
                "tokens": _estimate_tokens(init_ans),
                "history": [(float(s0), e0, init_ans, "init")],
            }
            pools.append([item])
            metrics_list.append({
                "best_so_far": [],
                "pool_mean": [],
                "replacement_ratio": [],
                "lift_per_1k_tokens": [],
                "ci_width": [],
                "ucb_gap": [],
            })
        
        # ---- 3) Optional cold-start: probe each arm once per question from the init answer
        rounds_consumed = 0
        if cold_start:
            # Build contexts from (question, init_answer)
            context_x = self.get_context_x_batch(question_list, init_response_list)

            for arm_idx in range(self.n_arms):
                # propose one mutation per question using this arm's bias strategy
                strategy_list = [Bias_types[arm_idx]] * n  # or global Bias_types
                try:
                    new_responses = await self.bias_modifier.batch_principle_guided_mutation(
                        init_response_list, strategy_list
                    )
                except Exception:
                    new_responses = [None] * n

                # judge
                try:
                    reward_list, new_scores, new_expls = await self.get_batch_reward(
                        question_list, new_responses, original_score_list, baseline_response_list
                    )
                except Exception:
                    reward_list = [0.0] * n
                    new_scores = [pools[i][0]["score"] for i in range(n)]
                    new_expls = [pools[i][0]["explanation"] for i in range(n)]

                # update policy + pools
                for i in range(n):
                    # policy update only if we actually produced a response
                    self.update(arm_idx, context_x[i], float(reward_list[i]), i)

                    if new_responses[i] is None:
                        continue

                    new_ans = new_responses[i]
                    new_s = float(new_scores[i])
                    new_e = new_expls[i]
                    new_tok = _estimate_tokens(new_ans)
                    totals_tokens[i] += new_tok

                    seed = pools[i][0]  # from init
                    new_item = {
                        "question": question_list[i],
                        "score": new_s,
                        "answer": new_ans,
                        "explanation": new_e,
                        "origin": strategy_list[i],
                        "tokens": new_tok,
                        "history": seed["history"] + [(new_s, new_e, new_ans, strategy_list[i])],
                    }

                    pool = pools[i]
                    if len(pool) < pool_size:
                        pool.append(new_item)
                        replacements[i] += 1
                    else:
                        wi = _worst_index(pool)
                        if new_s > pool[wi]["score"] + replacement_margin:
                            pool[wi] = new_item
                            replacements[i] += 1

                rounds_consumed += 1
                # per-round metrics after each arm probe
                for i in range(n):
                    pool = pools[i]
                    best_now = _best_item(pool)["score"]
                    pool_mean = sum(x["score"] for x in pool) / len(pool)
                    m = metrics_list[i]
                    m["best_so_far"].append(best_now)
                    m["pool_mean"].append(pool_mean)
                    m["replacement_ratio"].append(replacements[i] / max(1, rounds_consumed))
                    denom = max(1, totals_tokens[i] / 1000)
                    m["lift_per_1k_tokens"].append((best_now - best0_list[i]) / denom)

                    # UCB-specific metrics (use context from current iteration)
                    x_col = context_x[i]  # Use current context instead of init context
                    ci_width_i, ucb_gap_i = self._compute_gap_metrics(x_col, i)
                    m["ci_width"].append(ci_width_i)
                    m["ucb_gap"].append(ucb_gap_i)

        # Remaining rounds for the main LinUCB loop
        remaining_rounds = max(0, budget - rounds_consumed)

        # ---- 4) Main loop
        for t in range(1, remaining_rounds + 1):
            # (a) pick one seed per question (uniform over each pool)
            seeds = []
            curr_answers = []
            for pool in pools:
                si = _select_from_pool_uniform(pool)
                seed = pool[si]
                seeds.append(seed)
                curr_answers.append(seed["answer"])

            # (b) build contexts on the current answers
            context_x = self.get_context_x_batch(question_list, curr_answers)

            # (c) choose arms with LinUCB (shape: (n, 1) or list of [arm])
            chosen_arm_list = self.batch_choose_arm(context_x, list(range(len(context_x))))

            # (d) mutate answers according to chosen arms
            strategy_list = [
                Bias_types[int(a[0])] if isinstance(a, (list, np.ndarray)) else Bias_types[int(a)]
                for a in chosen_arm_list
            ]
            try:
                new_responses = await self.bias_modifier.batch_principle_guided_mutation(
                    curr_answers, strategy_list
                )
            except Exception:
                new_responses = [None] * n

            # (e) judge
            try:
                reward_list, new_scores, new_expls = await self.get_batch_reward(
                    question_list, new_responses, original_score_list, baseline_response_list
                )
            except Exception:
                reward_list = [0.0] * n
                new_scores = [seeds[i]["score"] for i in range(n)]
                new_expls = [seeds[i]["explanation"] for i in range(n)]

            # (f) policy update
            for i in range(n):
                arm_idx = int(chosen_arm_list[i][0]) if isinstance(chosen_arm_list[i], (list, np.ndarray)) else int(chosen_arm_list[i])
                self.update(arm_idx, context_x[i], float(reward_list[i]), i)

            # (g) pool replacement
            for i in range(n):
                if new_responses[i] is None:
                    continue

                pool = pools[i]
                seed = seeds[i]
                new_ans = new_responses[i]
                new_s = float(new_scores[i])
                new_e = new_expls[i]
                new_tok = _estimate_tokens(new_ans)
                totals_tokens[i] += new_tok

                arm_idx = int(chosen_arm_list[i][0]) if isinstance(chosen_arm_list[i], (list, np.ndarray)) else int(chosen_arm_list[i])
                new_item = {
                    "question": question_list[i],
                    "score": new_s,
                    "answer": new_ans,
                    "explanation": new_e,
                    "origin": strategy_list[i],
                    "tokens": new_tok,
                    "history": seed["history"] + [(new_s, new_e, new_ans, strategy_list[i])],
                }

                if len(pool) < pool_size:
                    pool.append(new_item)
                    replacements[i] += 1
                else:
                    wi = _worst_index(pool)
                    if new_s > pool[wi]["score"] + replacement_margin:
                        pool[wi] = new_item
                        replacements[i] += 1

            # (h) metrics for this round
            total_round = rounds_consumed + t
            for i in range(n):
                pool = pools[i]
                best_now = _best_item(pool)["score"]
                pool_mean = sum(x["score"] for x in pool) / len(pool)
                m = metrics_list[i]
                m["best_so_far"].append(best_now)
                m["pool_mean"].append(pool_mean)
                m["replacement_ratio"].append(replacements[i] / max(1, total_round))
                denom = max(1, totals_tokens[i] / 1000)
                m["lift_per_1k_tokens"].append((best_now - best0_list[i]) / denom)

                # UCB-specific metrics for this sample
                x_col = context_x[i]                      # shape (d,1)
                ci_width_i, ucb_gap_i = self._compute_gap_metrics(x_col, i)
                
                m = metrics_list[i]
                m["ci_width"].append(ci_width_i)
                m["ucb_gap"].append(ucb_gap_i)

        # ---- 5) Collect best items
        best_items: List[Dict] = []
        for i in range(n):
            best_item = _best_item(pools[i])
            best_items.append(best_item)

        return best_items, metrics_list

    async def warmup(
        self,
        question_list: List[str],
        init_response_list: List[str],
        original_score_list: List[float],
        original_explanation_list: List[str],
        baseline_response_list: Optional[List[str]] = None,
        # schedule
        burnin_passes: int = 1,                 # each pass: probe every arm once on all questions
        ucb_passes: int = 2,                    # each pass: choose arm per question via LinUCB(+ε) and update
        epsilon_schedule: Optional[List[float]] = None,  # e.g. [0.15, 0.10]
        # early stop on uncertainty collapse
        enable_ci_early_stop: bool = True,
        ci_width_threshold: float = 0.12,
        patience: int = 2,
        # persistence
        save_path: Optional[str] = None,        # if set, saves learned global model here
    ) -> Dict[str, Any]:
        """
        Warm up ONE global LinUCB (model_idx=0) using a corpus.
        Phase 1: burn-in (coverage, round-robin arms).
        Phase 2: LinUCB+ε (focused exploration).
        Returns a summary with median CI width / index gap trajectories. The saved model
        can be reused later with batch_explore(..., init_model_path=save_path, cold_start=False).
        """
        n = len(question_list)
        assert len(init_response_list) == n
        assert len(original_score_list) == n
        assert len(original_explanation_list) == n
        if baseline_response_list is not None:
            assert len(baseline_response_list) == n

        # ---- init one global model
        self.init_policy_model(1)   # ONE shared LinUCB across the corpus
        self.validate_judge_requirements(baseline_response=baseline_response_list)
        if epsilon_schedule is None:
            epsilon_schedule = [0.15] * max(1, ucb_passes)

        # maintain current answers (start from init)
        curr_answers = list(init_response_list)

        median_ci, median_gap = [], []
        ok_streak = 0
        rounds = 0

        def _ctx_batch(qs: List[str], ans: List[str]) -> np.ndarray:
            return self.get_context_x_batch(qs, ans)

        def _arm_stats_for_x(x_col: np.ndarray):
            """Return per-arm (mean, s, ucb) using the ONE global model."""
            alpha = getattr(self, "alpha", 1.2)
            means, ss, ucbs = [], [], []
            for a in range(self.n_arms):
                Ainv = self.As_inv[0][a]
                theta = self.thetas[0][a]
                mean = float(theta.T @ x_col)
                s2 = float(x_col.T @ (Ainv @ x_col))
                s = float(np.sqrt(max(1e-18, s2)))
                means.append(mean); ss.append(s); ucbs.append(mean + alpha * s)
            return means, ss, ucbs

        def _batch_arm_stats(X: np.ndarray):
            """Vectorized arm stats for entire batch - OPTIMIZATION."""
            alpha = getattr(self, "alpha", 1.2)
            n_samples = X.shape[0]
            
            batch_means = np.zeros((n_samples, self.n_arms))
            batch_ss = np.zeros((n_samples, self.n_arms))
            batch_ucbs = np.zeros((n_samples, self.n_arms))
            
            for a in range(self.n_arms):
                Ainv = self.As_inv[0][a]  # (d, d)
                theta = self.thetas[0][a]  # (d, 1)
                
                # Vectorized mean: X @ theta for all samples
                X_flat = X.squeeze(-1)  # (n_samples, d)
                theta_flat = theta.flatten()  # (d,)
                batch_means[:, a] = X_flat @ theta_flat
                
                # Vectorized uncertainty: sqrt(X @ Ainv @ X.T) for all samples
                X_Ainv = X_flat @ Ainv  # (n_samples, d)
                s2_vals = np.sum(X_Ainv * X_flat, axis=1)  # (n_samples,)
                batch_ss[:, a] = np.sqrt(np.maximum(s2_vals, 1e-18))
                
                # UCB scores
                batch_ucbs[:, a] = batch_means[:, a] + alpha * batch_ss[:, a]
            
            return batch_means, batch_ss, batch_ucbs

        def _ucb_choose_arm(x_col: np.ndarray, eps: float) -> int:
            if np.random.rand() < eps:
                return np.random.randint(self.n_arms)
            _, _, ucbs = _arm_stats_for_x(x_col)
            return int(np.argmax(ucbs))

        def _round_medians(qs: List[str], ans: List[str]) -> Dict[str, float]:
            """Optimized metrics computation using vectorized operations."""
            alpha = getattr(self, "alpha", 1.2)
            X = _ctx_batch(qs, ans)
            
            # Use vectorized arm stats computation
            batch_means, batch_ss, batch_ucbs = _batch_arm_stats(X)
            
            # Compute CI width and gaps vectorized
            best_arms = np.argmax(batch_ucbs, axis=1)  # (n_samples,)
            ci_vals = 2.0 * alpha * batch_ss[np.arange(len(qs)), best_arms]
            
            # Compute gaps efficiently
            gaps = []
            for i in range(len(qs)):
                ucbs = batch_ucbs[i]
                if len(ucbs) >= 2:
                    sorted_ucbs = np.sort(ucbs)[::-1]  # Sort descending
                    gaps.append(float(sorted_ucbs[0] - sorted_ucbs[1]))
                else:
                    gaps.append(0.0)
            
            return {"med_ci": float(np.median(ci_vals)), "med_gap": float(np.median(gaps))}

        def _maybe_stop() -> bool:
            nonlocal ok_streak
            if not enable_ci_early_stop or not median_ci:
                return False
            ok = median_ci[-1] < ci_width_threshold
            ok_streak = ok_streak + 1 if ok else 0
            return ok_streak >= patience

        # ======================
        # Phase 1: Burn-in passes (OPTIMIZED)
        # ======================
        for _ in range(max(0, burnin_passes)):
            for arm_idx in range(self.n_arms):
                # mutate ALL questions with this arm
                strategy = (self.bias_modifier.Bias_types[arm_idx]
                            if hasattr(self.bias_modifier, "Bias_types") else Bias_types[arm_idx])
                try:
                    new_responses = await self.bias_modifier.batch_principle_guided_mutation(
                        curr_answers, [strategy] * n
                    )
                except Exception:
                    continue

                # judge
                try:
                    rewards, _scores, _expls = await self.get_batch_reward(
                        question_list, new_responses, original_score_list, baseline_response_list
                    )
                except Exception:
                    continue

                # OPTIMIZATION: Compute contexts only once per arm
                X_new = _ctx_batch(question_list, new_responses)
                for i in range(n):
                    self.update(arm_idx, X_new[i], float(rewards[i]), model_idx=0)

                # advance current answers to diversify contexts
                curr_answers = new_responses

                # OPTIMIZATION: Compute metrics only if early stopping enabled or every N rounds
                rounds += 1
                if enable_ci_early_stop or (rounds % max(1, self.n_arms // 2) == 0):
                    m = _round_medians(question_list, curr_answers)
                    median_ci.append(m["med_ci"])
                    median_gap.append(m["med_gap"])
                    if _maybe_stop():
                        if save_path:
                            os.makedirs(save_path, exist_ok=True)
                            self.save_policy_model(save_path)
                        return {"phase": "burnin", "rounds": rounds, "median_ci": median_ci, "median_gap": median_gap,
                                "saved_to": save_path}
                else:
                    # Fill with previous values to maintain tracking
                    if median_ci:
                        median_ci.append(median_ci[-1])
                        median_gap.append(median_gap[-1])

        # ==========================
        # Phase 2: UCB+ε warm-up passes (OPTIMIZED)
        # ==========================
        for u in range(max(0, ucb_passes)):
            eps = float(epsilon_schedule[min(u, len(epsilon_schedule) - 1)])

            # OPTIMIZATION: Compute contexts and arm selection in batch
            X = _ctx_batch(question_list, curr_answers)
            
            # Vectorized UCB+ε arm selection
            if eps > 0.0:
                # Mixed strategy: some random, some UCB
                random_mask = np.random.rand(n) < eps
                chosen_arms = np.zeros(n, dtype=int)
                
                # Random arms for exploration
                chosen_arms[random_mask] = np.random.randint(0, self.n_arms, size=np.sum(random_mask))
                
                # UCB arms for exploitation (vectorized)
                if not np.all(random_mask):
                    X_ucb = X[~random_mask]
                    if len(X_ucb) > 0:
                        _, _, batch_ucbs = _batch_arm_stats(X_ucb)
                        chosen_arms[~random_mask] = np.argmax(batch_ucbs, axis=1)
            else:
                # Pure UCB selection (vectorized)
                _, _, batch_ucbs = _batch_arm_stats(X)
                chosen_arms = np.argmax(batch_ucbs, axis=1)
            
            strategies = [
                (self.bias_modifier.Bias_types[a]
                if hasattr(self.bias_modifier, "Bias_types") else Bias_types[a])
                for a in chosen_arms
            ]

            # mutate & judge
            try:
                new_responses = await self.bias_modifier.batch_principle_guided_mutation(curr_answers, strategies)
            except Exception:
                continue
            try:
                rewards, _scores, _expls = await self.get_batch_reward(
                    question_list, new_responses, original_score_list, baseline_response_list
                )
            except Exception:
                continue

            # OPTIMIZATION: Batch context computation and updates
            X_new = _ctx_batch(question_list, new_responses)
            for i in range(n):
                self.update(int(chosen_arms[i]), X_new[i], float(rewards[i]), model_idx=0)

            # advance current answers
            curr_answers = new_responses

            # metrics + early stop
            rounds += 1
            m = _round_medians(question_list, curr_answers)
            median_ci.append(m["med_ci"])
            median_gap.append(m["med_gap"])
            if _maybe_stop():
                break

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.save_policy_model(save_path)

        return {"phase": "complete", "rounds": rounds, "median_ci": median_ci, "median_gap": median_gap,
                "saved_to": save_path}


async def main(args):
    llm_agent = load_model(args.llm_agent_name)
    embedding_model = MiniLMTextEncoder()
    judge_type = JudgeType(args.judge_type)
    judge_model_backbone = args.judge_model_name

    # Use the enhanced reward system instead of manual pairwise scoring
    question_list, init_response_list, category_list, original_score_list, original_explanation_list, baseline_response_list = await prepare_dataset_for_exploration(args.data_dir, args.dataset_name, args.response_model_name, judge_type, judge_model_backbone, args.baseline_response_model_name, args.answer_position)
    test_results, selected_idxs = exclude_perfect_response(judge_type, question_list, init_response_list, category_list, original_score_list, original_explanation_list, baseline_response_list)
    logger.info(f"Skipped {len(test_results)} samples")
    logger.info(f"Dataset for exploration: {len(selected_idxs)} samples...")
    eval_num, selected_idxs, question_list, init_response_list, original_score_list, original_explanation_list, category_list, baseline_response_list = sample_and_filter_data(selected_idxs, args.eval_num, question_list, init_response_list, original_score_list, original_explanation_list, category_list, baseline_response_list)

    logger.info(f"Initializing the agent...")
    agent = ContextualLinUCBAgent(args.n_features, llm_agent, embedding_model, judge_type, judge_model_backbone, args.reward_type, args.alpha, args.lambda_reg, args.answer_position)
    logger.info(f"Agent initialized.")
    logger.info("-"*100)


    start_time = time.time()

    trajectories = []
    if args.test_mode == "ucb":
        logger.info(f"Running single exploration...")
        trajectories, metrics_list = await agent.batch_explore(question_list, init_response_list, original_score_list, original_explanation_list, args.Budget, args.pool_size, cold_start=args.cold_start, init_model_path=args.init_model_path, baseline_response_list=baseline_response_list)
        logger.info(f"UCB exploration finished.")
        logger.info("-"*100)
    elif args.test_mode == "random":
        logger.info(f"Running random exploration...")
        trajectories, metrics_list = await agent.batch_explore_with_random_arm(question_list, init_response_list, original_score_list, original_explanation_list, args.Budget, args.pool_size, baseline_response_list=baseline_response_list)
        logger.info(f"Random exploration finished.")
        logger.info("-"*100)
    else:
        raise ValueError(f"Invalid test mode: {args.test_mode}")
    
    new_test_results = extract_result_from_trajectories(question_list, init_response_list, category_list, original_score_list, original_explanation_list, baseline_response_list, trajectories)
    test_results.extend(new_test_results)
    
    end_time = time.time()
        
    analysis = get_result_analysis(test_results)
    meta_info = {
        "strategy": args.test_mode,
        "judge_type": args.judge_type,
        "answer_position": args.answer_position,
        "dataset_name": args.dataset_name,
        "judge_backbone": get_model_name(args.judge_model_name),
        "baseline_response_model_name": get_model_name(args.baseline_response_model_name),
        "llm_agent_name": get_model_name(args.llm_agent_name),
        "response_model_name": get_model_name(args.response_model_name),
        "test_mode": args.test_mode,
        "lambda_reg": args.lambda_reg,
        "n_features": args.n_features,
        "budget": args.Budget,
        "pool_size": args.pool_size,
        "eval_num": eval_num,
        "reward_type": args.reward_type,
        "alpha": args.alpha,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_taken": end_time - start_time,
    }
    analysis.update(meta_info)
    trajectories_to_save = meta_info.copy()
    trajectories_to_save.update({"trajectories": trajectories})

    metrics_to_save = meta_info.copy()
    metrics_to_save.update({"metrics": metrics_list})

    save_result_analysis(analysis, args.save_analysis_path)
    save_trajectories(trajectories_to_save, args.save_trajectory_path, f"{args.test_mode}")
    save_metrics(metrics_to_save, args.save_metrics_path, f"{args.test_mode}")

    logger.info(f"Total time taken: {end_time - start_time:.2f} seconds for exploration with {args.Budget} budget and {args.pool_size} pool size and {eval_num} eval num")


if __name__ == "__main__":
    setup_logging(task_name="UCB")

    parser = argparse.ArgumentParser()
    parser.add_argument("--Budget", type=int, default=20)
    parser.add_argument("--pool_size", type=int, default=3)
    parser.add_argument("--judge_model_name", type=str, default="gemini-2.0-flash")
    parser.add_argument("--judge_type", type=str, default="pointwise", choices=["pointwise", "pairwise", "pairwise_fine_grained", "alpaca_eval", "arena_hard_auto", "mt_bench", "mlr_bench"])
    parser.add_argument("--answer_position", type=str, default=None, choices=["first", "second", None], help="The position of the answer in the pairwise comparison")
    parser.add_argument("--baseline_response_model_name", type=str, default=None, help="The model name of the baseline response model")
    parser.add_argument("--llm_agent_name", type=str, default="gpt-4.1-nano")
    parser.add_argument("--response_model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--dataset_name", type=str, default="AlpacaEval")
    parser.add_argument("--reward_type", type=str, default="relative", choices=["relative", "absolute"])
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--n_features", type=int, default=384)
    parser.add_argument("--cold_start", type=bool, default=False)
    parser.add_argument("--init_model_path", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/models/")
    parser.add_argument("--test_mode", type=str, default="ucb", choices=["ucb", "random"])
    parser.add_argument("--data_dir", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data")
    parser.add_argument("--eval_num", type=int, default=805)
    parser.add_argument("--save_analysis_path", type=str, default="results/")
    parser.add_argument("--save_trajectory_path", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/")
    parser.add_argument("--save_metrics_path", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/metrics/")

    parser.add_argument("--alpha", type=float, default=1.0)

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    # Run the async main function
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    finally:
        logger.info("Application finished.")