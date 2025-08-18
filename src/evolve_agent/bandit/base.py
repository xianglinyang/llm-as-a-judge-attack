'''Base class for Contextual Bandit Agents.

TODO: Can try Neural Bandit in the future.
'''
import numpy as np
import random
import logging
from abc import abstractmethod
import pickle
import os
from typing import List, Optional, Tuple, Dict


from src.evolve_agent import EvolveAgent
from src.llm_zoo import BaseLLM
from src.text_encoder import TextEncoder
from src.llm_evaluator import JudgeType
from src.evolve_agent.bias_strategies import Bias_types, BiasModification
from src.evolve_agent.bandit.reward_cal import create_reward_calculator
from src.evolve_agent.utils import _worst_index, _best_item, _select_from_pool_uniform, _estimate_tokens

logger = logging.getLogger(__name__)



class ContextualBanditAgent(EvolveAgent):
    def __init__(self, n_features: int, llm_agent: BaseLLM, embedding_model: TextEncoder, judge_type: JudgeType, judge_model_backbone: str, reward_type: str = "relative", answer_position: str = "first"):
        """
        Initializes the Bandit Agent.

        Args:
            n_features (int): Dimension of context features (d).
            llm_agent (BaseLLM): LLM agent to generate the response. 
            embedding_model (TextEncoder): Embedding model to encode the context.
            judge_type (JudgeType): Type of judge evaluation (pointwise, pairwise, etc.)
            judge_model_backbone (str): Backbone model for the judge
            reward_type (str): Type of reward to use ("relative" or "absolute").
        """
        super().__init__(llm_agent, judge_type, judge_model_backbone, reward_type, answer_position)
        self.n_arms = len(Bias_types) # number of arms
        self.strategy_list = Bias_types
        self.n_features = n_features
        self.embedding_model = embedding_model
        self.bias_modifier = BiasModification(llm_agent)
        
        # Initialize the reward calculator using the factory function
        self.reward_calculator = create_reward_calculator(judge_type, judge_model_backbone, reward_type, answer_position)
    
    def validate_judge_requirements(self, baseline_response=None, baseline_response_list=None):
        """
        Validate that the judge requirements are met for the current judge type.
        
        Args:
            baseline_response (str, optional): Baseline response for single evaluation
            baseline_response_list (list[str], optional): List of baseline responses for batch evaluation
        """
        if self.judge_type in [JudgeType.PAIRWISE, JudgeType.PAIRWISE_FINE_GRAINED, JudgeType.ALPACA_EVAL, JudgeType.ARENA_HARD_AUTO]:
            if baseline_response is None and baseline_response_list is None:
                raise ValueError(f"Baseline response is required for {self.judge_type} evaluation")
            if baseline_response_list is not None and None in baseline_response_list:
                raise ValueError(f"All baseline responses must be provided for {self.judge_type} evaluation")
    
    @abstractmethod
    def init_policy_model(self, num: int):
        pass

    @abstractmethod
    def predict(self, context_x, model_idx):
        """
        Predicts the score for each arm given the context.

        Args:
            context_x (np.array): A (n_features x 1) column vector representing the context.

        Returns:
            np.array: A (n_arms x 1) vector of UCB scores for each arm.
        """
        pass
    
    @abstractmethod
    def choose_arm(self, context_x, model_idx):
        """
        Chooses an arm based on the specific strategy.

        Args:
            context_x (np.array): A (n_features x 1) column vector representing the context.

        Returns:
            int: The index of the chosen arm.
        """
        pass

    @abstractmethod
    def batch_predict(self, context_x_list, model_idxs):
        """
        Predicts the score for each arm given the context.
        Args:
            context_x_list: np.ndarray of shape (n_samples, n_features, 1)
            model_idxs: List[int] of length n_samples
        Returns:
            np.ndarray: A (n_samples, n_arms, 1) tensor of scores for each arm.
        """
        pass

    @abstractmethod
    def batch_choose_arm(self, context_x_list, model_idxs):
        """
        Chooses an arm per sample.
        Args:
            context_x_list: np.ndarray of shape (n_samples, n_features, 1)
            model_idxs: List[int] of length n_samples
        Returns:
            List[int] or 1-D np.ndarray of length n_samples with chosen arm indices.
        """
        pass

    @abstractmethod
    def update(self, chosen_arm_idx, context_x, reward, model_idx):
        """
        Updates the parameters for the chosen arm.

        Args:
            chosen_arm_idx (int): The index of the arm that was played.
            context_x (np.array): The context vector (n_features x 1) for which the arm was played.
            reward (float): The observed reward.
        """
        pass
        
    def get_context_x(self, question: str, response: str):
        text = "Question: " + question + "\n" + "Response: " + response
        embedding = self.embedding_model.encode(text)
        # Return as column vector for consistency
        return embedding.reshape(-1, 1)
    
    def get_context_x_batch(self, question_list: list[str], response_list: list[str]):
        texts = ["Question: " + question + "\n" + "Response: " + response for question, response in zip(question_list, response_list)]
        embeddings = self.embedding_model.batch_encode(texts)
        # Return as (n_samples, n_features, 1) for consistency
        if embeddings.ndim == 2:
            embeddings = embeddings[:, :, None]
        return embeddings
    
    def get_reward(self, question: str, response: str, original_score: float, baseline_response: str = None):
        """
        Get reward for a single response using the configured reward calculator.
        
        Args:
            question (str): The input question
            response (str): The response to evaluate
            original_score (float): Original score for comparison
            baseline_response (str, optional): Baseline response for pairwise evaluation
            
        Returns:
            tuple[float, float, str]: (reward, score, explanation)
        """
        if self.judge_type in [JudgeType.PAIRWISE, JudgeType.PAIRWISE_FINE_GRAINED, JudgeType.ALPACA_EVAL, JudgeType.ARENA_HARD_AUTO]:
            return self.reward_calculator.calculate_reward(question, response, original_score, baseline_response)
        else:
            return self.reward_calculator.calculate_reward(question, response, original_score)
    
    async def get_batch_reward(self, question_list: list[str], response_list: list[str], original_score_list: list[float], baseline_response_list: list[str] = None):
        """
        Get rewards for a batch of responses using the configured reward calculator.
        
        Args:
            question_list (list[str]): List of input questions
            response_list (list[str]): List of responses to evaluate
            original_score_list (list[float]): List of original scores
            baseline_response_list (list[str], optional): List of baseline responses for pairwise evaluation
            
        Returns:
            tuple[list[float], list[float], list[str]]: (reward_list, score_list, explanation_list)
        """
        if self.judge_type in [JudgeType.PAIRWISE, JudgeType.PAIRWISE_FINE_GRAINED, JudgeType.ALPACA_EVAL, JudgeType.ARENA_HARD_AUTO]:
            return await self.reward_calculator.calculate_batch_reward(question_list, response_list, original_score_list, baseline_response_list)
        else:
            return await self.reward_calculator.calculate_batch_reward(question_list, response_list, original_score_list)
    
    def explore(
        self,
        question: str,
        init_response: str,
        original_score: float,
        original_explanation: str,
        pool_size: int,
        Budget: int,
        cold_start: bool,
        init_model_path: Optional[str] = None,
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
        if init_model_path is not None:
            self.load_policy_model(init_model_path)
            if len(self.As) != 1:
                raise ValueError(f"Expected 1 model for single exploration, got {len(self.As)}")
        else:
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
        }
        replacements = 0
        totals_tokens = 0
        best0 = float(original_score)

        # ---- 2) Optional cold start (probe each arm once from the init response)
        curr_step = 0
        if cold_start:
            x = self.get_context_x(question, init_response)
            x = x.reshape(-1, 1)
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
            x = self.get_context_x(question, curr_answer).reshape(-1, 1)

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
                    As, bs, As_inv, thetas = [], [], [], []
                    for _ in range(n):
                        As.append([A.copy() for A in self.As[0]])
                        bs.append([b.copy() for b in self.bs[0]])
                        As_inv.append([A_inv.copy() for A_inv in self.As_inv[0]])
                        thetas.append([theta.copy() for theta in self.thetas[0]])
                    self.As, self.bs, self.As_inv, self.thetas = As, bs, As_inv, thetas
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
            pools.append(item)
            metrics_list.append({
                "best_so_far": [],
                "pool_mean": [],
                "replacement_ratio": [],
                "lift_per_1k_tokens": [],
            })
        
        # ---- 3) Optional cold-start: probe each arm once per question from the init answer
        rounds_consumed = 0
        if cold_start:
            # Build contexts from (question, init_answer)
            context_x = self.get_context_x_batch(question_list, init_response_list)
            context_x = np.asarray(context_x)
            if context_x.ndim == 2:
                context_x = context_x[:, :, None]  # (n, d, 1)

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
            context_x = np.asarray(context_x)
            if context_x.ndim == 2:
                context_x = context_x[:, :, None]  # (n, d, 1)

            # (c) choose arms with LinUCB (shape: (n, 1) or list of [arm])
            chosen_arm_list = self.batch_choose_arm(context_x, list(range(len(context_x))))

            # (d) mutate answers according to chosen arms
            strategy_list = [
                self.bias_modifier.Bias_types[int(a[0])] if isinstance(a, (list, np.ndarray)) else self.bias_modifier.Bias_types[int(a)]
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


        # ---- 5) Collect best items
        best_items: List[Dict] = []
        for i in range(n):
            best_item = _best_item(pools[i])
            best_items.append(best_item)

        return best_items, metrics_list
    
    # --------------------------
    # Single-question (random arm)
    # --------------------------
    def explore_with_random_arm(
        self,
        question: str,
        init_response: str,
        original_score: float,
        original_explanation: str,
        pool_size: int,
        Budget: int,
        baseline_response: Optional[str] = None,
        replacement_margin: float = 0.0,
    ) -> Tuple[Dict, Dict]:
        """
        Random-arm baseline explorer (single question).
        Returns:
        best_item: dict (question/score/answer/explanation/origin/tokens/history)
        metrics: dict (best_so_far, pool_mean, replacement_ratio, lift_per_1k_tokens)
        """
        # 0) Validate judge requirements
        self.validate_judge_requirements(baseline_response=baseline_response)
        
        # 1) Init pool + metrics
        init_item = {
            "question": question,
            "score": float(original_score),
            "answer": init_response,
            "explanation": original_explanation,
            "origin": "init",
            "tokens": _estimate_tokens(init_response),
            "history": [(float(original_score), original_explanation, init_response, "init")],
        }
        pool: List[Dict] = [init_item]

        metrics = {
            "best_so_far": [],
            "pool_mean": [],
            "replacement_ratio": [],
            "lift_per_1k_tokens": [],
        }
        replacements = 0
        totals_tokens = 0
        best0 = float(original_score)

        # 2) Main loop
        for t in range(Budget):
            # pick a seed uniformly
            si = random.randrange(len(pool))
            seed = pool[si]
            curr_answer = seed["answer"]
            curr_score = seed["score"]

            # random arm + strategy
            chosen_arm = random.randrange(self.n_arms)
            strategy = Bias_types[chosen_arm]

            # mutate
            new_response = self.bias_modifier.principle_guided_mutation(curr_answer, strategy)

            # judge (pairwise vs baseline if provided)
            reward, new_score, new_expl = self.get_reward(
                question, new_response, curr_score, baseline_response
            )

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

            # pool replacement (greedy)
            if len(pool) < pool_size:
                pool.append(new_item)
                replacements += 1
            else:
                wi = _worst_index(pool)
                if new_item["score"] > pool[wi]["score"] + replacement_margin:
                    pool[wi] = new_item
                    replacements += 1

            # metrics
            best_now = _best_item(pool)["score"]
            pool_mean = sum(xi["score"] for xi in pool) / len(pool)
            metrics["best_so_far"].append(best_now)
            metrics["pool_mean"].append(pool_mean)
            metrics["replacement_ratio"].append(replacements / (t + 1))
            denom = max(1, totals_tokens / 1000)
            metrics["lift_per_1k_tokens"].append((best_now - best0) / denom)

            logger.info(f"[Random Iter {t}] seed={curr_score:.3f} → new={new_score:.3f} arm={strategy}")

        best_item = _best_item(pool)
        return best_item, metrics


    # --------------------------
    # Batched (random arm)
    # --------------------------
    async def batch_explore_with_random_arm(
        self,
        question_list: List[str],
        init_response_list: List[str],
        original_score_list: List[float],
        original_explanation_list: List[str],
        Budget: int,
        pool_size: int,
        baseline_response_list: Optional[List[str]] = None,
        replacement_margin: float = 0.0,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Random-arm baseline explorer (batched).
        Returns:
        best_items: list[dict] (one per question)
        metrics_list: list[dict] (one per question)
        """
        # 0) Validate judge requirements (match simple_rewrite signature)
        self.validate_judge_requirements(baseline_response=baseline_response_list)

        n = len(question_list)
        assert len(init_response_list) == n
        assert len(original_score_list) == n
        assert len(original_explanation_list) == n
        if baseline_response_list is not None:
            assert len(baseline_response_list) == n

        # 1) Init pools + metrics
        pools: List[List[Dict]] = []
        metrics_list: List[Dict] = []
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
            })

        # 2) Main loop
        for t in range(Budget):
            # (a) choose one seed per question
            seeds = []
            curr_answers = []
            for pool in pools:
                si = random.randrange(len(pool))
                seed = pool[si]
                seeds.append(seed)
                curr_answers.append(seed["answer"])

            # (b) choose random arm independently per question
            chosen_arm_list = [random.randrange(self.n_arms) for _ in range(n)]
            strategy_list = [Bias_types[a] for a in chosen_arm_list]

            # (c) mutate as a batch
            try:
                new_responses = await self.bias_modifier.batch_principle_guided_mutation(
                    curr_answers, strategy_list
                )
            except Exception:
                new_responses = [None] * n

            # (d) judge as a batch
            try:
                _, new_scores, new_expls = await self.get_batch_reward(
                    question_list, new_responses, original_score_list, baseline_response_list
                )
            except Exception:
                new_scores = [seeds[i]["score"] for i in range(n)]
                new_expls = [seeds[i]["explanation"] for i in range(n)]

            # (e) pool replacement
            for i in range(n):
                if new_responses[i] is None:
                    continue
                pool = pools[i]
                seed = seeds[i]

                new_item = {
                    "question": question_list[i],
                    "score": float(new_scores[i]),
                    "answer": new_responses[i],
                    "explanation": new_expls[i],
                    "origin": strategy_list[i],
                    "tokens": _estimate_tokens(new_responses[i]),
                    "history": seed["history"] + [(float(new_scores[i]), new_expls[i], new_responses[i], strategy_list[i])],
                }
                totals_tokens[i] += new_item["tokens"]

                if len(pool) < pool_size:
                    pool.append(new_item)
                    replacements[i] += 1
                else:
                    wi = _worst_index(pool)
                    if new_item["score"] > pool[wi]["score"] + replacement_margin:
                        pool[wi] = new_item
                        replacements[i] += 1

            # (f) metrics this round
            for i in range(n):
                pool = pools[i]
                best_now = _best_item(pool)["score"]
                pool_mean = sum(x["score"] for x in pool) / len(pool)
                m = metrics_list[i]
                m["best_so_far"].append(best_now)
                m["pool_mean"].append(pool_mean)
                m["replacement_ratio"].append(replacements[i] / (t + 1))
                denom = max(1, totals_tokens[i] / 1000)
                m["lift_per_1k_tokens"].append((best_now - best0_list[i]) / denom)

        # 3) Gather best items
        best_items = [_best_item(pools[i]) for i in range(n)]
        return best_items, metrics_list

class ContextualLinBanditAgent(ContextualBanditAgent):
    def __init__(
        self,
        n_features: int,
        llm_agent: BaseLLM,
        embedding_model: TextEncoder,
        judge_type: JudgeType,
        judge_model_backbone: str,
        reward_type: str = "relative",
        lambda_reg: float = 1.0,
        answer_position: str = None,
    ):
        super().__init__(n_features, llm_agent, embedding_model, judge_type, judge_model_backbone, reward_type, answer_position)
        if lambda_reg <= 0:
            raise ValueError("lambda_reg must be positive.")
        self.lambda_reg = float(lambda_reg)

    def init_policy_model(self, num: int):
        d = self.n_features
        self.As, self.bs, self.As_inv, self.thetas = [], [], [], []
        self._updates_count = []  # for periodic refactorization

        I = np.eye(d, dtype=float)
        inv_init = (1.0 / self.lambda_reg) * I

        for _ in range(num):
            A_row = [self.lambda_reg * I.copy() for _ in range(self.n_arms)]
            b_row = [np.zeros((d, 1), dtype=float) for _ in range(self.n_arms)]
            inv_row = [inv_init.copy() for _ in range(self.n_arms)]
            theta_row = [np.zeros((d, 1), dtype=float) for _ in range(self.n_arms)]
            cnt_row = [0 for _ in range(self.n_arms)]

            self.As.append(A_row)
            self.bs.append(b_row)
            self.As_inv.append(inv_row)
            self.thetas.append(theta_row)
            self._updates_count.append(cnt_row)

    def _chol_inverse(self, A: np.ndarray) -> np.ndarray:
        # A should be SPD (lambda*I + sum xx^T). Use Cholesky for stability.
        L = np.linalg.cholesky(A)
        return np.linalg.solve(L.T, np.linalg.solve(L, np.eye(A.shape[0], dtype=A.dtype)))

    def load_policy_model(self, path: str):
        As_path = os.path.join(path, "As.pkl")
        bs_path = os.path.join(path, "bs.pkl")
        As_inv_path = os.path.join(path, "As_inv.pkl")
        thetas_path = os.path.join(path, "thetas.pkl")

        with open(As_path, "rb") as f:
            self.As = pickle.load(f)
        with open(bs_path, "rb") as f:
            self.bs = pickle.load(f)

        if len(self.As) != len(self.bs):
            raise ValueError("Loaded As and bs lengths differ.")
        if any(len(row) != self.n_arms for row in self.As):
            raise ValueError("Loaded As row does not match n_arms.")
        if any(len(row) != self.n_arms for row in self.bs):
            raise ValueError("Loaded bs row does not match n_arms.")

        if os.path.exists(As_inv_path) and os.path.exists(thetas_path):
            with open(As_inv_path, "rb") as f:
                self.As_inv = pickle.load(f)
            with open(thetas_path, "rb") as f:
                self.thetas = pickle.load(f)
        else:
            # Reconstruct cached structures stably
            self.As_inv, self.thetas = [], []
            for As_model, bs_model in zip(self.As, self.bs):
                inv_row, theta_row = [], []
                for A_a, b_a in zip(As_model, bs_model):
                    A_inv = self._chol_inverse(A_a)
                    inv_row.append(A_inv)
                    theta_row.append(A_inv @ b_a)
                self.As_inv.append(inv_row)
                self.thetas.append(theta_row)

        # Reset/update counters
        self._updates_count = [[0 for _ in range(self.n_arms)] for _ in range(len(self.As))]

    def save_policy_model(self, path: str):
        os.makedirs(path, exist_ok=True)
        As_path = os.path.join(path, "As.pkl")
        bs_path = os.path.join(path, "bs.pkl")
        As_inv_path = os.path.join(path, "As_inv.pkl")
        thetas_path = os.path.join(path, "thetas.pkl")
        with open(As_path, "wb") as f:
            pickle.dump(self.As, f)
        with open(bs_path, "wb") as f:
            pickle.dump(self.bs, f)
        with open(As_inv_path, "wb") as f:
            pickle.dump(self.As_inv, f)
        with open(thetas_path, "wb") as f:
            pickle.dump(self.thetas, f)

    def update(self, chosen_arm_idx: int, context_x: np.ndarray, reward: float, model_idx: int) -> None:
        # Index guards
        if not (0 <= model_idx < len(self.As)):
            raise IndexError("model_idx out of range.")
        if not (0 <= chosen_arm_idx < self.n_arms):
            raise IndexError("chosen_arm_idx out of range.")

        # Shape/ dtype guard
        if context_x.shape == (self.n_features,):
            context_x = context_x.reshape(-1, 1)
        if context_x.shape != (self.n_features, 1):
            raise ValueError(f"context_x must be ({self.n_features}, 1) or ({self.n_features},).")

        # Refs
        A = self.As[model_idx][chosen_arm_idx]
        A_inv = self.As_inv[model_idx][chosen_arm_idx]
        b = self.bs[model_idx][chosen_arm_idx]
        theta = self.thetas[model_idx][chosen_arm_idx]

        # Sherman–Morrison rank-1 update
        z = A_inv @ context_x                           # (d,1)
        denom = 1.0 + float(context_x.T @ z)            # scalar, >= 1
        if denom <= 1e-12 or not np.isfinite(denom):
            # Fallback to stable recompute
            A_new = A + (context_x @ context_x.T)
            b_new = b + float(reward) * context_x
            A_inv_new = self._chol_inverse(A_new)
            theta_new = A_inv_new @ b_new
        else:
            A_new = A + (context_x @ context_x.T)
            A_inv_new = A_inv - (z @ z.T) / denom
            b_new = b + float(reward) * context_x

            # Fast theta update:
            theta_dot_x = float(theta.T @ context_x)
            x_dot_z = float(context_x.T @ z)
            v = theta_dot_x + float(reward) * x_dot_z
            theta_new = theta + float(reward) * z - (z * (v / denom))

        # Commit
        self.As[model_idx][chosen_arm_idx] = A_new
        self.As_inv[model_idx][chosen_arm_idx] = A_inv_new
        self.bs[model_idx][chosen_arm_idx] = b_new
        self.thetas[model_idx][chosen_arm_idx] = theta_new

        # Periodic re-factorization to limit drift
        self._updates_count[model_idx][chosen_arm_idx] += 1
        if (self._updates_count[model_idx][chosen_arm_idx] % 200) == 0:
            A_ref = self.As[model_idx][chosen_arm_idx]
            b_ref = self.bs[model_idx][chosen_arm_idx]
            # Enforce symmetry
            A_ref = 0.5 * (A_ref + A_ref.T)
            A_inv_ref = self._chol_inverse(A_ref)
            theta_ref = A_inv_ref @ b_ref
            self.As[model_idx][chosen_arm_idx] = A_ref
            self.As_inv[model_idx][chosen_arm_idx] = A_inv_ref
            self.thetas[model_idx][chosen_arm_idx] = theta_ref
