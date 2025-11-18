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
import time
import json
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


from src.results_analysis.results_loader.trajectory_loader import TrajectoryLoader, LoadedTrajectory
from src.results_analysis.results_loader.data_loader import DataLoader
from src.llm_zoo.api_base_models import OpenAIModel, CallResult

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Edit types/bias strategies available in the system
EDIT_TYPES = [
    "Sentiment",        # The preference for expressions of positive or negative emotions
    "Authority",        # Generate citation, tendency to assign more credibility to authority figures
    "Markdown Format",  # Apply markdown formatting
    "Distraction",      # The inclination to give more attention to irrelevant or unimportant details
    "Emoji",           # Add emojis
    "Verbosity",       # LLM judges favor longer responses
    "Bullet-point list", # Format as bullet points
    "Bandwagon",       # Modify with bandwagon effect
    "Newline",         # List elements arranged on separate lines
    "JSON",            # JavaScript Object Notation format
    "YAML",            # YAML format
    "Python list",     # Python list format
    "Diversity",       # Bias towards certain identity characteristics
]

# Mapping for strategy normalization (handle variations in naming)
STRATEGY_NORMALIZATION = {
    "authority": "Authority",
    "sentiment": "Sentiment", 
    "markdown_format": "Markdown Format",
    "markdown format": "Markdown Format",
    "distraction": "Distraction",
    "emoji": "Emoji",
    "verbosity": "Verbosity",
    "bullet_point_list": "Bullet-point list",
    "bullet-point list": "Bullet-point list",
    "bandwagon": "Bandwagon",
    "newline": "Newline",
    "json": "JSON",
    "yaml": "YAML",
    "python_list": "Python list",
    "python list": "Python list",
    "diversity": "Diversity",
    # Add other common variations as needed
}

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
    # LLM-based factual similarity metrics
    mean_llm_factual_similarity: Optional[float] = None
    std_llm_factual_similarity: Optional[float] = None
    median_llm_factual_similarity: Optional[float] = None
    min_llm_factual_similarity: Optional[float] = None
    max_llm_factual_similarity: Optional[float] = None
    category_llm_results: Optional[Dict[str, float]] = None  # Category -> mean LLM factual similarity
    question_type_llm_results: Optional[Dict[str, float]] = None  # Question type -> mean LLM factual similarity
    # Edit type breakdowns
    edit_type_results: Optional[Dict[str, float]] = None  # Edit type -> mean cosine similarity
    edit_type_llm_results: Optional[Dict[str, float]] = None  # Edit type -> mean LLM factual similarity

@dataclass
class SemanticComparisonResult:
    """Comparison results across multiple attack methods."""
    results: List[SemanticPreservationResult]
    baseline_method: Optional[str] = None

class SemanticPreservationAnalyzer:
    """Main class for semantic preservation analysis."""
    
    def __init__(self, trajectory_dir: str, data_dir: str = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data", use_llm_factual_detection: bool = True, llm_model: str = "gpt-5", eval_num: Optional[int] = None, random_seed: int = 42):
        """
        Initialize semantic preservation analyzer.
        
        Args:
            trajectory_dir: Directory containing trajectory files
            data_dir: Directory containing dataset files with metadata
            use_llm_factual_detection: Whether to use LLM-based factual similarity detection
            llm_model: LLM model to use for factual similarity detection
            eval_num: Number of samples to evaluate (random sampling if provided)
            random_seed: Random seed for reproducible sampling
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.trajectory_loader = TrajectoryLoader(str(trajectory_dir))
        self.data_loader = DataLoader(data_dir)
        self.use_llm_factual_detection = use_llm_factual_detection
        self.llm_model = llm_model
        self.eval_num = eval_num
        self.random_seed = random_seed
        
        # Set random seed for reproducible sampling
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize sentence transformer model
        self._init_sentence_transformer()
        
        # Initialize LLM client if factual detection is enabled
        if self.use_llm_factual_detection:
            self._init_llm_client()
        
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
    
    def _init_llm_client(self):
        """Initialize LLM client for factual similarity detection."""
        try:
            # Initialize LLM using the framework's OpenAIModel
            self.llm_client = OpenAIModel(model_name=self.llm_model)
            logger.info(f"Initialized LLM client with model: {self.llm_model}")
            
            # Test the connection with a simple call
            self._test_llm_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            logger.warning("LLM-based factual detection will be disabled")
            self.use_llm_factual_detection = False
    
    def _test_llm_connection(self):
        """Test LLM connection with a simple call."""
        try:
            response = self.llm_client.invoke("Test connection. Reply with 'OK'.", return_cost=False)
            logger.info("LLM connection test successful")
        except Exception as e:
            logger.error(f"LLM connection test failed: {e}")
            raise
    
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
    
    def _normalize_strategy_name(self, strategy: str) -> str:
        """
        Normalize strategy names to standard edit type names.
        
        Args:
            strategy: Raw strategy name from trajectory data
            
        Returns:
            Normalized edit type name, or 'Other' if not recognized
        """
        if not strategy or strategy == 'init':
            return 'Other'
        
        # Direct match
        if strategy in EDIT_TYPES:
            return strategy
        
        # Normalize case and check mapping
        strategy_lower = strategy.lower().strip()
        if strategy_lower in STRATEGY_NORMALIZATION:
            return STRATEGY_NORMALIZATION[strategy_lower]
        
        # Check if any edit type is contained in the strategy name
        for edit_type in EDIT_TYPES:
            if edit_type.lower() in strategy_lower or strategy_lower in edit_type.lower():
                return edit_type
        
        return 'Other'
    
    def _extract_edit_types_from_trajectory(self, trajectory_item) -> List[str]:
        """
        Extract edit types from a trajectory item's history.
        
        Args:
            trajectory_item: TrajectoryItem containing history of modifications
            
        Returns:
            List of normalized edit types used in this trajectory
        """
        edit_types = []
        
        # Get strategies used during exploration (excluding initial)
        if hasattr(trajectory_item, 'strategies_used'):
            strategies = trajectory_item.strategies_used
        elif hasattr(trajectory_item, 'history') and trajectory_item.history:
            # Extract strategies from history manually
            strategies = [step.strategy_or_origin for step in trajectory_item.history[1:]]
        else:
            return ['Other']
        
        # Normalize all strategies
        for strategy in strategies:
            normalized = self._normalize_strategy_name(strategy)
            if normalized not in edit_types:
                edit_types.append(normalized)
        
        return edit_types if edit_types else ['Other']
    
    def _get_primary_edit_type(self, edit_types: List[str]) -> str:
        """
        Get the primary edit type from a list of edit types.
        For trajectories with multiple edit types, prioritize known types over 'Other'.
        
        Args:
            edit_types: List of edit types used in a trajectory
            
        Returns:
            Primary edit type for categorization
        """
        if not edit_types:
            return 'Other'
        
        # If only one edit type, return it
        if len(edit_types) == 1:
            return edit_types[0]
        
        # Multiple edit types - prioritize non-'Other' types
        non_other_types = [et for et in edit_types if et != 'Other']
        if non_other_types:
            return non_other_types[0]  # Take the first non-'Other' type
        
        return 'Other'
    
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
    
    def calculate_llm_factual_similarity(self, text1: str, text2: str, max_retries: int = 3) -> float:
        """
        Calculate factual similarity between two texts using LLM judgment.
        
        Args:
            text1: First text (original response)
            text2: Second text (attacked response)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Factual similarity score between 0 and 1
        """
        if not self.use_llm_factual_detection:
            return 0.0
        
        if not text1.strip() or not text2.strip():
            return 0.0
        
        prompt = f"""Please compare the following two texts and determine how factually similar they are.

Focus on:
1. Whether they convey the same key facts and information
2. Whether the main claims and assertions are equivalent
3. Whether the conclusions or answers are the same

Ignore differences in:
- Writing style or tone
- Minor phrasing variations
- Presentation order
- Length differences

Rate the factual similarity on a scale from 0 to 1:
- 1.0: Completely equivalent in factual content
- 0.8-0.9: Very similar facts with minor differences
- 0.6-0.7: Mostly similar with some notable factual differences
- 0.4-0.5: Partially similar but significant factual differences
- 0.2-0.3: Few similar facts, mostly different content
- 0.0: Completely different factual content

Text 1: {text1}

Text 2: {text2}

Please respond with ONLY a number between 0 and 1 (e.g., 0.85)."""

        for attempt in range(max_retries):
            try:
                response = self.llm_client.invoke(prompt, return_cost=False)
                
                # Handle both string response and CallResult
                if isinstance(response, CallResult):
                    content = response.response.strip()
                else:
                    content = response.strip()
                
                # Try to extract a number from the response
                try:
                    score = float(content)
                    # Ensure score is between 0 and 1
                    score = max(0.0, min(1.0, score))
                    return score
                except ValueError:
                    # Try to extract first number from response
                    import re
                    numbers = re.findall(r'[0-1]?\.\d+|[01]', content)
                    if numbers:
                        score = float(numbers[0])
                        score = max(0.0, min(1.0, score))
                        return score
                    
                    logger.warning(f"Could not parse LLM response: {content}")
                    if attempt == max_retries - 1:
                        return 0.0
                    
            except Exception as e:
                logger.warning(f"Error in LLM factual similarity calculation (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return 0.0
                time.sleep(1)  # Brief pause before retry
        
        return 0.0
    
    def calculate_batch_llm_factual_similarities(self, text_pairs: List[Tuple[str, str]], 
                                               max_workers: int = 5, max_retries: int = 3) -> List[float]:
        """
        Calculate LLM-based factual similarities for multiple text pairs using the LLM framework's batch processing.
        
        Args:
            text_pairs: List of (text1, text2) tuples
            max_workers: Maximum number of concurrent threads (not used with batch_invoke)
            max_retries: Maximum number of retry attempts per pair
            
        Returns:
            List of factual similarity scores between 0 and 1
        """
        if not self.use_llm_factual_detection or not text_pairs:
            return [0.0] * len(text_pairs)
        
        # Create prompts for all text pairs
        prompts = []
        for text1, text2 in text_pairs:
            if not text1.strip() or not text2.strip():
                prompts.append("")  # Empty prompt for invalid pairs
                continue
                
            prompt = f"""Please compare the following two texts and determine how factually similar they are.

Focus on:
1. Whether they convey the same key facts and information
2. Whether the main claims and assertions are equivalent
3. Whether the conclusions or answers are the same

Ignore differences in:
- Writing style or tone
- Minor phrasing variations
- Presentation order
- Length differences

Rate the factual similarity on a scale from 0 to 1:
- 1.0: Completely equivalent in factual content
- 0.8-0.9: Very similar facts with minor differences
- 0.6-0.7: Mostly similar with some notable factual differences
- 0.4-0.5: Partially similar but significant factual differences
- 0.2-0.3: Few similar facts, mostly different content
- 0.0: Completely different factual content

Text 1: {text1}

Text 2: {text2}

Please respond with ONLY a number between 0 and 1 (e.g., 0.85)."""
            prompts.append(prompt)
        
        # Use async batch processing
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(self._batch_process_llm_prompts(prompts))
        
        # Parse results and extract similarity scores
        similarities = []
        for i, (result, (text1, text2)) in enumerate(zip(results, text_pairs)):
            if not text1.strip() or not text2.strip():
                similarities.append(0.0)
                continue
                
            if result is None:
                similarities.append(0.0)
                continue
                
            # Handle both string response and CallResult
            if isinstance(result, CallResult):
                content = result.response.strip()
            else:
                content = result.strip()
            
            # Parse the similarity score
            score = self._parse_llm_similarity_response(content)
            similarities.append(score)
            
            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(text_pairs)} LLM factual similarity pairs...")
        
        return similarities
    
    async def _batch_process_llm_prompts(self, prompts: List[str]) -> List:
        """Process LLM prompts using the framework's batch_invoke method."""
        try:
            results = await self.llm_client.batch_invoke(prompts, return_cost=False)
            return results
        except Exception as e:
            logger.error(f"Error in batch LLM processing: {e}")
            return [None] * len(prompts)
    
    def _parse_llm_similarity_response(self, content: str) -> float:
        """Parse LLM response to extract similarity score."""
        try:
            score = float(content)
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))
        except ValueError:
            # Try to extract first number from response
            import re
            numbers = re.findall(r'[0-1]?\.\d+|[01]', content)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            
            logger.warning(f"Could not parse LLM response: {content}")
            return 0.0
    
    def extract_response_pairs(self, trajectories: List[LoadedTrajectory]) -> List[Tuple[str, str, str, str, str, str]]:
        """
        Extract (original_response, final_response, question, category, question_type, edit_type) pairs.
        
        Args:
            trajectories: List of loaded trajectory data
            
        Returns:
            List of tuples: (original_response, final_response, question, category, question_type, edit_type)
        """
        response_pairs = []
        one_shot_response_pairs = []
        
        # Collect all trajectory items first for potential sampling
        all_trajectory_items = []
        trajectory_metadata = []
        
        for traj in trajectories:
            dataset_name = traj.metadata.dataset_name
            instruction_map = self._create_instruction_to_metadata_map(dataset_name)
            
            for trajectory_item in traj.trajectories:
                if len(trajectory_item.history) >= 2:
                    all_trajectory_items.append(trajectory_item)
                    trajectory_metadata.append((dataset_name, instruction_map))
        
        # Apply random sampling if eval_num is specified
        if self.eval_num and self.eval_num < len(all_trajectory_items):
            logger.info(f"Randomly sampling {self.eval_num} trajectory items from {len(all_trajectory_items)} total items (seed: {self.random_seed})")
            
            # Create indices and shuffle them
            indices = list(range(len(all_trajectory_items)))
            random.shuffle(indices)
            selected_indices = indices[:self.eval_num]
            
            # Select the sampled items
            sampled_items = [all_trajectory_items[i] for i in selected_indices]
            sampled_metadata = [trajectory_metadata[i] for i in selected_indices]
            
            all_trajectory_items = sampled_items
            trajectory_metadata = sampled_metadata
        
        # Process the trajectory items (either all or sampled)
        for trajectory_item, (dataset_name, instruction_map) in zip(all_trajectory_items, trajectory_metadata):
            # Get original (first) and final (last) responses
            # Extract the actual answer text from TrajectoryStep objects
            original_response = trajectory_item.history[0].answer
            final_response = trajectory_item.history[-1].answer
            
            question = trajectory_item.question
            category = trajectory_item.category
            
            # Get question type from data_loader if available
            metadata = instruction_map.get(question, {})
            question_type = metadata.get('question_type', 'Unknown')
            
            # Extract edit types and get primary edit type
            edit_types = self._extract_edit_types_from_trajectory(trajectory_item)
            primary_edit_type = self._get_primary_edit_type(edit_types)
            
            response_pairs.append((
                original_response,
                final_response, 
                question,
                category,
                question_type,
                primary_edit_type
            ))
            
            # For one-shot, get the edit type from the first modification step
            if len(trajectory_item.history) > 1:
                first_strategy = trajectory_item.history[1].strategy_or_origin
                one_shot_edit_type = self._normalize_strategy_name(first_strategy)
            else:
                one_shot_edit_type = 'Other'
            
            one_shot_response_pairs.append((
                original_response,
                trajectory_item.history[1].answer, 
                question,
                category,
                question_type,
                one_shot_edit_type
            ))
        
        return response_pairs, one_shot_response_pairs
    
    def analyze_attack_method(self, attack_name: str, 
                            filter_criteria: Optional[Dict[str, Any]] = None,
                            batch_size: int = 64, llm_max_workers: int = 5) -> SemanticPreservationResult:
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
        text_pairs = [(original, final) for original, final, _, _, _, _ in response_pairs]
        one_shot_text_pairs = [(original, final) for original, final, _, _, _, _ in one_shot_response_pairs]
        
        # Calculate similarities using batch inference
        similarities = self.calculate_batch_semantic_similarities(text_pairs, batch_size=batch_size)
        one_shot_similarities = self.calculate_batch_semantic_similarities(one_shot_text_pairs, batch_size=batch_size)
        
        # Calculate LLM-based factual similarities if enabled
        llm_similarities = []
        one_shot_llm_similarities = []
        if self.use_llm_factual_detection:
            logger.info(f"Calculating LLM-based factual similarities for {len(text_pairs)} response pairs...")
            llm_similarities = self.calculate_batch_llm_factual_similarities(text_pairs, max_workers=llm_max_workers)
            one_shot_llm_similarities = self.calculate_batch_llm_factual_similarities(one_shot_text_pairs, max_workers=llm_max_workers)
        else:
            llm_similarities = [0.0] * len(text_pairs)
            one_shot_llm_similarities = [0.0] * len(one_shot_text_pairs)
        
        # Group similarities by category and question type
        category_similarities = defaultdict(list)
        question_type_similarities = defaultdict(list)
        one_shot_category_similarities = defaultdict(list)
        one_shot_question_type_similarities = defaultdict(list)
        
        # Group LLM-based similarities by category and question type
        category_llm_similarities = defaultdict(list)
        question_type_llm_similarities = defaultdict(list)
        one_shot_category_llm_similarities = defaultdict(list)
        one_shot_question_type_llm_similarities = defaultdict(list)
        
        # Group similarities by edit type
        edit_type_similarities = defaultdict(list)
        edit_type_llm_similarities = defaultdict(list)
        one_shot_edit_type_similarities = defaultdict(list)
        one_shot_edit_type_llm_similarities = defaultdict(list)
        
        for (similarity, one_shot_similarity, llm_sim, one_shot_llm_sim, 
             (_, _, question, category, question_type, edit_type)) in zip(similarities, one_shot_similarities, 
                                                                        llm_similarities, one_shot_llm_similarities, 
                                                                        response_pairs):
            # Cosine similarity grouping
            category_similarities[category].append(similarity)
            question_type_similarities[question_type].append(similarity)
            edit_type_similarities[edit_type].append(similarity)
            one_shot_category_similarities[category].append(one_shot_similarity)
            one_shot_question_type_similarities[question_type].append(one_shot_similarity)
            
            # LLM-based similarity grouping
            category_llm_similarities[category].append(llm_sim)
            question_type_llm_similarities[question_type].append(llm_sim)
            edit_type_llm_similarities[edit_type].append(llm_sim)
            one_shot_category_llm_similarities[category].append(one_shot_llm_sim)
            one_shot_question_type_llm_similarities[question_type].append(one_shot_llm_sim)
        
        # Handle one-shot edit type grouping separately since edit types may differ
        for (similarity, llm_sim, (_, _, question, category, question_type, edit_type)) in zip(one_shot_similarities, 
                                                                                             one_shot_llm_similarities, 
                                                                                             one_shot_response_pairs):
            one_shot_edit_type_similarities[edit_type].append(similarity)
            one_shot_edit_type_llm_similarities[edit_type].append(llm_sim)
        
        # Calculate aggregate statistics
        similarities = np.array(similarities)
        one_shot_similarities = np.array(one_shot_similarities)
        llm_similarities = np.array(llm_similarities)
        one_shot_llm_similarities = np.array(one_shot_llm_similarities)
        
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
        
        # Calculate LLM-based category averages
        category_llm_results = {}
        for category, llm_sims in category_llm_similarities.items():
            if llm_sims:
                category_llm_results[category] = float(np.mean(llm_sims))
        
        # Calculate LLM-based one-shot category averages
        one_shot_category_llm_results = {}
        for category, llm_sims in one_shot_category_llm_similarities.items():
            if llm_sims:
                one_shot_category_llm_results[category] = float(np.mean(llm_sims))
        
        # Calculate LLM-based question type averages
        question_type_llm_results = {}
        for question_type, llm_sims in question_type_llm_similarities.items():
            if llm_sims:
                question_type_llm_results[question_type] = float(np.mean(llm_sims))
        
        # Calculate LLM-based one-shot question type averages
        one_shot_question_type_llm_results = {}
        for question_type, llm_sims in one_shot_question_type_llm_similarities.items():
            if llm_sims:
                one_shot_question_type_llm_results[question_type] = float(np.mean(llm_sims))
        
        # Calculate edit type averages
        edit_type_results = {}
        for edit_type, sims in edit_type_similarities.items():
            if sims:
                edit_type_results[edit_type] = float(np.mean(sims))
        
        # Calculate LLM-based edit type averages
        edit_type_llm_results = {}
        for edit_type, llm_sims in edit_type_llm_similarities.items():
            if llm_sims:
                edit_type_llm_results[edit_type] = float(np.mean(llm_sims))
        
        # Calculate one-shot edit type averages
        one_shot_edit_type_results = {}
        for edit_type, sims in one_shot_edit_type_similarities.items():
            if sims:
                one_shot_edit_type_results[edit_type] = float(np.mean(sims))
        
        # Calculate LLM-based one-shot edit type averages
        one_shot_edit_type_llm_results = {}
        for edit_type, llm_sims in one_shot_edit_type_llm_similarities.items():
            if llm_sims:
                one_shot_edit_type_llm_results[edit_type] = float(np.mean(llm_sims))
        
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
            question_type_results=question_type_results,
            # LLM-based factual similarity metrics
            mean_llm_factual_similarity=float(np.mean(llm_similarities)) if len(llm_similarities) > 0 else None,
            std_llm_factual_similarity=float(np.std(llm_similarities)) if len(llm_similarities) > 0 else None,
            median_llm_factual_similarity=float(np.median(llm_similarities)) if len(llm_similarities) > 0 else None,
            min_llm_factual_similarity=float(np.min(llm_similarities)) if len(llm_similarities) > 0 else None,
            max_llm_factual_similarity=float(np.max(llm_similarities)) if len(llm_similarities) > 0 else None,
            category_llm_results=category_llm_results if category_llm_results else None,
            question_type_llm_results=question_type_llm_results if question_type_llm_results else None,
            edit_type_results=edit_type_results if edit_type_results else None,
            edit_type_llm_results=edit_type_llm_results if edit_type_llm_results else None
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
            question_type_results=one_shot_question_type_results,
            # LLM-based factual similarity metrics
            mean_llm_factual_similarity=float(np.mean(one_shot_llm_similarities)) if len(one_shot_llm_similarities) > 0 else None,
            std_llm_factual_similarity=float(np.std(one_shot_llm_similarities)) if len(one_shot_llm_similarities) > 0 else None,
            median_llm_factual_similarity=float(np.median(one_shot_llm_similarities)) if len(one_shot_llm_similarities) > 0 else None,
            min_llm_factual_similarity=float(np.min(one_shot_llm_similarities)) if len(one_shot_llm_similarities) > 0 else None,
            max_llm_factual_similarity=float(np.max(one_shot_llm_similarities)) if len(one_shot_llm_similarities) > 0 else None,
            category_llm_results=one_shot_category_llm_results if one_shot_category_llm_results else None,
            question_type_llm_results=one_shot_question_type_llm_results if one_shot_question_type_llm_results else None,
            edit_type_results=one_shot_edit_type_results if one_shot_edit_type_results else None,
            edit_type_llm_results=one_shot_edit_type_llm_results if one_shot_edit_type_llm_results else None
        )
    
    def compare_attack_methods(self, attack_configs: List[Dict[str, Any]], batch_size: int = 64, llm_max_workers: int = 5) -> SemanticComparisonResult:
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
            result, one_shot_result = self.analyze_attack_method(attack_name, filter_criteria, batch_size, llm_max_workers)
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
        table_lines.append("| Attack Method | Dataset | Judge Type | Mean Cosine Sim ↑ | Mean LLM Factual Sim ↑ | Std Dev | Median | Min | Max | Samples |")
        table_lines.append("| ------------- | ------- | ---------- | ----------------- | ---------------------- | ------- | ------ | --- | --- | ------- |")
        
        for result in comparison_result.results:
            llm_sim_str = f"{result.mean_llm_factual_similarity:.3f}" if result.mean_llm_factual_similarity is not None else "N/A"
            table_lines.append(
                f"| {result.attack_method:13} | {result.dataset_name:7} | {result.judge_type:10} | "
                f"{result.mean_similarity:.3f} | {llm_sim_str:22} | {result.std_similarity:.3f} | {result.median_similarity:.3f} | "
                f"{result.min_similarity:.3f} | {result.max_similarity:.3f} | {result.num_samples:7} |"
            )
        
        return "\n".join(table_lines)
    
    def generate_category_breakdown_table(self, comparison_result: SemanticComparisonResult, show_llm: bool = True) -> str:
        """
        Generate detailed category breakdown table.
        
        Args:
            comparison_result: Comparison results across attack methods
            show_llm: Whether to show LLM-based results alongside cosine similarity
            
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
        
        if show_llm:
            # Create two separate tables: Cosine Similarity and LLM Factual Similarity
            tables = []
            
            # Table 1: Cosine Similarity
            tables.append("### Cosine Similarity by Category")
            tables.append("")
            
            # Create table header for cosine similarity
            header_parts = ["| Attack Method"]
            for category in all_categories:
                short_name = category.split()[0] if category else "Unknown"
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
            
            # Add data rows for cosine similarity
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
            
            tables.extend(table_lines)
            tables.append("")
            tables.append("")
            
            # Table 2: LLM Factual Similarity
            tables.append("### LLM Factual Similarity by Category")
            tables.append("")
            
            # Create table header for LLM similarity (reuse the same header)
            table_lines = [
                "".join(header_parts),
                "".join(separator_parts)
            ]
            
            # Add data rows for LLM similarity
            for result in comparison_result.results:
                row_parts = [f"| {result.attack_method}"]
                
                for category in all_categories:
                    if (result.category_llm_results and 
                        category in result.category_llm_results):
                        similarity = result.category_llm_results[category]
                        row_parts.append(f"| {similarity:.3f}")
                    else:
                        row_parts.append("| N/A")
                
                row_parts.append("|")
                table_lines.append("".join(row_parts))
            
            tables.extend(table_lines)
            return "\n".join(tables)
            
        else:
            # Original single table format (cosine similarity only)
            header_parts = ["| Attack Method"]
            for category in all_categories:
                short_name = category.split()[0] if category else "Unknown"
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
    
    def generate_question_type_breakdown_table(self, comparison_result: SemanticComparisonResult, show_llm: bool = True) -> str:
        """
        Generate question type breakdown table.
        
        Args:
            comparison_result: Comparison results across attack methods
            show_llm: Whether to show LLM-based results alongside cosine similarity
            
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
        
        if show_llm:
            # Create two separate tables: Cosine Similarity and LLM Factual Similarity
            tables = []
            
            # Table 1: Cosine Similarity by Question Type
            tables.append("### Cosine Similarity by Question Type")
            tables.append("")
            
            tables.append("| Attack Method | " + " | ".join([f"{qt} ↑" for qt in all_question_types]) + " |")
            tables.append("| ------------- | " + " | ".join(["-------" for _ in all_question_types]) + " |")
            
            # Add data rows for cosine similarity
            for result in comparison_result.results:
                row_parts = [result.attack_method]
                
                for question_type in all_question_types:
                    if question_type in result.question_type_results:
                        similarity = result.question_type_results[question_type]
                        row_parts.append(f"{similarity:.3f}")
                    else:
                        row_parts.append("—")
                
                tables.append("| " + " | ".join(row_parts) + " |")
            
            tables.append("")
            tables.append("")
            
            # Table 2: LLM Factual Similarity by Question Type
            tables.append("### LLM Factual Similarity by Question Type")
            tables.append("")
            
            tables.append("| Attack Method | " + " | ".join([f"{qt} ↑" for qt in all_question_types]) + " |")
            tables.append("| ------------- | " + " | ".join(["-------" for _ in all_question_types]) + " |")
            
            # Add data rows for LLM similarity
            for result in comparison_result.results:
                row_parts = [result.attack_method]
                
                for question_type in all_question_types:
                    if (result.question_type_llm_results and 
                        question_type in result.question_type_llm_results):
                        similarity = result.question_type_llm_results[question_type]
                        row_parts.append(f"{similarity:.3f}")
                    else:
                        row_parts.append("N/A")
                
                tables.append("| " + " | ".join(row_parts) + " |")
            
            return "\n".join(tables)
            
        else:
            # Original single table format (cosine similarity only)
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
    
    def generate_edit_type_breakdown_table(self, comparison_result: SemanticComparisonResult, show_llm: bool = True) -> str:
        """
        Generate edit type breakdown table.
        
        Args:
            comparison_result: Comparison results across attack methods
            show_llm: Whether to show LLM-based results alongside cosine similarity
            
        Returns:
            Markdown table string with edit type breakdowns
        """
        if not comparison_result.results:
            return "No results to display."
        
        # Get all edit types across all results
        all_edit_types = set()
        for result in comparison_result.results:
            if result.edit_type_results:
                all_edit_types.update(result.edit_type_results.keys())
        
        all_edit_types = sorted(list(all_edit_types))
        
        if not all_edit_types:
            return "No edit type data found."
        
        if show_llm:
            # Create two separate tables: Cosine Similarity and LLM Factual Similarity
            tables = []
            
            # Table 1: Cosine Similarity by Edit Type
            tables.append("### Cosine Similarity by Edit Type")
            tables.append("")
            
            tables.append("| Attack Method | " + " | ".join([f"{et} ↑" for et in all_edit_types]) + " |")
            tables.append("| ------------- | " + " | ".join(["-------" for _ in all_edit_types]) + " |")
            
            # Add data rows for cosine similarity
            for result in comparison_result.results:
                row_parts = [result.attack_method]
                
                for edit_type in all_edit_types:
                    if (result.edit_type_results and 
                        edit_type in result.edit_type_results):
                        similarity = result.edit_type_results[edit_type]
                        row_parts.append(f"{similarity:.3f}")
                    else:
                        row_parts.append("—")
                
                tables.append("| " + " | ".join(row_parts) + " |")
            
            tables.append("")
            tables.append("")
            
            # Table 2: LLM Factual Similarity by Edit Type
            tables.append("### LLM Factual Similarity by Edit Type")
            tables.append("")
            
            tables.append("| Attack Method | " + " | ".join([f"{et} ↑" for et in all_edit_types]) + " |")
            tables.append("| ------------- | " + " | ".join(["-------" for _ in all_edit_types]) + " |")
            
            # Add data rows for LLM similarity
            for result in comparison_result.results:
                row_parts = [result.attack_method]
                
                for edit_type in all_edit_types:
                    if (result.edit_type_llm_results and 
                        edit_type in result.edit_type_llm_results):
                        similarity = result.edit_type_llm_results[edit_type]
                        row_parts.append(f"{similarity:.3f}")
                    else:
                        row_parts.append("N/A")
                
                tables.append("| " + " | ".join(row_parts) + " |")
            
            return "\n".join(tables)
            
        else:
            # Original single table format (cosine similarity only)
            table_lines = []
            table_lines.append("| Attack Method | " + " | ".join([f"{et} ↑" for et in all_edit_types]) + " |")
            table_lines.append("| ------------- | " + " | ".join(["-------" for _ in all_edit_types]) + " |")
            
            # Add data rows
            for result in comparison_result.results:
                row_parts = [result.attack_method]
                
                for edit_type in all_edit_types:
                    if (result.edit_type_results and 
                        edit_type in result.edit_type_results):
                        similarity = result.edit_type_results[edit_type]
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
        report_lines.append("This analysis includes both cosine similarity (based on sentence embeddings) and LLM-based factual similarity judgments.")
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
        
        # Edit type breakdown
        report_lines.append("## Semantic Preservation by Edit Type")
        report_lines.append("")
        report_lines.append("Breakdown of semantic preservation scores across different edit types (sentiment, authority, formatting, etc.):")
        report_lines.append("")
        
        edit_type_table = self.generate_edit_type_breakdown_table(comparison_result)
        report_lines.append(edit_type_table)
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
    parser.add_argument('--trajectory_dir', type=str, default='/data2/xianglin/A40/llm-as-a-judge-attack/trajectories',
                       help='Directory containing trajectory files')
    parser.add_argument('--data_dir', type=str, default='/data2/xianglin/A40/llm-as-a-judge-attack/data',
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
    parser.add_argument('--disable_llm', action='store_true',
                       help='Disable LLM-based factual similarity detection (use cosine similarity only)')
    parser.add_argument('--llm_model', type=str, default='gpt-5',
                       help='LLM model to use for factual similarity detection (default: gpt-5)')
    parser.add_argument('--llm_max_workers', type=int, default=5,
                       help='Maximum number of concurrent LLM API calls (default: 5)')
    parser.add_argument('--edit_types', action='store_true',
                       help='Include edit type breakdown (Sentiment, Authority, etc.)')
    parser.add_argument('--filter_edit_type', type=str, default=None,
                       help='Filter analysis to specific edit type (e.g., Sentiment, Authority)')
    parser.add_argument('--list_edit_types', action='store_true',
                       help='List available edit types and exit')
    parser.add_argument('--eval_num', type=int, default=None,
                       help='Number of samples to evaluate (random sampling if specified)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible sampling (default: 42)')
    
    args = parser.parse_args()
    
    # Handle list edit types
    if args.list_edit_types:
        print("Available Edit Types:")
        print("=" * 40)
        for i, edit_type in enumerate(EDIT_TYPES, 1):
            print(f"{i:2d}. {edit_type}")
        print("\nUsage examples:")
        print("  --filter_edit_type Sentiment")
        print("  --filter_edit_type 'Markdown Format'")
        sys.exit(0)
    
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
    
    print("🔬 SEMANTIC PRESERVATION ANALYSIS WITH EDIT TYPE CATEGORIZATION")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Judge Type: {args.judge_type}")
    print(f"Trajectory Directory: {args.trajectory_dir}")
    if args.filter_edit_type:
        print(f"Edit Type Filter: {args.filter_edit_type}")
    if args.eval_num:
        print(f"Evaluation Subset: {args.eval_num} samples (random seed: {args.random_seed})")
    print("=" * 80)
    
    try:
        # Initialize analyzer
        use_llm = not args.disable_llm
        analyzer = SemanticPreservationAnalyzer(
            args.trajectory_dir, 
            args.data_dir, 
            use_llm_factual_detection=use_llm,
            llm_model=args.llm_model,
            eval_num=args.eval_num,
            random_seed=args.random_seed
        )
        
        # Perform analysis
        llm_status = "enabled" if use_llm else "disabled"
        eval_info = f"eval_num={args.eval_num}" if args.eval_num else "all samples"
        print(f"\n🚀 Starting semantic preservation analysis (batch_size={args.batch_size}, LLM factual detection: {llm_status}, {eval_info})...")
        if use_llm:
            print(f"Using LLM model: {args.llm_model} with {args.llm_max_workers} concurrent workers")
        if args.eval_num:
            print(f"Random sampling: {args.eval_num} samples (seed: {args.random_seed})")
        
        comparison_result = analyzer.compare_attack_methods(
            attack_configs, 
            batch_size=args.batch_size,
            llm_max_workers=args.llm_max_workers
        )
        
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
            
            # Edit type breakdown
            if args.edit_types:
                print("\n\n📋 EDIT TYPE BREAKDOWN")
                print("=" * 50)
                edit_type_table = analyzer.generate_edit_type_breakdown_table(comparison_result)
                print(edit_type_table) 
                
                output_content += f"\n## Edit Type Breakdown\n\n{edit_type_table}\n"
        
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