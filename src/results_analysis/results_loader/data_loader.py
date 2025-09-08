#!/usr/bin/env python3
"""
Data Loader for LLM-as-a-Judge Attack Dataset

This module provides functionality to load and parse dataset files containing
instructions, responses from different models, and metadata from various datasets
like AlpacaEval, ArenaHard, UltraFeedback, etc.

The data format follows the structure:
- Instructions with metadata (category, question_type, etc.)
- Model responses for each instruction
- Dataset-specific metadata

Usage:
    loader = DataLoader("/path/to/data")
    datasets = loader.load_all_datasets()
    alpaca_data = loader.load_dataset("AlpacaEval")
    model_responses = loader.get_model_responses("AlpacaEval", "gpt-4.1-mini")
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class InstructionMetadata:
    """Metadata for a single instruction."""
    instruction: str
    dataset: str
    category: str
    question_type: Optional[str] = None
    original_category: Optional[str] = None
    original_subcategory: Optional[str] = None
    uid: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstructionMetadata':
        """Create metadata from loaded instruction data."""
        return cls(
            instruction=data.get('instruction', ''),
            dataset=data.get('dataset', 'unknown'),
            category=data.get('category', 'unknown'),
            question_type=data.get('question_type'),
            original_category=data.get('original_category'),
            original_subcategory=data.get('original_subcategory'),
            uid=data.get('uid')
        )


@dataclass
class ModelResponse:
    """A single model response to an instruction."""
    instruction: str
    output: str
    dataset: str
    category: str
    model_name: str
    question_type: Optional[str] = None
    original_category: Optional[str] = None
    original_subcategory: Optional[str] = None
    uid: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], model_name: str) -> 'ModelResponse':
        """Create model response from loaded data."""
        return cls(
            instruction=data.get('instruction', ''),
            output=data.get('output', ''),
            dataset=data.get('dataset', 'unknown'),
            category=data.get('category', 'unknown'),
            model_name=model_name,
            question_type=data.get('question_type'),
            original_category=data.get('original_category'),
            original_subcategory=data.get('original_subcategory'),
            uid=data.get('uid')
        )


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    name: str
    path: str
    instructions: List[InstructionMetadata]
    available_models: Set[str]
    categories: Set[str]
    question_types: Set[str]
    total_instructions: int
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.total_instructions = len(self.instructions)
        self.categories = {inst.category for inst in self.instructions}
        self.question_types = {inst.question_type for inst in self.instructions if inst.question_type}


@dataclass
class LoadedDataset:
    """Complete loaded dataset with instructions and model responses."""
    info: DatasetInfo
    model_responses: Dict[str, List[ModelResponse]]
    
    def get_responses_by_model(self, model_name: str) -> List[ModelResponse]:
        """Get all responses from a specific model."""
        return self.model_responses.get(model_name, [])
    
    def get_responses_by_category(self, category: str) -> Dict[str, List[ModelResponse]]:
        """Get all responses for a specific category, grouped by model."""
        result = defaultdict(list)
        for model_name, responses in self.model_responses.items():
            category_responses = [r for r in responses if r.category == category]
            if category_responses:
                result[model_name] = category_responses
        return dict(result)
    
    def get_instruction_by_index(self, index: int) -> Optional[InstructionMetadata]:
        """Get instruction metadata by index."""
        if 0 <= index < len(self.info.instructions):
            return self.info.instructions[index]
        return None
    
    def get_response_by_instruction_index(self, model_name: str, index: int) -> Optional[ModelResponse]:
        """Get model response for a specific instruction index."""
        responses = self.get_responses_by_model(model_name)
        if 0 <= index < len(responses):
            return responses[index]
        return None


class DataLoader:
    """Main class for loading dataset files."""
    
    def __init__(self, data_dir: str = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"):
        """Initialize the data loader.
        
        Args:
            data_dir: Root directory containing dataset folders
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        self.available_datasets = self._discover_datasets()
        logger.info(f"Discovered {len(self.available_datasets)} datasets: {list(self.available_datasets.keys())}")
    
    def _discover_datasets(self) -> Dict[str, Path]:
        """Discover available datasets in the data directory."""
        datasets = {}
        for item in self.data_dir.iterdir():
            if item.is_dir():
                # Check if directory contains metadata.json or model response files
                metadata_file = item / "metadata.json"
                json_files = list(item.glob("*.json"))
                
                if metadata_file.exists() or json_files:
                    datasets[item.name] = item
                    logger.debug(f"Found dataset: {item.name}")
        
        return datasets
    
    def list_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        return list(self.available_datasets.keys())
    
    def list_models_in_dataset(self, dataset_name: str) -> List[str]:
        """Get list of available models for a specific dataset."""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset_path = self.available_datasets[dataset_name]
        models = []
        
        for json_file in dataset_path.glob("*.json"):
            filename = json_file.stem
            # Skip metadata files and dataset_for_exploration files
            if filename not in ['metadata'] and not filename.startswith('dataset_for_exploration'):
                models.append(filename)
        
        return sorted(models)
    
    def load_instructions(self, dataset_name: str) -> List[InstructionMetadata]:
        """Load instruction metadata for a dataset."""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset_path = self.available_datasets[dataset_name]
        metadata_file = dataset_path / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found for dataset '{dataset_name}'")
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            instructions = []
            for item in data:
                instructions.append(InstructionMetadata.from_dict(item))
            
            logger.info(f"Loaded {len(instructions)} instructions for dataset '{dataset_name}'")
            return instructions
            
        except Exception as e:
            logger.error(f"Error loading instructions for dataset '{dataset_name}': {e}")
            raise
    
    def load_model_responses(self, dataset_name: str, model_name: str) -> List[ModelResponse]:
        """Load model responses for a specific dataset and model."""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset_path = self.available_datasets[dataset_name]
        model_file = dataset_path / f"{model_name}.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        try:
            with open(model_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            responses = []
            for item in data:
                responses.append(ModelResponse.from_dict(item, model_name))
            
            logger.info(f"Loaded {len(responses)} responses for model '{model_name}' in dataset '{dataset_name}'")
            return responses
            
        except Exception as e:
            logger.error(f"Error loading responses for model '{model_name}' in dataset '{dataset_name}': {e}")
            raise
    
    def load_dataset(self, dataset_name: str, models: Optional[List[str]] = None) -> LoadedDataset:
        """Load a complete dataset with instructions and model responses.
        
        Args:
            dataset_name: Name of the dataset to load
            models: List of specific models to load. If None, loads all available models.
        
        Returns:
            LoadedDataset containing instructions and model responses
        """
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        # Load instructions
        instructions = self.load_instructions(dataset_name)
        
        # Get available models
        available_models = set(self.list_models_in_dataset(dataset_name))
        
        # Filter models if specified
        if models is not None:
            models_to_load = set(models) & available_models
            missing_models = set(models) - available_models
            if missing_models:
                logger.warning(f"Models not found in dataset '{dataset_name}': {missing_models}")
        else:
            models_to_load = available_models
        
        # Create dataset info
        dataset_info = DatasetInfo(
            name=dataset_name,
            path=str(self.available_datasets[dataset_name]),
            instructions=instructions,
            available_models=models_to_load,
            categories=set(),  # Will be filled in __post_init__
            question_types=set(),  # Will be filled in __post_init__
            total_instructions=0  # Will be filled in __post_init__
        )
        
        # Load model responses
        model_responses = {}
        for model_name in models_to_load:
            try:
                responses = self.load_model_responses(dataset_name, model_name)
                model_responses[model_name] = responses
            except Exception as e:
                logger.warning(f"Failed to load responses for model '{model_name}': {e}")
        
        return LoadedDataset(
            info=dataset_info,
            model_responses=model_responses
        )
    
    def load_all_datasets(self, models: Optional[List[str]] = None) -> Dict[str, LoadedDataset]:
        """Load all available datasets.
        
        Args:
            models: List of specific models to load for each dataset. If None, loads all available models.
        
        Returns:
            Dictionary mapping dataset names to LoadedDataset objects
        """
        datasets = {}
        for dataset_name in self.available_datasets:
            try:
                datasets[dataset_name] = self.load_dataset(dataset_name, models)
                logger.info(f"Successfully loaded dataset: {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to load dataset '{dataset_name}': {e}")
        
        return datasets
    
    def get_dataset_statistics(self, dataset_name: str) -> Dict[str, Any]:
        """Get statistics for a specific dataset."""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        try:
            instructions = self.load_instructions(dataset_name)
            available_models = self.list_models_in_dataset(dataset_name)
            
            # Count categories and question types
            categories = defaultdict(int)
            question_types = defaultdict(int)
            
            for inst in instructions:
                categories[inst.category] += 1
                if inst.question_type:
                    question_types[inst.question_type] += 1
            
            return {
                'dataset_name': dataset_name,
                'total_instructions': len(instructions),
                'available_models': len(available_models),
                'model_names': available_models,
                'categories': dict(categories),
                'question_types': dict(question_types),
                'path': str(self.available_datasets[dataset_name])
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics for dataset '{dataset_name}': {e}")
            raise
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all available datasets."""
        stats = {}
        for dataset_name in self.available_datasets:
            try:
                stats[dataset_name] = self.get_dataset_statistics(dataset_name)
            except Exception as e:
                logger.warning(f"Failed to get statistics for dataset '{dataset_name}': {e}")
                stats[dataset_name] = {'error': str(e)}
        
        return stats
    
    def search_instructions(self, query: str, dataset_name: Optional[str] = None, 
                          category: Optional[str] = None) -> List[Tuple[str, InstructionMetadata]]:
        """Search for instructions containing a query string.
        
        Args:
            query: Search query string
            dataset_name: Specific dataset to search in (optional)
            category: Specific category to search in (optional)
        
        Returns:
            List of tuples (dataset_name, InstructionMetadata)
        """
        results = []
        datasets_to_search = [dataset_name] if dataset_name else self.available_datasets.keys()
        
        for ds_name in datasets_to_search:
            try:
                instructions = self.load_instructions(ds_name)
                for inst in instructions:
                    # Check category filter
                    if category and inst.category != category:
                        continue
                    
                    # Check query match
                    if query.lower() in inst.instruction.lower():
                        results.append((ds_name, inst))
                        
            except Exception as e:
                logger.warning(f"Error searching in dataset '{ds_name}': {e}")
        
        return results


def main():
    """Example usage of the DataLoader."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize loader
    loader = DataLoader()
    
    # Print available datasets
    print("Available datasets:")
    for dataset in loader.list_datasets():
        print(f"  - {dataset}")
    
    # Get statistics for all datasets
    print("\nDataset statistics:")
    stats = loader.get_all_statistics()
    for dataset_name, stat in stats.items():
        if 'error' not in stat:
            print(f"\n{dataset_name}:")
            print(f"  Instructions: {stat['total_instructions']}")
            print(f"  Models: {stat['available_models']}")
            print(f"  Categories: {len(stat['categories'])}")
            print(f"  Question types: {len(stat['question_types'])}")
    
    # Example: Load a specific dataset
    if 'AlpacaEval' in loader.list_datasets():
        print("\nLoading AlpacaEval dataset...")
        alpaca_dataset = loader.load_dataset('AlpacaEval', models=['gpt-4.1-mini', 'claude-3-opus-20240229'])
        
        print(f"Loaded {len(alpaca_dataset.info.instructions)} instructions")
        print(f"Available models: {alpaca_dataset.info.available_models}")
        
        # Get responses from a specific model
        if 'gpt-4.1-mini' in alpaca_dataset.model_responses:
            gpt4_responses = alpaca_dataset.get_responses_by_model('gpt-4.1-mini')
            print(f"GPT-4.1-mini responses: {len(gpt4_responses)}")
            
            # Show first response
            if gpt4_responses:
                first_response = gpt4_responses[0]
                print(f"\nFirst instruction: {first_response.instruction[:100]}...")
                print(f"Category: {first_response.category}")
                print(f"Response: {first_response.output[:200]}...")
    
    if "ArenaHard" in loader.list_datasets():
        print("\nLoading ArenaHard dataset...")
        arena_hard_dataset = loader.load_dataset('ArenaHard', models=['gpt-4.1-mini', 'claude-3-opus-20240229'])
        print(f"Loaded {len(arena_hard_dataset.info.instructions)} instructions")
        print(f"Available models: {arena_hard_dataset.info.available_models}")
        
        if 'gpt-4.1-mini' in arena_hard_dataset.model_responses:
            gpt4_responses = arena_hard_dataset.get_responses_by_model('gpt-4.1-mini')
            print(f"GPT-4.1-mini responses: {len(gpt4_responses)}")
            
            if gpt4_responses:
                first_response = gpt4_responses[0]
                print(f"\nFirst instruction: {first_response.instruction[:100]}...")
                print(f"Category: {first_response.category}")
                print(f"Response: {first_response.output[:200]}...")


if __name__ == "__main__":
    main()
