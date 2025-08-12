'''
Dataset for evaluation:
1. AlpacaEval
2. ArenaHard
3. MTBench
4. UltraFeedback
5. Dolly
6. Oasst
7. JudgeBench
'''
import json
import os
import asyncio
import argparse
import logging

from src.logging_utils import setup_logging
from src.llm_zoo import load_model
from src.data.data_utils import load_metadata
from src.llm_zoo.api_zoo import get_model_name, is_valid_model
logger = logging.getLogger(__name__)

async def get_response_from_model(save_dir, dataset_name, response_model_implementation_name, use_vllm=False, **kwargs):
    '''
    Get the response from the model and save it to the local directory.
    Args:
        save_dir: str, the directory to save the dataset
        dataset_name: str, the name of the dataset
        response_model_implementation_name: str, the implementation name of the model to get the response
        use_vllm: bool, whether to use vllm to get the response
        **kwargs: dict, the arguments to pass to the load_model function
    '''
    try:
        # Load metadata and extract questions
        metadata = load_metadata(save_dir, dataset_name)
    except Exception as e:
        raise ValueError(f"No metadata found for dataset {dataset_name} in {save_dir}")

    # check if the response file already exists
    response_model_name = get_model_name(response_model_implementation_name)
    save_path = os.path.join(save_dir, dataset_name, f"{response_model_name}.json")
    if os.path.exists(save_path):
        logger.info(f"Response file {save_path} already exists. Skipping response generation.")
        return

        
    questions = [item["instruction"] for item in metadata]
    logger.info(f"Loaded {len(questions)} questions from dataset {dataset_name}")

    response_model = load_model(response_model_implementation_name, use_vllm=use_vllm, **kwargs)

    # Track costs
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    if use_vllm:
        # For vLLM, we don't have cost tracking yet, so we'll estimate
        responses = response_model.batch_invoke(questions)
        logger.info(f"Generated {len(responses)} responses using vLLM")
        logger.info("Note: Cost tracking not available for vLLM models")
    else:
        # For API models, we get CallResult objects
        results = await response_model.batch_invoke(questions, return_cost=True)
        
        # Extract responses and calculate costs
        responses = []
        for i, result in enumerate(results):
            if result is not None:
                responses.append(result.response)
                total_cost += result.cost
                total_input_tokens += result.input_tokens
                total_output_tokens += result.output_tokens
            else:
                responses.append("")  # Handle failed calls
                logger.warning(f"Call {i} failed")
        
        # Log cost information
        logger.info(f"Cost Summary for {dataset_name}:")
        logger.info(f"  Total cost: ${total_cost:.6f}")
        logger.info(f"  Total input tokens: {total_input_tokens:,}")
        logger.info(f"  Total output tokens: {total_output_tokens:,}")
        logger.info(f"  Average cost per call: ${total_cost/len(results):.6f}")
    
    new_dataset = []
    for item, response in zip(metadata, responses):
        item['output'] = response
        new_dataset.append(item.copy())
    
    with open(save_path, "w") as f:
        json.dump(new_dataset, f, indent=4)
    
    return new_dataset

async def main(args):
    setup_logging(task_name="get_response_from_model")

    dataset_name = args.dataset_name
    response_model_implementation_name = args.response_model_name
    use_vllm = args.use_vllm
    tensor_parallel_size = args.tensor_parallel_size
    gpu_memory_utilization = args.gpu_memory_utilization
    data_dir = args.data_dir

    if not is_valid_model(response_model_implementation_name):
        raise ValueError(f"Model {response_model_implementation_name} is not valid!")
    
    logger.info(f"Starting response generation for model: {response_model_implementation_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Using vLLM: {use_vllm}")
    
    try:
        if use_vllm:
            await get_response_from_model(
                data_dir, 
                dataset_name, 
                response_model_implementation_name, 
                use_vllm=True, 
                tensor_parallel_size=tensor_parallel_size, 
                gpu_memory_utilization=gpu_memory_utilization
            )
        else:
            await get_response_from_model(
                data_dir, 
                dataset_name, 
                response_model_implementation_name
            )
        logger.info(f"Response generation for {response_model_implementation_name} on {dataset_name} completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to generate responses: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process")
    parser.add_argument("--response_model_name", type=str, required=True, help="Implementation name of the model to generate responses")
    parser.add_argument("--use_vllm", action="store_true", default=False, help="Whether to use vLLM for inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="GPU memory utilization for vLLM")
    args = parser.parse_args()

    asyncio.run(main(args))




    
    
