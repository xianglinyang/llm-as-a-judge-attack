
from abc import ABC, abstractmethod

import torch
import os
import time
from typing import List, Dict
import logging
import asyncio

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from openai import OpenAI, AsyncOpenAI
from together import Together
from google import genai
from google.genai import types
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class ModelWrapper(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass

    @abstractmethod
    def batch_invoke(self, prompts: List[str]) -> str:
        pass

# HuggingFace models
class HuggingFaceModel(ModelWrapper):
    def __init__(self, model_name, torch_dtype=torch.bfloat16, device="cuda"):
        super().__init__(model_name)
        self.model_name_or_path=model_name
        self.torch_dtype=torch_dtype
        self.device=device
        self._load_tokenizer()
        self._load_model()

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        # Set padding side to left for decoder-only models
            # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path, torch_dtype=self.torch_dtype, device_map=self.device)
        logger.info("model loaded")

    def invoke(self, prompt, system_prompt: str = None, max_new_tokens=2048, temperature=0.7, verbose=False):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        # 3: Tokenize the chat (This can be combined with the previous step using tokenize=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        # Move the tokenized inputs to the same device the model is on (GPU/CPU)
        inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
        if verbose: logger.info("Tokenized inputs:\n", inputs)
        
        # 4: Generate text from the model
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
        if verbose: logger.info("Generated tokens:\n", outputs)

        # 5: Decode the output back to a string
        decoded_output = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        if verbose: logger.info("Decoded output:\n", decoded_output)
        
        return decoded_output
    
    def batch_invoke(self, prompts: List[str], system_prompt: str = None, max_new_tokens=2048, temperature=0.7, verbose=False):
        responses = list()
        for prompt in prompts:
            response = self.invoke(prompt, system_prompt, max_new_tokens, temperature, verbose)
            responses.append(response)
        return responses

# vllm models
class VLLMModel(ModelWrapper):
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.95):
        '''
        Args:
            model_name: str, the name of the model
            device: str, the device to use
            torch_dtype: torch.dtype, the dtype to use
            tensor_parallel_size: int, the number of GPUs to use
        '''
        super().__init__(model_name)
        self.device = device
        self.torch_dtype = torch_dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        
        logger.info(f"Initializing LLM with model: {model_name}...")
        time_start = time.time()
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True, #  Needed for some models like Mistral, already default in recent vLLM
            dtype=self.torch_dtype, # or "bfloat16" if supported and desired. "auto" by default.
        )
        time_end = time.time()
        logger.info(f"LLM initialization took {time_end - time_start:.2f} seconds.")
        
    def invoke(self, prompt: str, n: int = 1, top_p: float = 0.95, temperature: float = 0.7, max_new_tokens: int = 2048) -> str:
        sampling_params = SamplingParams(
            n=n,  # Number of output sequences to return for each prompt
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output. Adjust as needed.
            # stop=["\n\n", "---"], # Sequences at which to stop generation.
        )
        logger.info(f"Using sampling parameters: {sampling_params}")

        logger.info("\nGenerating responses...")
        start_generation_time = time.time()

        # vLLM can process a list of prompts in a batch very efficiently.
        # The `llm.generate` method takes a list of prompts and sampling parameters.
        prompts_dataset = [prompt]
        outputs = self.llm.generate(prompts_dataset, sampling_params)

        end_generation_time = time.time()
        logger.info(f"Generation for {len(prompts_dataset)} prompts took {end_generation_time - start_generation_time:.2f} seconds.")
        return outputs[0].outputs[0].text.strip()
    
    def batch_invoke(self, prompts: List[str], n: int = 1, top_p: float = 0.95, temperature: float = 0.7, max_new_tokens: int = 2048) -> str:
        sampling_params = SamplingParams(
            n=n,  # Number of output sequences to return for each prompt
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output. Adjust as needed.
            # stop=["\n\n", "---"], # Sequences at which to stop generation.
        )
        logger.info(f"Using sampling parameters: {sampling_params}")

        logger.info("\nGenerating responses...")
        start_generation_time = time.time()

        # vLLM can process a list of prompts in a batch very efficiently.
        # The `llm.generate` method takes a list of prompts and sampling parameters.
        outputs = self.llm.generate(prompts, sampling_params)

        end_generation_time = time.time()
        logger.info(f"Generation for {len(prompts)} prompts took {end_generation_time - start_generation_time:.2f} seconds.")
        
        # Process and display results
        results = [output.outputs[0].text.strip() for output in outputs]
        logger.info("\nProcessing complete.")

        # # If you want to calculate tokens per second:
        # total_prompt_tokens = sum(r.prompt_token_count for r in outputs)
        # total_generated_tokens = sum(r.generated_token_count for r in outputs)
        # total_tokens = total_prompt_tokens + total_generated_tokens
        # generation_time = end_generation_time - start_generation_time
        # if generation_time > 0:
        #     tokens_per_second = total_generated_tokens / generation_time # Often interested in output tokens/sec
        #     print(f"\n--- Performance ---")
        #     print(f"Total prompts: {len(prompts)}")
        #     print(f"Total prompt tokens: {total_prompt_tokens}")
        #     print(f"Total generated tokens: {total_generated_tokens}")
        #     print(f"Total tokens processed (prompt + generated): {total_tokens}")
        #     print(f"Generation time: {generation_time:.2f} seconds")
        #     print(f"Throughput (generated tokens/sec): {tokens_per_second:.2f} tokens/sec")
        #     print(f"Throughput (prompts/sec): {len(prompts) / generation_time:.2f} prompts/sec")
        return results



# API based models

# gpt-4o-2024-05-13, gpt-4-turbo-2024-04-09, gpt-4-0613, gpt-3.5-turbo-0125
class OpenAIModel(ModelWrapper):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()
    
    def invoke(self, prompt: str, system_prompt: str = None, max_new_tokens=2048, temperature=0.7) -> str:
        """Generates model output using OpenAI's API"""
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            n=1,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    
    async def batch_invoke(self, prompts: List[str], system_prompt: str = None, max_new_tokens=2048, temperature=0.7) -> str:

        async def get_completion(prompt_content: str):
            """
            Asynchronously gets a completion from the OpenAI API.
            """
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt_content})
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    n=1,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"An error occurred for prompt '{prompt_content}': {e}")
                return None # Or handle error more gracefully

        """
        Processes a list of prompts concurrently using AsyncOpenAI.
        """
        tasks = [get_completion(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=False) # Set return_exceptions=True to get exceptions instead of None
        return results


class DashScope(ModelWrapper):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    def invoke(self, prompt: str, system_prompt: str = None, max_new_tokens=2048, temperature=0.7) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            n=1,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    
    def batch_invoke(self, prompts: List[str], system_prompt: str = None, max_new_tokens=2048, temperature=0.7) -> str:
        responses = list()
        for prompt in prompts:
            response = self.invoke(prompt, system_prompt, max_new_tokens, temperature)
            responses.append(response)
        return responses


class GeminiModel(ModelWrapper):
    """Wrapper for Google Gemini models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        # genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client = genai.Client()


    def invoke(self, prompt: str, system_prompt: str = None, max_new_tokens=2048, temperature=0.7) -> str:
        """Generates model output using the Gemini API."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_new_tokens,
                temperature=temperature,
                system_instruction=system_prompt,
                # top_k= 2,
                # top_p= 0.5,
                # response_mime_type= 'application/json',
                # stop_sequences= ['\n'],
                # seed=42,
            ),
        )
        return response.text.strip()
    
    async def batch_invoke(self, prompts: List[str], system_prompt: str = None, max_new_tokens=2048, temperature=0.7) -> str:

        async def get_completion(prompt_content: str):
            """
            Asynchronously gets a completion from the Gemini API.
            """
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt_content,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_new_tokens,
                        temperature=temperature,
                        system_instruction=system_prompt,
                    )
                )
                
                return response.text.strip()
            except Exception as e:
                print(f"An error occurred for prompt '{prompt_content}': {e}")
                return None # Or handle error more gracefully

        """
        Processes a list of prompts concurrently using client.aio api
        """
        tasks = [get_completion(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=False) # Set return_exceptions=True to get exceptions instead of None
        return results


class TogetherModel(ModelWrapper):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    def invoke(self, prompt: str, system_prompt: str = None, max_new_tokens=2048, temperature=0.7) -> str:
        """Generates model output using the TogetherAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            )
        return response.choices[0].message.content
    
    def batch_invoke(self, prompts: List[str], system_prompt: str = None, max_new_tokens=2048, temperature=0.7) -> str:
        responses = list()
        for prompt in prompts:
            response = self.invoke(prompt, system_prompt, max_new_tokens, temperature)
            responses.append(response)
        return responses

class ClaudeModel(ModelWrapper):
    """Wrapper for Anthropic Claude models."""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Anthropic()

    def invoke(self, prompt: str, system_prompt: str = None, max_new_tokens=2048, temperature=0.7) -> str:
        """Generates model output using the Anthropic Messages API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        output = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return output.content[0].text.strip()
    
    def batch_invoke(self, prompts: List[str], system_prompt: str = None, max_new_tokens=2048, temperature=0.7) -> str:
        responses = list()
        for prompt in prompts:
            response = self.invoke(prompt, system_prompt, max_new_tokens, temperature)
            responses.append(response)
        return responses


# TODO: fix me
def load_model(model_name: str, use_vllm: bool = False, **kwargs) -> ModelWrapper:
    if "gpt" in model_name:
        available_models = [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o-mini",
            "gpt-4.1-nano"
        ]
        if model_name not in available_models:
            raise ValueError(f"Model {model_name} not implemented!")
        return OpenAIModel(model_name, **kwargs)
    elif "gemini" in model_name:
        available_models = [
            "gemini-2.5-flash-preview-05-20",
            "gemini-2.5-pro-preview-06-05",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
        ]
        if model_name not in available_models:
            raise ValueError(f"Model {model_name} not implemented!")
        return GeminiModel(model_name, **kwargs)
    elif "/" in model_name:
        if use_vllm:
            return VLLMModel(model_name, **kwargs)
        else:
            return HuggingFaceModel(model_name, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not implemented!")


if __name__ == "__main__":
    model_name = "gemini-1.5-flash"
    model = load_model(model_name)
    response = model.invoke("What is the capital of France?")
    print(response)

    # Test vllm model
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model = load_model(model_name, use_vllm=True)
    response = model.invoke("What is the capital of France?")
    print(response)