
from abc import ABC, abstractmethod

import torch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from openai import OpenAI
from together import Together
import google.generativeai as genai
from anthropic import Anthropic

class ModelWrapper(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def invoke(self, prompt: str) -> str:
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
        print("model loaded")

    def invoke(self, prompt, max_new_tokens=2048, temperature=0.1, verbose=False):
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        # 3: Tokenize the chat (This can be combined with the previous step using tokenize=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        # Move the tokenized inputs to the same device the model is on (GPU/CPU)
        inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
        if verbose: print("Tokenized inputs:\n", inputs)
        
        # 4: Generate text from the model
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
        if verbose: print("Generated tokens:\n", outputs)

        # 5: Decode the output back to a string
        decoded_output = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        if verbose: print("Decoded output:\n", decoded_output)
        
        return decoded_output


# vllm models
class VLLMModel(ModelWrapper):
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16, device_num: int = 1):
        super().__init__(model_name)
        # TODO: add vllm
        pass
        
    def invoke(self, prompt: str, max_new_tokens: int = 2048) -> str:
        pass


# API based models

# gpt-4o-2024-05-13, gpt-4-turbo-2024-04-09, gpt-4-0613, gpt-3.5-turbo-0125
class GPT(ModelWrapper):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI()
    
    def invoke(self, prompt: str, max_new_tokens=2048, temperature=0.7) -> str:
        """Generates model output using OpenAI's API"""
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            n=1,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


class DashScope(ModelWrapper):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    def invoke(self, prompt: str, max_new_tokens=2048, temperature=0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            n=1,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


class GeminiModel(ModelWrapper):
    """Wrapper for Google Gemini models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client = genai.GenerativeModel(self.model_name)


    def invoke(self, prompt: str, max_new_tokens=2048, temperature=0.7) -> str:
        """Generates model output using the Gemini API."""
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens
        )
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config,
        )

        return response.text.strip()


class TogetherModel(ModelWrapper):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    def invoke(self, prompt: str, max_new_tokens=2048, temperature=0.7) -> str:
        """Generates model output using the TogetherAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            )
        return response.choices[0].message.content


class ClaudeModel(ModelWrapper):
    """Wrapper for Anthropic Claude models."""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Anthropic()

    def invoke(self, prompt: str, max_new_tokens=2048, temperature=0.7) -> str:
        """Generates model output using the Anthropic Messages API."""

        message = self.client.messages.create(
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
        return message.content[0].text.strip()


# TODO: fix me
def load_model(model_name: str) -> ModelWrapper:
    if "gpt" in model_name:
        available_models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4o-2024-05-13", "gpt-4-turbo-2024-04-09", "gpt-4-0613", "gpt-3.5-turbo-0125"]
        if model_name not in available_models:
            raise ValueError(f"Model {model_name} not implemented!")
        return GPT(model_name)
    elif "gemini" in model_name:
        available_models = ["gemini-1.5-pro", "gemini-1.5-flash"]
        if model_name not in available_models:
            raise ValueError(f"Model {model_name} not implemented!")
        return GeminiModel(model_name)
    elif "/" in model_name:
        return HuggingFaceModel(model_name)
    else:
        raise ValueError(f"Model {model_name} not implemented!")


if __name__ == "__main__":
    model_name = "gemini-1.5-flash"
    model = load_model(model_name)
    response = model.invoke("What is the capital of France?")
    print(response)