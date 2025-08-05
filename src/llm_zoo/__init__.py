'''
A LLM wrapper for all models
'''
from src.llm_zoo.base_model import BaseLLM
from src.llm_zoo.api_base_models import OpenAIModel, OpenRouterModel, GeminiModel, ClaudeModel, DashScopeModel
from src.llm_zoo.code_base_models import VLLMModel, HuggingFaceModel

__all__ = ['load_model', 'OpenAIModel', 'OpenRouterModel', 'GeminiModel', 'ClaudeModel', 'DashScopeModel', 'VLLMModel', 'HuggingFaceModel']


openai_models = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "o4-mini",
]

gemini_models = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]

claude_models = [
    "claude-3-7-sonnet-20250219",
]

openrouter_models = [
    # openai
    "openai/o3-mini",
    "openai/o3-mini-high",
    "openai/o3"
    "openai/o3-pro",
    "openai/o4-mini-high",
    "openai/o4-mini"
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/o1",
    "openai/o1-mini",
    "openai/chatgpt-4o-latest",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "openai/gpt-3.5-turbo",

    # gemini
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash-lite-preview-06-17",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-pro-preview",

    # DeepSeek
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-chat-v3-0324",

    # anthropic
    "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-opus"

    # qwen
    # 3.0
    "qwen/qwen3-30b-a3b-instruct-2507",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen3-235b-a22b-2507",
    # 2.5
    "qwen/qwen-turbo",
    "qwen/qwen-plus",
    "qwen/qwen-max"

    # Meta Llam
    "meta-llama/llama-4-scout",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",

]

dashscope_models = [
    "deepseek-r1",
    "deepseek-v3",
    "deepseek-r1-distill-qwen-7b",
    "deepseek-r1-distill-qwen-14b",
    "deepseek-r1-distill-qwen-32b",
    "deepseek-r1-distill-llama-8b",
    "deepseek-r1-distill-llama-70b"

    # 3.0
    "qwen-plus",
    "qwen-turbo",
    "qwen-max",
    "qwen3-235b-a22b-thinking-2507",
    "qwen3-235b-a22b-instruct-2507",
    "qwen3-30b-a3b-thinking-2507",
    "qwen3-30b-a3b-instruct-2507",
    "qwen3-235b-a22b",
    "qwen3-32b",
    "qwen3-30b-a3b",
    "qwen3-14b",
    "qwen3-8b"
    # 2.5
    "qwq-plus",
    "qwq-32b"
]

# TODO: fix me
def load_model(model_name: str, use_vllm: bool = False, **kwargs) -> BaseLLM:
    if model_name in openai_models:
        return OpenAIModel(model_name, **kwargs)
    elif model_name in gemini_models:
        return GeminiModel(model_name, **kwargs)
    elif model_name in claude_models:
        return ClaudeModel(model_name, **kwargs)
    elif model_name in dashscope_models:
        return DashScopeModel(model_name, **kwargs)
    elif model_name in openrouter_models:
        return OpenRouterModel(model_name, **kwargs)
    elif "/" in model_name:
        if use_vllm:
            return VLLMModel(model_name, **kwargs)
        else:
            return HuggingFaceModel(model_name, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not implemented!")