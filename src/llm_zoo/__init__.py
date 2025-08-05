'''
A LLM wrapper for all models
'''
from src.llm_zoo.base_model import BaseLLM
from src.llm_zoo.api_base_models import OpenAIModel, OpenRouterModel, GeminiModel, ClaudeModel
from src.llm_zoo.code_base_models import VLLMModel, HuggingFaceModel

__all__ = ['load_model', 'OpenAIModel', 'OpenRouterModel', 'GeminiModel', 'ClaudeModel', 'VLLMModel', 'HuggingFaceModel']


# TODO: fix me
def load_model(model_name: str, use_vllm: bool = False, **kwargs) -> BaseLLM:
    if model_name in [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o-mini",
        "gpt-4.1-nano",
        "o4-mini", # o4-mini-2025-04-16
    ]:
        return OpenAIModel(model_name, **kwargs)
    elif model_name in [
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
    ]:
        return GeminiModel(model_name, **kwargs)
    elif model_name in [
        "claude-3-7-sonnet-20250219",
    ]:
        return ClaudeModel(model_name, **kwargs)
    elif model_name in [
        "openrouter/deepseek-r1-distill-qwen-32b",
        "openai/gpt-4.1-mini",
        "openai/gpt-4o-mini",
        "openai/gpt-4.1-nano",
        "openai/o4-mini", # o4-mini-2025-04-16
    ]:
        return OpenRouterModel(model_name, **kwargs)
    elif "/" in model_name:
        if use_vllm:
            return VLLMModel(model_name, **kwargs)
        else:
            return HuggingFaceModel(model_name, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not implemented!")