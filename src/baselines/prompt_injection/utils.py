

model2implementation = {
    "gpt-5-nano": "openai/gpt-5-nano",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gpt-5": "openai/gpt-5",
    "o3-mini": "openai/o3-mini",
    "o3": "openai/o3",
    "o4-mini": "openai/o4-mini",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-2.0-flash": "google/gemini-2.0-flash",
    "gemini-2.0-flash-lite": "google/gemini-2.0-flash-lite",
    "gemini-1.5-flash": "google/gemini-1.5-flash",
    "gemini-1.5-flash-8b": "google/gemini-1.5-flash-8b",
    "claude-3-7-sonnet-20250219": "anthropic/claude-3-7-sonnet-20250219",
    "qwen3-235b-a22b-2507": "qwen/qwen3-235b-a22b-2507",
    "deepseek-r1-0528": "deepseek/deepseek-r1-0528",
    "llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",

}

def get_implementation_name(model_name: str) -> str:
    return model2implementation[model_name]