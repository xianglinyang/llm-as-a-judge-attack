python -m src.baselines.prompt_injection.naive --filter judge_type=alpaca_eval,llm_agent_name=gemini-2.0-flash-lite,dataset_name=AlpacaEval
python -m src.baselines.prompt_injection.fake_completion --filter judge_type=alpaca_eval,llm_agent_name=gemini-2.0-flash-lite,dataset_name=AlpacaEval
python -m src.baselines.prompt_injection.context_ignore --filter judge_type=alpaca_eval,llm_agent_name=gemini-2.0-flash-lite,dataset_name=AlpacaEval
python -m src.baselines.prompt_injection.escape_chr --filter judge_type=alpaca_eval,llm_agent_name=gemini-2.0-flash-lite,dataset_name=AlpacaEval
python -m src.baselines.prompt_injection.null_model_attack --filter judge_type=alpaca_eval,llm_agent_name=gemini-2.0-flash-lite,dataset_name=AlpacaEval


python -m src.baselines.jailbreak.PAIR.main --filter judge_type=alpaca_eval,llm_agent_name=gemini-2.0-flash-lite,dataset_name=AlpacaEval,judge_backbone=deepseek-r1-0528
python -m src.baselines.jailbreak.TAP.main --filter judge_type=alpaca_eval,dataset_name=AlpacaEval,llm_agent_name=gemini-2.0-flash-lite,judge_backbone=o3-mini
python -m src.baselines.jailbreak.AutoDAN.main --filter judge_type=alpaca_eval,dataset_name=AlpacaEval,llm_agent_name=gemini-2.0-flash-lite,judge_backbone=o3-mini