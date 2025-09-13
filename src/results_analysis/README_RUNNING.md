# Main Results

1. best so far
2. win rate per prompt
3. pool mean
4. replacement ratio

5. ci width
6. ucb gap


```bash
python -m src.results_analysis.plot_main_results --report_metric pool_mean
```

```bash
python -m src.results_analysis.win_rate_analysis --compare ucb,simple_rewrite_improve
python -m src.results_analysis.win_rate_analysis --compare ucb,random
```
Then create a table myself.


# Semantic
```bash
# Basic analysis
python -m src.results_analysis.semantic_preservation_analysis --dataset_name AlpacaEval --judge_type pointwise

# Detailed analysis with category breakdown
python -m src.results_analysis.semantic_preservation_analysis --detailed --dataset_name AlpacaEval --judge_type pointwise

# Quick summary only (faster)
python -m src.results_analysis.semantic_preservation_analysis --summary_only --dataset_name AlpacaEval --judge_type pointwise

# Save results to file
python -m src.results_analysis.semantic_preservation_analysis --detailed --output_file semantic_analysis.md --dataset_name AlpacaEval --judge_type pointwise

# Different datasets and judge types
python -m src.results_analysis.semantic_preservation_analysis --dataset_name ArenaHard --judge_type arena_hard_auto

# Question type breakdown only
python -m src.results_analysis.semantic_preservation_analysis --question_types --dataset_name AlpacaEval --judge_type pointwise

# Question types + categories
python -m src.results_analysis.semantic_preservation_analysis --question_types --detailed --dataset_name AlpacaEval --judge_type pointwise

# (use this one) Comprehensive report (everything)
python -m src.results_analysis.semantic_preservation_analysis --comprehensive --dataset_name AlpacaEval --judge_type pointwise

# Save comprehensive analysis
python -m src.results_analysis.semantic_preservation_analysis --comprehensive --output_file semantic_full_report.md --dataset_name AlpacaEval --judge_type pointwise
```



# Question Types

```bash
# Generate question type analysis using data_loader integration
python -m src.results_analysis.question_type_analysis --question_type_analysis --dataset_name AlpacaEval --judge_type pointwise

# Include instruction samples report
python question_type_analysis.py --question_type_analysis --instruction_samples --dataset_name AlpacaEval

# Specify custom data directory
python question_type_analysis.py --question_type_analysis --data_dir /path/to/data
```


# Transfer

```bash
# "qwen/qwen3-235b-a22b-2507"
# "meta-llama/llama-3.3-70b-instruct"
# "deepseek/deepseek-r1-0528"
# "google/gemini-2.5-flash"
# "openai/o3-mini"
# "openai/gpt-5"

# Basic transfer analysis
python -m src.results_analysis.transfer_analysis --source_judge openai/o3-mini --target_judge meta-llama/llama-3.3-70b-instruct --strategy ucb --dataset_name AlpacaEval

# Multiple target judges
python -m src.results_analysis.transfer_analysis.py --source_judge gpt-4 --multiple_targets "gpt-3.5-turbo,claude-3-sonnet" --strategy ucb --output_file ./reports/transfer_report.md
```

Visualization
```bash
python -m src.results_analysis.transfer_heatmap_visualization
```


# Feature Analysis
```bash
python -m src.results_analysis.regression_analyzer --exclude strategy=simple_rewrite_holistic,llm_agent=gpt-4.1-nano,judge_backbone=gpt-5
```