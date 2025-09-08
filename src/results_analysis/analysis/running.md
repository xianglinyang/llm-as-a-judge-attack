# Semantic
```bash
# Basic analysis
python src/results_analysis/analysis/semantic_preservation_analysis.py --dataset_name AlpacaEval --judge_type pointwise

# Detailed analysis with category breakdown
python src/results_analysis/analysis/semantic_preservation_analysis.py --detailed --dataset_name AlpacaEval --judge_type pointwise

# Quick summary only (faster)
python src/results_analysis/analysis/semantic_preservation_analysis.py --summary_only --dataset_name AlpacaEval --judge_type pointwise

# Save results to file
python src/results_analysis/analysis/semantic_preservation_analysis.py --detailed --output_file semantic_analysis.md --dataset_name AlpacaEval --judge_type pointwise

# Different datasets and judge types
python src/results_analysis/analysis/semantic_preservation_analysis.py --dataset_name ArenaHard --judge_type arena_hard_auto

# Question type breakdown only
python -m src.results_analysis.analysis.semantic_preservation_analysis --question_types --dataset_name AlpacaEval --judge_type pointwise

# Question types + categories
python src/results_analysis/analysis/semantic_preservation_analysis.py --question_types --detailed --dataset_name AlpacaEval --judge_type pointwise

# Comprehensive report (everything)
python src/results_analysis/analysis/semantic_preservation_analysis.py --comprehensive --dataset_name AlpacaEval --judge_type pointwise

# Save comprehensive analysis
python src/results_analysis/analysis/semantic_preservation_analysis.py --comprehensive --output_file semantic_full_report.md --dataset_name AlpacaEval --judge_type pointwise
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
# Basic transfer analysis
python src/results_analysis/transfer_analysis.py \
    --trajectory_dir /path/to/trajectories \
    --source_judge gpt-4 \
    --target_judge gpt-3.5-turbo \
    --strategy ucb \
    --dataset_name AlpacaEval

# Multiple target judges
python src/results_analysis/transfer_analysis.py \
    --trajectory_dir /path/to/trajectories \
    --source_judge gpt-4 \
    --multiple_targets "gpt-3.5-turbo,claude-3-sonnet" \
    --strategy ucb \
    --output_file transfer_report.md

# Cross-dataset analysis
python src/results_analysis/transfer_analysis.py \
    --trajectory_dir /path/to/trajectories \
    --source_judge gpt-4 \
    --target_judge gpt-3.5-turbo \
    --strategy ucb \
    --dataset_name MTBench
```


# 