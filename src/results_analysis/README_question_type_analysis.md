# Table Generator for Attack Comparison

The Table Generator provides functionality to create formatted comparison tables showing ASR (Attack Success Rate) and SLTR (Score Lift Transfer Rate) metrics across different question categories and attack methods.

## Overview

This module generates tables similar to:

| Attack           | Objective ASR ↑ | Subjective ASR ↑ | Δ     | Objective SLTR ↑ | Subjective SLTR ↑ | Δ     |
| ---------------- | --------------- | ---------------- | ----- | ---------------- | ----------------- | ----- |
| **BITE (ours)**  | **48.7%**       | **75.1%**        | +26.4 | **0.62**         | **0.80**          | +0.18 |
| Random           | 22.3%           | 48.7%            | +26.4 | 0.44             | 0.62              | +0.18 |

## Category Mapping

Based on the 7-category classification system:

### Subjective Categories (1, 4, 5, 6)
- **Computer Science & Programming** (1)
- **Business & Finance** (4)  
- **Writing & Communication** (5)
- **Social & Daily Life** (6)

### Objective Categories (2, 3)
- **Mathematics & Statistics** (2)
- **Science & Engineering** (3)

## Metrics

### ASR (Attack Success Rate)
- **Definition**: Percentage of responses that improved after attack
- **Formula**: `(number of improved responses) / (total responses) × 100`
- **Range**: 0-100%
- **Higher is better**

### SLTR (Score Lift Transfer Rate)
- **Definition**: Average score improvement after attack
- **Formula**: `mean(final_scores - initial_scores)`
- **Range**: Depends on judge scoring scale
- **Higher is better**

## Usage

### Basic Usage

```python
from results_analysis.table_generator import generate_attack_comparison_table

# Define attack configurations
attack_configs = [
    {
        "name": "BITE (ours)",
        "filter_criteria": {
            "strategy": "ucb",
            "judge_backbone": "gpt-4",
            "dataset_name": "AlpacaEval"
        }
    },
    {
        "name": "Random",
        "filter_criteria": {
            "strategy": "random",
            "judge_backbone": "gpt-4", 
            "dataset_name": "AlpacaEval"
        }
    }
]

# Generate table
table = generate_attack_comparison_table(
    trajectory_dir="/path/to/trajectories",
    attack_configs=attack_configs,
    baseline_idx=1  # Use Random as baseline
)

print(table)
```

### Advanced Usage

```python
from results_analysis.table_generator import TableGenerator

# Create generator instance for more control
generator = TableGenerator("/path/to/trajectories")

# Analyze specific attack method
result = generator.analyze_attack_method(
    attack_name="BITE",
    filter_criteria={"strategy": "ucb", "budget": 20}
)

print(f"Objective ASR: {result.objective_asr:.1f}%")
print(f"Subjective ASR: {result.subjective_asr:.1f}%")

# Generate detailed category breakdown
detailed_table = generator.generate_detailed_category_table(attack_configs)
print(detailed_table)
```

### Group By Analysis

```python
from results_analysis.table_generator import generate_grouped_table, generate_multi_group_analysis

# Group by strategy
strategy_table = generate_grouped_table(
    trajectory_dir="/path/to/trajectories",
    group_by="strategy",
    filter_criteria={"judge_backbone": "gpt-4"},
    baseline_group="random"
)
print(strategy_table)

# Group by judge model
judge_table = generate_grouped_table(
    trajectory_dir="/path/to/trajectories", 
    group_by="judge_backbone",
    filter_criteria={"dataset_name": "AlpacaEval"}
)
print(judge_table)

# Multi-group analysis
group_configs = [
    {
        "group_by": "strategy",
        "filter_criteria": {"judge_backbone": "gpt-4"},
        "baseline_group": "random",
        "title": "Strategy Comparison (GPT-4 Judge)"
    },
    {
        "group_by": "dataset_name",
        "filter_criteria": {"strategy": "ucb"},
        "title": "Dataset Performance (BITE Method)"
    }
]

multi_analysis = generate_multi_group_analysis("/path/to/trajectories", group_configs)
print(multi_analysis)
```

### Command Line Usage

```bash
# Basic usage
python examples/generate_comparison_table.py \
    --trajectory_dir /path/to/trajectories \
    --judge_backbone gpt-4 \
    --dataset_name AlpacaEval

# With detailed analysis and output file
python examples/generate_comparison_table.py \
    --trajectory_dir /path/to/trajectories \
    --judge_backbone gpt-4 \
    --dataset_name AlpacaEval \
    --detailed \
    --output_file results.md

# Group by strategy
python examples/generate_comparison_table.py \
    --trajectory_dir /path/to/trajectories \
    --group_by strategy \
    --baseline_group random

# Group by judge model
python examples/generate_comparison_table.py \
    --trajectory_dir /path/to/trajectories \
    --group_by judge_backbone \
    --dataset_name AlpacaEval

# Multi-group analysis
python examples/generate_comparison_table.py \
    --trajectory_dir /path/to/trajectories \
    --multi_group \
    --output_file multi_analysis.md
```

## Configuration

### Attack Configuration Format

Each attack configuration should include:

```python
{
    "name": "Display name for the attack",
    "filter_criteria": {
        "strategy": "attack_strategy",      # e.g., "ucb", "random", "simple_rewrite"
        "judge_backbone": "judge_model",    # e.g., "gpt-4", "gpt-3.5-turbo"
        "dataset_name": "dataset",          # e.g., "AlpacaEval", "MTBench"
        "budget": 20,                       # Optional: filter by budget
        "pool_size": 3,                     # Optional: filter by pool size
        # ... other trajectory metadata fields
    }
}
```

### Filter Criteria Options

You can filter trajectories by any metadata field:

- `strategy`: Attack strategy name
- `judge_type`: "pointwise" or "pairwise"  
- `judge_backbone`: Judge model name
- `dataset_name`: Dataset name
- `response_model_name`: Response model name
- `budget`: Exploration budget
- `pool_size`: Pool size for exploration
- `eval_num`: Number of evaluation samples

## Output Formats

### Main Comparison Table

Shows objective vs subjective metrics with deltas:

```
| Attack           | Objective ASR ↑ | Subjective ASR ↑ | Δ     | Objective SLTR ↑ | Subjective SLTR ↑ | Δ     |
| ---------------- | --------------- | ---------------- | ----- | ---------------- | ----------------- | ----- |
| **BITE (ours)**  | **48.7%**       | **75.1%**        | +26.4 | **0.62**         | **0.80**          | +0.18 |
```

### Detailed Category Table

Shows metrics for all 7 categories individually:

```
| Attack | Computer ASR ↑ | Computer SLTR ↑ | Mathematics ASR ↑ | Mathematics SLTR ↑ | ... |
| ------ | -------------- | --------------- | ----------------- | ------------------ | --- |
| BITE   | 52.3%          | 0.75            | 45.1%             | 0.58               | ... |
```

## Data Requirements

### Trajectory File Format

The generator expects trajectory files with this structure:

```json
{
    "strategy": "ucb",
    "judge_backbone": "gpt-4",
    "dataset_name": "AlpacaEval",
    "trajectories": [
        {
            "question": "Question text",
            "score": 8.5,
            "category": "Computer Science & Programming",
            "history": [
                [7.0, "explanation", "initial_answer", "init"],
                [8.5, "explanation", "final_answer", "strategy"]
            ]
        }
    ]
}
```

### Required Fields

- `trajectories[].category`: Must match one of the 7 CATEGORIES
- `trajectories[].history`: List of [score, explanation, answer, origin] tuples
- `trajectories[].history[0]`: Initial score/answer (first entry)
- `trajectories[].history[-1]`: Final score/answer (last entry)

## Error Handling

The generator handles common issues gracefully:

- **Missing categories**: Warns and returns 0.0 metrics
- **Empty trajectories**: Returns empty results with warning
- **Invalid scores**: Skips invalid entries with logging
- **Missing files**: Raises clear error messages

## Logging

Enable logging to see detailed progress:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

This will show:
- Number of trajectories loaded
- Category filtering results  
- Metric calculation progress
- Warning for missing data

## Integration

The table generator integrates with:

- **TrajectoryLoader**: For loading and filtering trajectory data
- **Category System**: Uses the 7-category classification from `assign_category.py`
- **Analysis Pipeline**: Can be used in broader analysis workflows

## Examples

See `examples/generate_comparison_table.py` for complete usage examples including:

- Basic table generation
- Advanced filtering
- Batch processing multiple configurations
- Output file generation
- Error handling
