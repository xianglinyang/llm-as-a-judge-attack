# Transfer Analysis for LLM-as-a-Judge Attacks

The Transfer Analysis module evaluates how well attacks optimized for one judge model transfer to other judge models. This is crucial for understanding the generalization and robustness of attack methods.

## Overview

Transfer analysis follows this workflow:
1. **Load source trajectories**: Get optimized responses from attacks run on Judge A
2. **Re-evaluate on target**: Score the same responses using Judge B  
3. **Calculate transfer metrics**: Compare effectiveness across judges
4. **Category analysis**: Break down results by question categories

## Key Concepts

### Transfer Workflow
```
Source Judge (A) ──> Optimized Responses ──> Target Judge (B)
     ↓                        ↓                      ↓
Initial Scores           Final Scores          Transfer Scores
     ↓                        ↓                      ↓
Source Metrics  ──────> Transfer Metrics <────── Target Metrics
```

### Metrics

#### Attack Success Rate (ASR)
- **Source ASR**: Success rate on the judge used for optimization
- **Target ASR**: Success rate when responses evaluated on target judge  
- **Transfer ASR**: How many responses that improved on source also improve on target

#### Score Lift Transfer Rate (SLTR)
- **Source SLTR**: Average score improvement on source judge
- **Target SLTR**: Average score improvement on target judge
- **Transfer SLTR**: Ratio of target lift to source lift

#### Transfer Effectiveness
- **Formula**: `Transfer SLTR / Source SLTR`
- **Interpretation**: 
  - 1.0 = Perfect transfer (same effectiveness)
  - > 1.0 = Better on target than source
  - < 1.0 = Degraded transfer performance

## Usage

### Basic Transfer Analysis

```python
from results_analysis.transfer_analysis import analyze_judge_transfer

# Analyze transfer between two judges
result = await analyze_judge_transfer(
    trajectory_dir="/path/to/trajectories",
    source_judge="gpt-4",
    target_judge="gpt-3.5-turbo",
    strategy="ucb",
    dataset_name="AlpacaEval"
)

print(f"Transfer ASR: {result.transfer_asr:.1f}%")
print(f"Transfer Effectiveness: {result.transfer_effectiveness:.2f}")
```

### Advanced Analysis

```python
from results_analysis.transfer_analysis import TransferAnalyzer

# Create analyzer for more control
analyzer = TransferAnalyzer("/path/to/trajectories")

# Analyze with category breakdown
result = await analyzer.analyze_transfer(
    source_judge="gpt-4",
    target_judge="gpt-3.5-turbo", 
    strategy="ucb",
    dataset_name="AlpacaEval",
    judge_type="pointwise"
)

# Access category-specific results
for category, cat_result in result.category_results.items():
    print(f"{category}: Transfer ASR {cat_result.transfer_asr:.1f}%")
```

### Multiple Transfer Comparison

```python
from results_analysis.transfer_analysis import analyze_multiple_transfers

# Compare transfer to multiple targets
judge_pairs = [
    ("gpt-4", "gpt-3.5-turbo"),
    ("gpt-4", "claude-3-sonnet"),
    ("gpt-3.5-turbo", "gpt-4")
]

results = await analyze_multiple_transfers(
    trajectory_dir="/path/to/trajectories",
    judge_pairs=judge_pairs,
    strategy="ucb"
)

# Generate comprehensive report
analyzer = TransferAnalyzer("/path/to/trajectories")
report = analyzer.generate_transfer_report(results)
print(report)
```

### Command Line Usage

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

## Data Requirements

### Trajectory Files
Transfer analysis requires trajectory files with:
- Source judge trajectories containing optimized responses
- Consistent question-response pairs across experiments
- Category assignments for breakdown analysis

### Judge Model Support
- Compatible with all judge types: pointwise, pairwise, etc.
- Supports any judge model available in the system
- Handles different scoring scales automatically

## Output Formats

### TransferResult Object
```python
@dataclass
class TransferResult:
    source_judge: str
    target_judge: str
    source_asr: float         # ASR on source judge
    target_asr: float         # ASR on target judge  
    transfer_asr: float       # Transfer ASR
    source_sltr: float        # SLTR on source judge
    target_sltr: float        # SLTR on target judge
    transfer_sltr: float      # Transfer SLTR
    transfer_effectiveness: float  # Effectiveness ratio
    num_questions: int
    category_results: Dict[str, CategoryTransferResult]
```

### Transfer Report
The generated report includes:

```markdown
# Transfer Analysis Report

## Transfer Performance Summary
| Source Judge | Target Judge | Source ASR | Target ASR | Transfer ASR | Transfer Effectiveness | Questions |
| ------------ | ------------ | ---------- | ---------- | ------------ | --------------------- | --------- |
| gpt-4        | gpt-3.5      | 75.2%      | 68.4%      | 71.8%        | 0.85                  | 100       |

## gpt-4 → gpt-3.5-turbo
- **Source ASR**: 75.2%
- **Target ASR**: 68.4%  
- **Transfer ASR**: 71.8%
- **Transfer Effectiveness**: 0.85 (85.0%)

### Category Breakdown
| Category | Source ASR | Target ASR | Transfer ASR | Transfer SLTR | Questions |
| -------- | ---------- | ---------- | ------------ | ------------- | --------- |
| Math     | 80.1%      | 72.3%      | 76.2%        | 0.421         | 25        |
| Writing  | 70.5%      | 64.8%      | 67.1%        | 0.385         | 30        |
```

## Integration

### With Table Generator
```python
# Combine transfer analysis with table generation
from results_analysis.table_generator import TableGenerator
from results_analysis.transfer_analysis import TransferAnalyzer

# Analyze attack performance
table_gen = TableGenerator("/path/to/trajectories")
attack_table = table_gen.generate_grouped_comparison_table(
    group_by="strategy",
    filter_criteria={"judge_backbone": "gpt-4"}
)

# Analyze transfer performance  
transfer_analyzer = TransferAnalyzer("/path/to/trajectories")
transfer_results = await transfer_analyzer.compare_multiple_transfers([
    ("gpt-4", "gpt-3.5-turbo"),
    ("gpt-4", "claude-3-sonnet")
])
transfer_report = transfer_analyzer.generate_transfer_report(transfer_results)

# Combined analysis
print("Attack Performance:")
print(attack_table)
print("\nTransfer Performance:")
print(transfer_report)
```

### With Trajectory Loader
```python
# Use trajectory loader for custom filtering
from results_analysis.trajectory_loader import TrajectoryLoader

loader = TrajectoryLoader("/path/to/trajectories")
trajectories = loader.load_trajectories()

# Filter for specific conditions
filtered_trajs = loader.filter_trajectories(
    trajectories,
    strategy="ucb",
    budget=20,
    judge_backbone="gpt-4"
)

# Use with transfer analyzer
analyzer = TransferAnalyzer("/path/to/trajectories")
# analyzer will use the same filtering internally
```

## Best Practices

### Experimental Design
1. **Consistent datasets**: Use same questions across source/target evaluations
2. **Multiple seeds**: Run transfer analysis across multiple experimental runs
3. **Category balance**: Ensure representative sampling across question categories
4. **Judge coverage**: Test transfer across diverse judge models

### Interpretation Guidelines
1. **Transfer effectiveness > 0.8**: Good transfer
2. **Transfer effectiveness 0.6-0.8**: Moderate transfer  
3. **Transfer effectiveness < 0.6**: Poor transfer
4. **Category-specific patterns**: Some categories may transfer better than others

### Performance Considerations
1. **Batch evaluation**: Use async batch evaluation for efficiency
2. **Caching**: Consider caching judge evaluations for repeated analyses
3. **Memory usage**: Large trajectory sets may require chunked processing
4. **Rate limiting**: Respect API rate limits for external judge models

## Common Use Cases

### 1. Judge Robustness Evaluation
Assess how robust your attacks are across different judge models:
```python
judge_models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]
for source in judge_models:
    for target in judge_models:
        if source != target:
            result = await analyze_judge_transfer(
                trajectory_dir, source, target, "ucb"
            )
            print(f"{source} → {target}: {result.transfer_effectiveness:.2f}")
```

### 2. Attack Strategy Comparison
Compare how different strategies transfer:
```python
strategies = ["ucb", "random", "simple_rewrite"]
for strategy in strategies:
    result = await analyze_judge_transfer(
        trajectory_dir, "gpt-4", "gpt-3.5-turbo", strategy
    )
    print(f"{strategy}: {result.transfer_effectiveness:.2f}")
```

### 3. Dataset Generalization
Evaluate transfer across different datasets:
```python
datasets = ["AlpacaEval", "MTBench", "ArenaHard"]
for dataset in datasets:
    result = await analyze_judge_transfer(
        trajectory_dir, "gpt-4", "gpt-3.5-turbo", "ucb", dataset
    )
    print(f"{dataset}: {result.transfer_effectiveness:.2f}")
```

### 4. Category-Specific Analysis
Identify which question types transfer best:
```python
result = await analyzer.analyze_transfer("gpt-4", "gpt-3.5-turbo", "ucb")
for category, cat_result in result.category_results.items():
    effectiveness = cat_result.transfer_sltr / cat_result.source_sltr
    print(f"{category}: {effectiveness:.2f}")
```

## Troubleshooting

### Common Issues

1. **No trajectories found**: Check filter criteria and file paths
2. **Judge evaluation errors**: Verify judge model availability and API keys
3. **Inconsistent scoring**: Ensure same judge types across comparisons
4. **Memory errors**: Process large datasets in chunks
5. **Rate limiting**: Add delays between API calls

### Error Handling
The transfer analyzer includes robust error handling:
- Missing trajectories: Warns and skips
- Judge evaluation failures: Logs errors and continues
- Category mismatches: Handles gracefully with warnings
- Async exceptions: Proper cleanup and reporting

### Logging
Enable detailed logging for debugging:
```python
import logging
logging.basicConfig(level=logging.INFO)

# Transfer analyzer will log:
# - Trajectory loading progress
# - Judge evaluation status  
# - Category analysis details
# - Error conditions and warnings
```
