# Exploration Results Visualization Tools

This directory contains comprehensive visualization tools for analyzing UCB vs Random vs Baseline exploration strategies.

## Overview

The visualization suite provides tools to:
- **Compare exploration strategies** (UCB vs UCB with warmup vs Random vs Baseline)
- **Analyze UCB-specific metrics** (confidence intervals, gap evolution)
- **Track performance evolution** over exploration rounds
- **Evaluate token efficiency** and pool management
- **Generate comprehensive dashboards** with multiple views

## Tools Available

### 1. `plot_exploration_comparison.py` - Multi-Strategy Comparison
**Purpose**: Compare multiple exploration strategies with detailed analysis.

**Features**:
- Performance evolution plots
- Final performance box plots  
- Token efficiency analysis
- Hyperparameter comparison
- Summary statistics tables

**Usage**:
```bash
# Auto-discover files by strategy
python plot_exploration_comparison.py --metrics_dir /path/to/metrics/

# Specify files manually
python plot_exploration_comparison.py --metrics_files ucb1.json ucb2.json random1.json

# Save plots and show summary
python plot_exploration_comparison.py --metrics_dir /path/to/metrics/ --save_dir ./plots/ --show_table
```

### 2. `plot_strategy_comparison.py` - UCB vs Random Focus
**Purpose**: Clean side-by-side comparison of UCB vs Random strategies.

**Features**:
- 8-panel comparison layout
- Performance gap analysis
- UCB-specific metrics (CI width, gap)
- Statistical summaries
- Clean, publication-ready plots

**Usage**:
```bash
# Auto-find files in directory
python plot_strategy_comparison.py --metrics_dir /path/to/metrics/

# Specify exact files
python plot_strategy_comparison.py --ucb_file ucb_results.json --random_file random_results.json

# Show only statistics (no plots)
python plot_strategy_comparison.py --metrics_dir /path/to/metrics/ --stats_only
```

### 3. `exploration_dashboard.py` - Comprehensive Dashboard
**Purpose**: Interactive dashboard with all metrics and multiple visualizations.

**Features**:
- 4x4 comprehensive dashboard layout
- Performance evolution with confidence bands
- UCB-specific metrics analysis
- Pool quality and replacement efficiency
- Strategy statistics table
- Auto-detection of all strategies

**Usage**:
```bash
# Create full dashboard
python exploration_dashboard.py --metrics_dir /path/to/metrics/

# Specify files manually  
python exploration_dashboard.py --metrics_files file1.json file2.json file3.json

# Summary only (no plots)
python exploration_dashboard.py --metrics_dir /path/to/metrics/ --summary_only
```

### 4. `plot_warmup_metrics.py` - Warmup Analysis
**Purpose**: Analyze LinUCB warmup convergence and phase evolution.

**Features**:
- Confidence interval convergence
- UCB gap evolution  
- Phase highlighting (burn-in vs UCB+ε)
- Hyperparameter comparison
- Early stopping analysis

**Usage**:
```bash
# Plot single warmup run
python plot_warmup_metrics.py --metrics_file warmup_results.json

# Compare multiple warmup runs
python plot_warmup_metrics.py --metrics_dir /path/to/warmup/metrics/ --compare

# Show summary table
python plot_warmup_metrics.py --metrics_dir /path/to/metrics/ --show_table
```

## Supported Metrics

### **Core Performance Metrics** (All Strategies)
- `best_so_far`: Best score achieved so far
- `pool_mean`: Average score in current pool
- `replacement_ratio`: Pool replacement efficiency  
- `lift_per_1k_tokens`: Score improvement per 1000 tokens

### **UCB-Specific Metrics**
- `ci_width`: Confidence interval width (2α√(x^T A^{-1} x))
- `ucb_gap`: Gap between best and second-best UCB scores

### **Warmup Metrics**
- `median_ci`: Median CI width across questions
- `median_gap`: Median UCB gap across questions
- Convergence tracking and early stopping

## File Organization

The tools automatically detect strategy types based on filename patterns:

```
metrics/
├── ucb_results_20240118.json          # UCB strategy
├── random_exploration_20240118.json   # Random strategy  
├── baseline_direct_20240118.json      # Baseline strategy
└── warmup_models/
    └── init_ucb_warmup_20240118.json  # Warmup results
```

**Detection Patterns**:
- **UCB**: `*ucb*.json`, `*UCB*.json` (excluding ucb_with_warmup)
- **UCB with warmup**: `*ucb_with_warmup*.json`
- **Random**: `*random*.json`
- **Holistic Rewrite**: `*holistic*.json`
- **Improve**: `*improve*.json`
- **Warmup**: `*warmup*.json`, `*init_ucb_warmup*.json`

**Note**: UCB with warmup uses pre-trained models for better initialization, while standard UCB starts with cold/fresh models, providing a comparison for understanding the impact of model warmup on UCB performance.

## Data Format

### **Exploration Results Format**:
```json
{
  "strategy": "UCB",
  "alpha": 1.0,
  "lambda_reg": 1.0,
  "dataset_name": "AlpacaEval",
  "metrics": [
    {
      "best_so_far": [0.5, 0.6, 0.7, ...],
      "pool_mean": [0.5, 0.55, 0.6, ...],
      "ci_width": [1.2, 0.8, 0.5, ...],
      "ucb_gap": [0.1, 0.3, 0.6, ...]
    }
  ]
}
```

### **Warmup Results Format**:
```json
{
  "warmup_summary": {
    "median_ci": [1.5, 1.0, 0.5, 0.2],
    "median_gap": [0.1, 0.2, 0.5, 0.8],
    "rounds": 15,
    "alpha": 1.2,
    "lambda_reg": 1.0
  }
}
```

## Plot Interpretation Guide

### **Performance Evolution**
- **Upward trend**: Strategy learning and improving
- **Plateau**: Strategy has converged  
- **UCB vs Random**: UCB should outperform Random over time

### **UCB-Specific Metrics**
- **CI Width**: Should decrease (less uncertainty)
  - High → Model uncertain about arm preferences
  - Low → Model confident about rankings
- **UCB Gap**: Should increase (clearer winner)
  - Small → Arms look similar
  - Large → Clear best arm identified

### **Token Efficiency**
- **Higher values**: Better score improvement per token spent
- **UCB advantage**: Should be more token-efficient than Random

### **Pool Quality**
- **Pool mean**: Average quality of maintained solutions
- **Replacement ratio**: How often pool gets updated with better solutions

## Example Workflows

### **1. Quick UCB vs Random Comparison**
```bash
python plot_strategy_comparison.py --metrics_dir ./results/
```

### **2. Comprehensive Analysis Dashboard**  
```bash
python exploration_dashboard.py --metrics_dir ./results/ --save_dir ./analysis/
```

### **3. Warmup Convergence Analysis**
```bash
python plot_warmup_metrics.py --metrics_dir ./warmup_results/ --compare --show_table
```

### **4. Multi-Strategy Performance Study**
```bash
python plot_exploration_comparison.py --metrics_dir ./all_results/ --save_dir ./paper_plots/ --show_table
```

## Output Files

All tools can save plots in multiple formats:
- **PNG**: High-resolution images (300 DPI)
- **PDF**: Vector graphics for publications  
- **Tables**: Text summaries and statistics

**Saved Files**:
- `strategy_comparison.png/pdf`
- `exploration_dashboard.png/pdf`  
- `warmup_convergence.png/pdf`
- `final_performance_comparison.png/pdf`

## Dependencies

```bash
pip install matplotlib seaborn numpy pandas
```

## Tips for Analysis

1. **Start with dashboard**: Get overall view with `exploration_dashboard.py`
2. **Deep dive**: Use specific tools for detailed analysis
3. **Compare hyperparameters**: Use warmup analysis to tune α and λ  
4. **Publication plots**: Use `--save_dir` for clean, publication-ready figures
5. **Quick stats**: Use `--stats_only` or `--show_table` for numerical summaries

The visualization suite provides everything needed to analyze and compare exploration strategies comprehensively!