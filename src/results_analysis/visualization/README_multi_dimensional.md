# Multi-Dimensional Visualization Guide

This guide explains how to visualize performance differences across different **models**, **datasets**, **strategies**, and **configurations** using the enhanced visualization tools.

## 🎯 **Quick Start Examples**

### **Compare by Models**
```bash
# Compare performance across different judge models
python plot_multi_dimensional_comparison.py \
    --metrics_dir /path/to/metrics \
    --group_by judge_model \
    --plot_type both \
    --save_dir plots/

# Specialized model-dataset dashboard
python model_dataset_dashboard.py \
    --metrics_dir /path/to/metrics \
    --save_dir plots/
```

### **Compare by Datasets**
```bash
# Compare performance across different datasets
python plot_multi_dimensional_comparison.py \
    --metrics_dir /path/to/metrics \
    --group_by dataset \
    --show_table

# Focus on specific metric
python plot_multi_dimensional_comparison.py \
    --metrics_dir /path/to/metrics \
    --group_by dataset,strategy \
    --plot_type heatmap
```

### **Compare Model-Dataset Combinations**
```bash
# Heatmap showing all model-dataset-strategy combinations
python plot_multi_dimensional_comparison.py \
    --metrics_dir /path/to/metrics \
    --group_by judge_model,dataset \
    --plot_type heatmap

# Filter to specific configurations
python plot_multi_dimensional_comparison.py \
    --metrics_dir /path/to/metrics \
    --group_by strategy,dataset \
    --filter "judge_model=gemini-2.0-flash,budget=20"
```

## 📊 **Available Visualization Tools**

### **1. Multi-Dimensional Comparison (`plot_multi_dimensional_comparison.py`)**

**Purpose**: Flexible comparison across any combination of dimensions

**Key Features**:
- ✅ **Line plots**: Performance evolution over exploration rounds
- ✅ **Heatmaps**: Performance matrices for 2D comparisons  
- ✅ **Summary tables**: Statistical summaries with filtering
- ✅ **Flexible grouping**: Any combination of dimensions

**Usage**:
```bash
python plot_multi_dimensional_comparison.py \
    --metrics_dir METRICS_DIR \
    --group_by DIMENSION1,DIMENSION2 \
    --plot_type [line|heatmap|both] \
    --filter "key1=value1,key2=value2" \
    --save_dir OUTPUT_DIR \
    --show_table
```

**Available Dimensions**:
- `strategy`: UCB, random, simple_rewrite_holistic, etc.
- `dataset`: AlpacaEval, UltraFeedback, ArenaHard, etc.
- `judge_model`: gemini-2.0-flash, gpt-4, etc.
- `llm_agent`: Model used for response generation
- `response_model`: Model that generated initial responses
- `budget`: Exploration budget (rounds)
- `pool_size`: Pool size parameter
- `alpha`: UCB exploration parameter
- `lambda_reg`: UCB regularization parameter
- `reward_type`: Reward calculation method (relative, absolute)

### **2. Model-Dataset Dashboard (`model_dataset_dashboard.py`)**

**Purpose**: Specialized analysis for model and dataset effects

**Key Features**:
- ✅ **Model-dataset heatmaps**: Performance across all combinations
- ✅ **Strategy effectiveness**: How strategies perform in different contexts
- ✅ **Dataset difficulty ranking**: Which datasets are hardest
- ✅ **Statistical significance testing**: Rigorous comparison
- ✅ **Comprehensive summaries**: Detailed performance statistics

**Usage**:
```bash
python model_dataset_dashboard.py \
    --metrics_dir METRICS_DIR \
    --focus_metric [final_score|improvement|token_efficiency] \
    --save_dir OUTPUT_DIR \
    --skip_stats  # Skip statistical testing for faster run
```

### **3. Enhanced Standard Comparison (`plot_exploration_comparison.py`)**

**Purpose**: Enhanced version of the original comparison tool

**New Features**:
- ✅ **Model/dataset grouping**: `--group_by_model`, `--group_by_dataset`
- ✅ **Filtering**: `--filter_model`, `--filter_dataset`
- ✅ **UCB with warmup**: Separate category for warmup-initialized runs

## 🔍 **Common Analysis Scenarios**

### **Scenario 1: "Which judge model works best?"**

```bash
# Overall comparison
python plot_multi_dimensional_comparison.py \
    --metrics_dir metrics/ \
    --group_by judge_model \
    --show_table

# Per-dataset breakdown
python plot_multi_dimensional_comparison.py \
    --metrics_dir metrics/ \
    --group_by judge_model,dataset \
    --plot_type heatmap
```

**Expected Output**: 
- Line plots showing performance by judge model
- Heatmap showing judge_model × dataset performance
- Summary table with statistical comparisons

### **Scenario 2: "How do datasets differ in difficulty?"**

```bash
# Dataset difficulty analysis
python model_dataset_dashboard.py \
    --metrics_dir metrics/ \
    --focus_metric final_score

# Strategy effectiveness per dataset
python plot_multi_dimensional_comparison.py \
    --metrics_dir metrics/ \
    --group_by dataset,strategy \
    --plot_type line
```

**Expected Output**:
- Dataset difficulty ranking plot
- Strategy performance variations across datasets
- Statistical significance of dataset effects

### **Scenario 3: "Does UCB work better with certain model combinations?"**

```bash
# UCB performance across contexts
python plot_multi_dimensional_comparison.py \
    --metrics_dir metrics/ \
    --group_by judge_model,dataset \
    --filter "strategy=ucb" \
    --plot_type heatmap

# Compare UCB vs Random per context
python model_dataset_dashboard.py \
    --metrics_dir metrics/ \
    --focus_metric improvement
```

**Expected Output**:
- Heatmap showing UCB performance across model-dataset combinations
- Statistical comparison of UCB vs baselines in different contexts

### **Scenario 4: "How do reward types affect performance?"**

```bash
# Reward type comparison
python model_dataset_dashboard.py \
    --metrics_dir metrics/ \
    --focus_metric final_score

# Strategy effectiveness by reward type
python plot_multi_dimensional_comparison.py \
    --metrics_dir metrics/ \
    --group_by reward_type,strategy \
    --plot_type line

# Model performance across reward types
python plot_multi_dimensional_comparison.py \
    --metrics_dir metrics/ \
    --group_by reward_type,judge_model \
    --plot_type heatmap
```

**Expected Output**:
- Box plots comparing relative vs absolute rewards
- Statistical significance testing between reward types
- Strategy performance differences across reward calculation methods

### **Scenario 5: "What's the effect of hyperparameters?"**

```bash
# Alpha parameter effect
python plot_multi_dimensional_comparison.py \
    --metrics_dir metrics/ \
    --group_by alpha,dataset \
    --filter "strategy=ucb" \
    --plot_type line

# Budget vs performance
python plot_multi_dimensional_comparison.py \
    --metrics_dir metrics/ \
    --group_by budget,strategy \
    --plot_type heatmap
```

**Expected Output**:
- Performance curves for different alpha values
- Heatmap showing budget × strategy effectiveness

## 📈 **Interpreting Results**

### **Line Plots**
- **X-axis**: Exploration rounds
- **Y-axis**: Best score so far
- **Colors**: Different groups (models, datasets, strategies)
- **Shaded areas**: Standard deviation across runs
- **Higher lines = better performance**

### **Heatmaps**
- **Cells**: Mean performance for that combination
- **Color intensity**: Performance level (darker = better for most colormaps)
- **Annotations**: Exact performance values
- **Missing cells**: No data for that combination

### **Summary Tables**
- **Runs**: Number of experiments
- **Mean Final Score**: Average final performance
- **Improvement**: Score gain from initial to final
- **Token Efficiency**: Performance improvement per 1K tokens
- **Win Rate**: Percentage of runs achieving top 25% performance

### **Statistical Tests**
- **p-value < 0.05**: Statistically significant difference
- **Effect size**: Magnitude of difference (>0.5 = meaningful)
- **Asterisks**: `*` p<0.05, `**` p<0.01, `***` p<0.001

## 🛠 **Tips for Effective Analysis**

### **1. Start Broad, Then Narrow**
```bash
# 1. Overall strategy comparison
python plot_multi_dimensional_comparison.py --group_by strategy

# 2. Focus on interesting patterns
python plot_multi_dimensional_comparison.py --group_by strategy,dataset

# 3. Deep dive into specific combinations
python model_dataset_dashboard.py --focus_metric improvement
```

### **2. Use Filtering for Clean Comparisons**
```bash
# Compare only pointwise evaluation
--filter "judge_type=pointwise"

# Focus on specific budget
--filter "budget=20"

# Specific model family
--filter "judge_model=gemini-2.0-flash"
```

### **3. Combine Multiple Views**
- **Line plots**: See performance evolution
- **Heatmaps**: Identify best combinations  
- **Tables**: Get exact numbers
- **Statistical tests**: Confirm significance

### **4. Save and Document Results**
```bash
# Always save plots for reports
--save_dir results/analysis_$(date +%Y%m%d)

# Use descriptive filters in filenames
--filter "dataset=AlpacaEval" --save_dir results/alpaca_analysis/
```

## 📁 **Output Files**

When `--save_dir` is specified, the tools generate:

### **Multi-Dimensional Comparison**:
- `grouped_performance_{dimensions}.png/pdf`: Line plots
- `heatmap_{dimensions}.png/pdf`: Heatmap visualization

### **Model-Dataset Dashboard**:
- `model_dataset_heatmap_{metric}.png/pdf`: Model×dataset heatmaps
- `strategy_effectiveness_by_context.png/pdf`: Strategy comparison
- `dataset_difficulty_ranking.png/pdf`: Dataset difficulty analysis
- `model_performance_comparison.png/pdf`: Model comparison matrices

### **Console Output**:
- Performance summary tables
- Statistical significance results
- Data loading statistics
- Filtering information

## 🚀 **Advanced Examples**

### **Complete Analysis Pipeline**
```bash
# 1. Overall landscape
python model_dataset_dashboard.py --metrics_dir metrics/ --save_dir analysis/overview/

# 2. Strategy-specific analysis
python plot_multi_dimensional_comparison.py --metrics_dir metrics/ --group_by strategy,dataset --plot_type both --save_dir analysis/strategies/

# 3. Model-specific analysis  
python plot_multi_dimensional_comparison.py --metrics_dir metrics/ --group_by judge_model,strategy --filter "dataset=AlpacaEval" --save_dir analysis/models/

# 4. Hyperparameter analysis
python plot_multi_dimensional_comparison.py --metrics_dir metrics/ --group_by alpha,lambda_reg --filter "strategy=ucb" --plot_type heatmap --save_dir analysis/hyperparams/
```

### **Publication-Ready Analysis**
```bash
# Statistical rigor + publication plots
python model_dataset_dashboard.py \
    --metrics_dir metrics/ \
    --focus_metric final_score \
    --save_dir paper_figures/ \
    > statistical_analysis.txt

# Clean comparison plots
python plot_multi_dimensional_comparison.py \
    --metrics_dir metrics/ \
    --group_by strategy \
    --plot_type line \
    --save_dir paper_figures/ \
    --show_table > performance_summary.txt
```

This provides a comprehensive framework for understanding how different models, datasets, and configurations affect the performance of your exploration strategies! 📊✨
