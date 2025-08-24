# Metrics to Report

## Primary Metrics (at fixed budgets)

1. **Best score so far (↑)**: mean ± 95% CI across seeds.

2. **Pool mean score (↑)**: average of current pool at each round.

3. **Per-prompt win-rate (↑)**: % prompts where method's best score > baseline's.

## Ranking Fidelity / Robustness

4. **Stability (↓)**: Variation of best score across seeds.

## Ablation 

5. **Median UCB width (↓)**: shows exploration uncertainty shrinking.

6. **Replacement ratio (↓)**: pool churn per round.

## Analysis

7. **Feature importance** (analysis experiment): standardized OLS coefficients + BH-p; partial R².

---

## Metric Definitions

- **↑** = Higher is better
- **↓** = Lower is better
- **CI** = Confidence Interval
- **ASR** = Attack Success Rate