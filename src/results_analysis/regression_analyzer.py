'''
This is used to analyze the features of the attacked answers.
'''

import argparse
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

from src.results_analysis.results_loader.trajectory_loader import load_trajectory_directory, LoadedTrajectory
from src.feature_analysis.feature_extractor import get_feature_names, extract_features
from src.results_analysis.results_loader.utils import parse_filter_criteria, parse_exclude_criteria

logger = logging.getLogger(__name__)


def get_available_groups(trajectories: List[LoadedTrajectory], group_by: str) -> List[str]:
    """
    Extract unique groups from trajectories based on the specified metadata field.
    
    Args:
        trajectories: List of loaded trajectory objects
        group_by: Metadata field to group by (e.g., 'judge_backbone', 'strategy')
        
    Returns:
        List of unique group values
    """
    groups = set()
    for traj in trajectories:
        # Get the metadata value for the specified field
        metadata_value = getattr(traj.metadata, group_by, None)
        if metadata_value is not None:
            groups.add(str(metadata_value))
    
    return sorted(list(groups))


def get_metadata_from_trajectories(trajectories: List[LoadedTrajectory]) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Extract features and scores from trajectory data.
    
    Args:
        trajectories: List of trajectory objects for a specific group
        
    Returns:
        Tuple of (init_features_df, modified_features_df, init_scores, modified_scores)
    """
    all_init_answers = []
    all_modified_answers = []
    all_init_scores = []
    all_modified_scores = []
    
    for traj in trajectories:
        for item in traj.trajectories:
            # Get initial answer and score
            if item.history:
                init_answer = item.initial_answer
                init_score = item.initial_score
                all_init_answers.append(init_answer)
                all_init_scores.append(init_score)
            
            # Get final (modified) answer and score
            modified_answer = item.final_answer
            modified_score = item.final_score
            all_modified_answers.append(modified_answer)
            all_modified_scores.append(modified_score)
    
    # return all_init_answers, all_modified_answers, np.array(all_init_scores), np.array(all_modified_scores)
    # Extract features from answers
    print(f"Extracting features from {len(all_init_answers)} initial answers...")
    init_features_df = extract_features(all_init_answers)
    
    print(f"Extracting features from {len(all_modified_answers)} modified answers...")
    modified_features_df = extract_features(all_modified_answers)
    
    return init_features_df, modified_features_df, np.array(all_init_scores), np.array(all_modified_scores)


def get_feature_model_correlation(init_df: pd.DataFrame, modified_df: pd.DataFrame, 
                                init_y: np.ndarray, modified_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform regression analysis to find correlations between features and model scores.
    
    Args:
        init_df: DataFrame of features for initial answers
        modified_df: DataFrame of features for modified answers
        init_y: Array of initial scores
        modified_y: Array of modified scores
        
    Returns:
        Tuple of (coefficients, p_values) for each feature
    """
    # Combine initial and modified features
    combined_df = pd.concat([init_df, modified_df], ignore_index=True)
    combined_y = np.concatenate([init_y, modified_y])
    
    # Handle missing values
    combined_df = combined_df.fillna(combined_df.median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined_df)
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(X_scaled, combined_y)
    
    # Calculate p-values for coefficients
    n_samples, n_features = X_scaled.shape
    y_pred = model.predict(X_scaled)
    residuals = combined_y - y_pred
    mse = np.mean(residuals**2)
    
    # Calculate standard errors and t-statistics
    X_with_intercept = np.column_stack([np.ones(n_samples), X_scaled])
    try:
        cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        std_errors = np.sqrt(np.diag(cov_matrix))[1:]  # Exclude intercept
        t_stats = model.coef_ / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_samples - n_features - 1))
    except np.linalg.LinAlgError:
        # Fallback if matrix is singular
        p_values = np.ones(len(model.coef_))
    
    return model.coef_, p_values


def generate_heatmap(coefficients_list: List[np.ndarray], p_values_list: List[np.ndarray], 
                    feature_names: List[str], groups: List[str], 
                    output_dir: str = "./plots"):
    """
    Generate a heatmap visualization of feature correlations across groups.
    
    Args:
        coefficients_list: List of coefficient arrays for each group
        p_values_list: List of p-value arrays for each group
        feature_names: List of feature names
        groups: List of group names
        output_path: Path to save the heatmap
    """
    # Create coefficient matrix
    coeff_matrix = np.array(coefficients_list)
    pval_matrix = np.array(p_values_list)
    
    # Create significance mask (p < 0.05)
    significance_mask = pval_matrix < 0.05
    
    # Create the heatmap
    plt.figure(figsize=(max(12, len(feature_names) * 0.8), max(8, len(groups) * 0.6)))
    
    # Create a custom colormap that highlights significant correlations
    sns.heatmap(coeff_matrix, 
                xticklabels=feature_names,
                yticklabels=groups,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                mask=~significance_mask,  # Only show significant correlations
                cbar_kws={'label': 'Regression Coefficient'},
                annot_kws={'size': 8})
    
    plt.title('Feature-Model Score Correlations\n(Only significant correlations shown, p < 0.05)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Groups', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "feature_correlation_heatmap.pdf"), dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {os.path.join(output_dir, 'feature_correlation_heatmap.pdf')}")
    
    # Also create a version showing all correlations (not just significant ones)
    plt.figure(figsize=(max(12, len(feature_names) * 0.8), max(8, len(groups) * 0.6)))
    
    sns.heatmap(coeff_matrix, 
                xticklabels=feature_names,
                yticklabels=groups,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Regression Coefficient'},
                annot_kws={'size': 8})
    
    plt.title('Feature-Model Score Correlations\n(All correlations shown)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Groups', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the full heatmap
    full_output_path = os.path.join(output_dir, "feature_correlation_heatmap_all.pdf")
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    print(f"Full heatmap saved to: {full_output_path}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and inspect trajectory files")
    parser.add_argument("--directory", type=str,
                       help="Directory containing trajectory files",
                       default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories")
    parser.add_argument("--filter", type=str, 
                       help="Include only files matching criteria (format: 'key1=value1,key2=value2'). "
                            "Example: --filter 'strategy=ucb,dataset_name=AlpacaEval'")
    parser.add_argument("--exclude", type=str, 
                       help="Exclude files matching criteria (format: 'key1=value1,key2=value2'). "
                            "Example: --exclude 'strategy=random,judge_backbone=gpt-3.5'")
    parser.add_argument("--group_by", type=str, 
                       help="Group by this feature",
                       default="judge_backbone")
    args = parser.parse_args()
    
    directory = args.directory
    filter_criteria = args.filter
    exclude_criteria = args.exclude
    group_by = args.group_by

    filter_criteria_parsed = parse_filter_criteria(filter_criteria)
    exclude_criteria_parsed = parse_exclude_criteria(exclude_criteria)
    
    trajectories = load_trajectory_directory(directory, filter_criteria=filter_criteria_parsed, exclude_criteria=exclude_criteria_parsed)
    groups = get_available_groups(trajectories, group_by)

    print("Available groups:")
    for group in groups:
        print(group)
    
    # Get feature names for visualization
    print("Feature names:")
    feature_names = get_feature_names()
    for feature_name in feature_names:
        print(feature_name)
    print("--------------------------------")

    coefficients_list = []
    p_values_list = []
    for group in groups:
        # Filter trajectories for this group
        group_trajectories = [traj for traj in trajectories if getattr(traj.metadata, group_by, None) == group]
        print(f"Processing group '{group}' with {len(group_trajectories)} trajectory files...")
        
        init_df, modified_df, init_y, modified_y = get_metadata_from_trajectories(group_trajectories)
        coefficients, p_values = get_feature_model_correlation(init_df, modified_df, init_y, modified_y)
        coefficients_list.append(coefficients)
        p_values_list.append(p_values)
    
    # visualize the coefficients and p-values
    generate_heatmap(coefficients_list, p_values_list, feature_names, groups)








