"""
As reviewer suggested, we further use all the bias features to control the style.
"""
import re
import argparse
import pandas as pd
import logging
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.feature_analysis.feature_extractor import extract_features, get_feature_names

from src.logging_utils import setup_logging
from src.results_analysis.results_loader.trajectory_loader import load_trajectory_directory, parse_filter_criteria, parse_exclude_criteria
from src.defense.style_control import trajectories_to_dataframe_with_one_shot, trajectories_to_dataframe

logger = logging.getLogger(__name__)

# -------------------------
# 2. Style-control model V2
# -------------------------

class StyleControlAbsoluteV2:
    def __init__(self):
        self.scaler = None
        self.model = None
        self.features = None

    def fit(self, df, features=get_feature_names(), score_col="score"):
        self.features = list(features)
        X = df[self.features].to_numpy()
        y = df[score_col].to_numpy()
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        
        self.model = KernelRidge(alpha=0.5, kernel="rbf", gamma=0.01)
        self.model.fit(Xs, y)
        return self.model

    def style_adjust_scores(self, df, score_col="score"):
        X = df[self.features].to_numpy()
        y = df[score_col].to_numpy()
        Xs = self.scaler.transform(X)
        style_contrib = self.model.predict(Xs) - y.mean()
        return y - style_contrib

    def report_coeffs(self):
        return pd.DataFrame({
            "feature": self.features,
            "coef": self.model.dual_coef_
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style control for LLM-as-a-Judge attack trajectories")
    parser.add_argument("--directory", type=str,
                       help="Directory containing trajectory files",
                       default="/data2/xianglin/A40/llm-as-a-judge-attack/trajectories")
    parser.add_argument("--filter", type=str, 
                       help="Include only files matching criteria (format: 'key1=value1,key2=value2'). "
                            "Example: --filter 'strategy=ucb,dataset_name=AlpacaEval'")
    parser.add_argument("--exclude", type=str, 
                       help="Exclude files matching criteria (format: 'key1=value1,key2=value2'). "
                            "Example: --exclude 'strategy=random,judge_backbone=gpt-3.5'")
    parser.add_argument("--output_dir", type=str, default="./reports",
                       help="Output directory to save results")
    
    args = parser.parse_args()

    setup_logging(task_name="style_control_v2")


    # 0. Parse filter and exclude criteria
    general_filter_criteria = parse_filter_criteria(args.filter) if args.filter else {}
    general_exclude_criteria = parse_exclude_criteria(args.exclude) if args.exclude else {}

    # 1. Load dataset of different strategies
    ucb_filter_criteria = parse_filter_criteria("strategy=ucb")
    holistic_rewrite_filter_criteria = parse_filter_criteria("strategy=simple_rewrite_improve")
    random_filter_criteria = parse_filter_criteria("strategy=random")

    ucb_filter_criteria.update(general_filter_criteria)
    holistic_rewrite_filter_criteria.update(general_filter_criteria)
    random_filter_criteria.update(general_filter_criteria)

    logging.info(f"Ucb filter criteria: {ucb_filter_criteria}")
    logging.info(f"Holistic rewrite filter criteria: {holistic_rewrite_filter_criteria}")
    logging.info(f"Random filter criteria: {random_filter_criteria}")

    # 2. load dataset
    ucb_trajectories = load_trajectory_directory(
        directory=args.directory,
        filter_criteria=ucb_filter_criteria,
        exclude_criteria=general_exclude_criteria
    )

    holistic_rewrite_trajectories = load_trajectory_directory(
        directory=args.directory,
        filter_criteria=holistic_rewrite_filter_criteria,
        exclude_criteria=general_exclude_criteria
    )

    random_trajectories = load_trajectory_directory(
        directory=args.directory,
        filter_criteria=random_filter_criteria,
        exclude_criteria=general_exclude_criteria
    )

    logging.info(f"Loaded {len(ucb_trajectories)} ucb trajectories")
    logging.info(f"Loaded {len(holistic_rewrite_trajectories)} holistic rewrite trajectories")
    logging.info(f"Loaded {len(random_trajectories)} random trajectories")

    ucb_df = trajectories_to_dataframe(ucb_trajectories)
    holistic_rewrite_df = trajectories_to_dataframe(holistic_rewrite_trajectories)
    random_df = trajectories_to_dataframe(random_trajectories)
    holistic_rewrite_one_shot_df = trajectories_to_dataframe_with_one_shot(holistic_rewrite_trajectories)

    # 3. split in training and testing
    ucb_train_df, ucb_test_df = train_test_split(ucb_df, test_size=0.2, random_state=42)
    holistic_rewrite_train_df, holistic_rewrite_test_df = train_test_split(holistic_rewrite_df, test_size=0.2, random_state=42)
    random_train_df, random_test_df = train_test_split(random_df, test_size=0.2, random_state=42)
    holistic_rewrite_one_shot_train_df, holistic_rewrite_one_shot_test_df = train_test_split(holistic_rewrite_one_shot_df, test_size=0.2, random_state=42)

    train_df = pd.concat([ucb_train_df, holistic_rewrite_train_df, random_train_df, holistic_rewrite_one_shot_train_df], ignore_index=True)
    test_df = pd.concat([ucb_test_df, holistic_rewrite_test_df, random_test_df, holistic_rewrite_one_shot_test_df], ignore_index=True)

    logging.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # 3. train the style-control model
    # 3.1. Extract style features
    train_features = extract_features(train_df["answer"].tolist())
    test_features = extract_features(test_df["answer"].tolist())

    train_df = pd.concat([train_df, train_features], axis=1)
    test_df = pd.concat([test_df, test_features], axis=1)

    # 3.2. Fit style-control regression
    sc = StyleControlAbsoluteV2()
    sc.fit(train_df, features=get_feature_names(), score_col="score")
    # print("Style coefficients:\n", sc.report_coeffs().sort_values(by="coef", ascending=False))

    # 3.3. Style-adjust scores
    train_df["score_sc"] = sc.style_adjust_scores(train_df, score_col="score")
    test_df["score_sc"] = sc.style_adjust_scores(test_df, score_col="score")

    # 3.4 print the results
    print("="*50)
    print("Style Control Results")
    print("="*50)
    print("Train Average Score:", train_df["score_sc"].mean())
    print("Test Average Score:", test_df["score_sc"].mean())

    # detailed results, ucb
    train_df_grouped = train_df.groupby("attack")
    test_df_grouped = test_df.groupby("attack")
    for attack, group in test_df_grouped:
        print("="*50)
        print(f"Attack: {attack}")
        print("Score before style control:", group["score"].mean())
        print("Score after style control:", group["score_sc"].mean())
    
