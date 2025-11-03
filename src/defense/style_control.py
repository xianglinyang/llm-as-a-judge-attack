'''
This script is used to perform style control for LLM-as-a-Judge attack trajectories. We would like to know the effect of style on the final score. If we remove the style, the score will be higher or lower?
'''
import re
import argparse
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.logging_utils import setup_logging
from src.results_analysis.results_loader.trajectory_loader import load_trajectory_directory, parse_filter_criteria, parse_exclude_criteria

logger = logging.getLogger(__name__)

# -------------------------
# 1. Feature extraction
# -------------------------

def extract_style_features(text):
    """Extract simple style features from an answer string."""
    tokens = text.split()
    n_tokens = len(tokens)
    n_headers = len(re.findall(r"^#+", text, flags=re.MULTILINE))
    n_lists = len(re.findall(r"(^\s*[-*] )|(^\s*\d+\.)", text, flags=re.MULTILINE))
    n_bold = len(re.findall(r"\*\*.*?\*\*", text))
    return {
        "len_tok": n_tokens,
        "n_headers": n_headers,
        "n_lists": n_lists,
        "n_bold": n_bold,
    }

def add_style_features(df, text_col="answer"):
    feats = df[text_col].apply(extract_style_features).apply(pd.Series)
    return pd.concat([df, feats], axis=1)

# -------------------------
# 2. Style-control model
# -------------------------

class StyleControlAbsolute:
    def __init__(self):
        self.scaler = None
        self.model = None
        self.features = None

    def fit(self, df, features=("len_tok","n_headers","n_lists","n_bold"), score_col="score"):
        self.features = list(features)
        X = df[self.features].to_numpy()
        y = df[score_col].to_numpy()
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.model = LinearRegression()
        self.model.fit(Xs, y)
        return self.model

    def style_adjust_scores(self, df, score_col="score"):
        X = df[self.features].to_numpy()
        y = df[score_col].to_numpy()
        Xs = self.scaler.transform(X)
        style_contrib = self.model.predict(Xs) - self.model.intercept_
        return y - style_contrib

    def report_coeffs(self):
        return pd.DataFrame({
            "feature": self.features,
            "coef": self.model.coef_
        })

def trajectories_to_dataframe(trajectories):
    '''Convert trajectories to DataFrame
    Example dataset
    data = [
        {"question": "Q1", "attack": "baseline", "answer": "Answer is 42.", "score": 3.0},
        {"question": "Q1", "attack": "BITE", "answer": "## Final Answer\nThe answer is 42.\n- It is correct.", "score": 4.2},
        {"question": "Q2", "attack": "baseline", "answer": "Paris is the capital of France.", "score": 3.5},
        {"question": "Q2", "attack": "BITE", "answer": "**Answer:** Paris is the capital of France.\n\nIt is well-known.", "score": 4.0},
    ]
    '''
    data = []
    for traj in trajectories:
        for item in traj.trajectories:
            data.append({
                "question": item.question,
                "attack": traj.metadata.strategy,
                "answer": item.final_answer,
                "score": item.final_score
            })
    logger.info(f"Loaded {len(data)} trajectories")
    return pd.DataFrame(data)

def trajectories_to_dataframe_with_one_shot(trajectories):
    '''Convert trajectories to DataFrame with one shot'''
    data = []
    for traj in trajectories:
        for item in traj.trajectories:
            if len(item.history) >1:
                data.append({
                    "question": item.question,
                    "attack": traj.metadata.strategy+"_one_shot",
                    "answer": item.history[1].answer,
                    "score": item.history[1].score
                })
    logger.info(f"Loaded {len(data)} trajectories")
    return pd.DataFrame(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style control for LLM-as-a-Judge attack trajectories")
    parser.add_argument("--directory", type=str,
                       help="Directory containing trajectory files",
                       default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories")
    parser.add_argument("--filter", type=str, 
                       help="Include only files matching criteria (format: 'key1=value1,key2=value2'). "
                            "Example: --filter 'strategy=ucb,dataset_name=AlpacaEval'")
    parser.add_argument("--exclude", type=str, 
                       help="Exclude files matching criteria (format: 'key1=value1,key2=value2'). "
                            "Example: --exclude 'strategy=random,judge_backbone=gpt-3.5'")
    parser.add_argument("--output_dir", type=str, default="./reports",
                       help="Output directory to save results")
    
    args = parser.parse_args()

    setup_logging(task_name="style_control")
    
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

    train_df = pd.concat([ucb_train_df, holistic_rewrite_train_df, random_train_df, holistic_rewrite_one_shot_train_df])
    test_df = pd.concat([ucb_test_df, holistic_rewrite_test_df, random_test_df, holistic_rewrite_one_shot_test_df])

    logging.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # 3. train the style-control model
    # 3.1. Extract style features
    train_df = add_style_features(train_df, text_col="answer")
    test_df = add_style_features(test_df, text_col="answer")

    # 3.2. Fit style-control regression
    sc = StyleControlAbsolute()
    sc.fit(train_df, features=["len_tok","n_headers","n_lists","n_bold"], score_col="score")
    print("Style coefficients:\n", sc.report_coeffs())

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
    
