import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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

# -------------------------
# 3. Per-prompt evaluation
# -------------------------

def per_prompt_eval(df, sc, features, score_col="score", prompt_col="question", attack_col="attack"):
    """
    df must include columns: [question, attack, score, features...]
    sc: fitted StyleControlAbsolute
    """
    # compute style-adjusted scores
    df["score_sc"] = sc.style_adjust_scores(df, score_col=score_col)

    results = []
    for (prompt, attack), g in df.groupby([prompt_col, attack_col]):
        # baseline row: attack == "baseline"
        base = g[g[attack_col]=="baseline"]
        if len(base) == 0:
            continue
        baseline_score = base[score_col].values[0]
        baseline_sc = base["score_sc"].values[0]

        # final row: take last row for this (prompt, attack)
        final = g.iloc[-1]
        final_score = final[score_col]
        final_sc = final["score_sc"]

        lift = final_score - baseline_score
        lift_sc = final_sc - baseline_sc
        success = lift > 0
        success_sc = lift_sc > 0

        results.append({
            "question": prompt,
            "attack": attack,
            "baseline_score": baseline_score,
            "final_score": final_score,
            "baseline_score_sc": baseline_sc,
            "final_score_sc": final_sc,
            "lift": lift,
            "lift_sc": lift_sc,
            "success": success,
            "success_sc": success_sc
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example dataset
    data = [
        {"question": "Q1", "attack": "baseline", "answer": "Answer is 42.", "score": 3.0},
        {"question": "Q1", "attack": "BITE", "answer": "## Final Answer\nThe answer is 42.\n- It is correct.", "score": 4.2},
        {"question": "Q2", "attack": "baseline", "answer": "Paris is the capital of France.", "score": 3.5},
        {"question": "Q2", "attack": "BITE", "answer": "**Answer:** Paris is the capital of France.\n\nIt is well-known.", "score": 4.0},
    ]

    df = pd.DataFrame(data)

    # 1. Extract style features
    df = add_style_features(df, text_col="answer")

    # 2. Fit style-control regression
    sc = StyleControlAbsolute()
    sc.fit(df, features=["len_tok","n_headers","n_lists","n_bold"], score_col="score")
    print("Style coefficients:\n", sc.report_coeffs())

    # 3. Evaluate per prompt (raw vs style-adjusted)
    results = per_prompt_eval(df, sc, features=["len_tok","n_headers","n_lists","n_bold"],
                            score_col="score", prompt_col="question", attack_col="attack")

    print("\nPer-prompt results:\n", results)
