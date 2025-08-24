'''
This file is used to fit a regression model to the data between the features and the score.

1. fit the regression model
2. get the coefficients and their p-values of the features
3. get the R-squared value of the model

https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
'''

import logging
import pandas as pd
import numpy as np

import statsmodels.api as sm

from src.analysis.feature_extractor import extract_features_for_analysis
from src.logging_utils import setup_logging

logger = logging.getLogger(__name__)

class RegressionModel:
    def __init__(self):
        pass
    
    def _convert_to_dataframe(self, X, y):
        # convert the data to a pandas dataframe if current data is not a pandas dataframe
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
                X_df = pd.DataFrame(X, columns=pd.Index(feature_names))
            else:
                # Handle list or other array-like objects
                X_array = np.array(X)
                feature_names = [f'Feature_{i+1}' for i in range(X_array.shape[1])]
                X_df = pd.DataFrame(X_array, columns=pd.Index(feature_names))
        else:
            X_df = X
        
        if not isinstance(y, pd.Series):
            y_series = pd.Series(y, name='Target')
        else:
            y_series = y
        # show the first 5 rows of the data
        logger.info("Sample X (first 5 rows):")
        logger.info(X_df.head())
        logger.info("\nSample y (first 5 rows):")
        logger.info(y_series.head())

        return X_df, y_series
        
    
    def linear_regression(self, X, y):
        # 1. convert the data to a pandas dataframe if current data is not a pandas dataframe
        X_df, y_series = self._convert_to_dataframe(X, y)
        
        # 2. Clean data to fix dtype errors
        # Drop rows with NaN values in target variable
        mask = ~y_series.isna()
        X_df = X_df[mask]
        y_series = y_series[mask]
        
        # Ensure all columns are numeric
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        
        # Drop columns that are all NaN after conversion
        X_df = X_df.dropna(axis=1, how='all')
        
        # Drop rows with any remaining NaN values
        initial_rows = len(X_df)
        X_df = X_df.dropna()
        y_series = y_series[X_df.index]
        
        if len(X_df) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(X_df)} rows with NaN values")
        
        if len(X_df) == 0:
            raise ValueError("No valid data remaining after cleaning")
        
        # Ensure target variable is numeric
        y_series = pd.to_numeric(y_series, errors='coerce')
        
        logger.info(f"Cleaned data shape: X={X_df.shape}, y={y_series.shape}")
        
        # 3. add a constant column to the data
        # X_with_constant = sm.add_constant(X_df)
        # 4. create and fit the OLS (Ordinary Least Squares) model
        model = sm.OLS(y_series, X_df)
        results = model.fit()
        # 4. print the comprehensive summary
        logger.info("\n--- Statsmodels OLS Regression Results ---")
        logger.info(results.summary())
        # 5. extract the coefficients and their p-values
        # 5. Extract specific information (coefficients and p-values)
        coefficients = results.params
        p_values = results.pvalues

        logger.info("\n--- Coefficients and P-values ---")
        for i, (coef_name, coef_value) in enumerate(coefficients.items()):
            p_value = p_values[coef_name]
            logger.info(f"{coef_name}: Coefficient = {coef_value:.4f}, P-value = {p_value:.4f}")

        # # You can also get other useful statistics:
        # logger.info("\nError:")
        # logger.info(results.resid)
        # logger.info("\nStandard Errors:")
        # logger.info(results.bse)
        # logger.info("\nT-values:")
        # logger.info(results.tvalues)
        return coefficients, p_values
    
    # TODO
    def kernel_ridge_regression(self, X, y):
        pass


def get_feature_model_correlation(data_dir, data_type, dataset_list, reward_type, judge_type, judge_backbone, response_model_name, helper_model_name, baseline_response_model_name, answer_position):

    init_df_list = []
    modified_df_list = []
    init_y_list = []
    modified_y_list = []
    feature_names_list = []

    for dataset_name in dataset_list:
        # 1. extract features
        init_df, modified_df, init_y, modified_y, feature_names = extract_features_for_analysis(data_dir, data_type, dataset_name, judge_type, judge_backbone, response_model_name, helper_model_name, reward_type, baseline_response_model_name, answer_position)

        # 2. collect the data
        init_df_list.append(init_df)
        modified_df_list.append(modified_df)
        init_y_list.append(init_y)
        modified_y_list.append(modified_y)
        feature_names_list.append(feature_names)
    
    # 3. merge the data
    init_df = pd.concat(init_df_list, axis=0)
    modified_df = pd.concat(modified_df_list, axis=0)
    init_y = pd.concat(init_y_list, axis=0)
    modified_y = pd.concat(modified_y_list, axis=0)
    feature_names = feature_names_list[0]

    # if no data available, return None
    if len(init_df) == 0:
        return None, None, None
    
    # 4. get the change difference
    X_change = modified_df - init_df
    y_change = modified_y - init_y
    # return X_change, y_change, feature_names

    # 5. show the first 5 rows of the data
    logger.info("Sample X_change (first 5 rows):")
    logger.info(X_change.head())
    logger.info("Sample y_change (first 5 rows):")
    logger.info(y_change.head())

    # 6. fit the regression model
    regression_model = RegressionModel()
    coefficients, p_values = regression_model.linear_regression(X_change, y_change)
    return coefficients, p_values, feature_names



if __name__ == "__main__":
    setup_logging(task_name="regression")
    
    # 0. load data
    # data_dir = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"
    # data_type = "perturbation"
    # reward_type = None
    data_dir = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories"
    data_type = "trajectory"
    reward_type = "relative"

    dataset_name = "AlpacaEval"
    judge_type = "pointwise"
    judge_backbone = "qwen3-235b-a22b-2507"
    response_model_name = "gpt-4.1-mini"
    helper_model_name = "gpt-4.1-nano"
    baseline_response_model_name = None
    answer_position = None

    coefficients, p_values, feature_names = get_feature_model_correlation(data_dir, data_type, ["AlpacaEval", "ArenaHard", "MTBench"], reward_type, judge_type, judge_backbone, response_model_name, helper_model_name, baseline_response_model_name, answer_position)
