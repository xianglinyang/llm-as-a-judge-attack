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
        # 2. add a constant column to the data
        X_with_constant = sm.add_constant(X_df)
        # 3. create and fit the OLS (Ordinary Least Squares) model
        model = sm.OLS(y_series, X_with_constant)
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

        # You can also get other useful statistics:
        logger.info("\nStandard Errors:")
        logger.info(results.bse)
        logger.info("\nT-values:")
        logger.info(results.tvalues)
    
    # TODO
    def kernel_ridge_regression(self, X, y):
        pass

   


if __name__ == "__main__":
    # Two settings
    # directly fit the score as the target
    # fit the change difference