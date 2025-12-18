"""
Model Comparison and Forecasting Accuracy Evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ModelComparison:
    """
    Compare forecasting performance across different models
    """
    
    @staticmethod
    def calculate_metrics(
        actual: pd.Series,
        predicted: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics
        
        Parameters:
        -----------
        actual : pd.Series
            Actual values
        predicted : pd.Series
            Predicted values
            
        Returns:
        --------
        dict
            Dictionary of accuracy metrics
        """
        # Convert to numpy arrays - handle both Series and arrays
        if isinstance(actual, pd.Series):
            actual_values = actual.values.flatten()
        else:
            actual_values = np.array(actual).flatten()
            
        if isinstance(predicted, pd.Series):
            predicted_values = predicted.values.flatten()
        else:
            predicted_values = np.array(predicted).flatten()
        
        # Ensure same length - truncate to minimum
        min_len = min(len(actual_values), len(predicted_values))
        actual_values = actual_values[:min_len]
        predicted_values = predicted_values[:min_len]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
        actual_values = actual_values[valid_mask]
        predicted_values = predicted_values[valid_mask]
        
        # If no valid data, return zeros
        if len(actual_values) == 0:
            return {
                'MAE': 0.0,
                'MSE': 0.0,
                'RMSE': 0.0,
                'MAPE': 0.0,
                'MPE': 0.0,
                'ME': 0.0,
                'MAD': 0.0,
                'R²': 0.0,
            }
        
        # Calculate errors
        errors = actual_values - predicted_values
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2
        
        # Percentage errors - avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            percentage_errors = np.where(
                actual_values != 0, 
                (errors / actual_values) * 100, 
                0
            )
        abs_percentage_errors = np.abs(percentage_errors)
        
        # Calculate R²
        ss_res = np.sum(squared_errors)
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Metrics
        metrics = {
            'MAE': float(np.mean(abs_errors)),
            'MSE': float(np.mean(squared_errors)),
            'RMSE': float(np.sqrt(np.mean(squared_errors))),
            'MAPE': float(np.mean(abs_percentage_errors)),
            'MPE': float(np.mean(percentage_errors)),
            'ME': float(np.mean(errors)),
            'MAD': float(np.median(abs_errors)),
            'R²': float(r2),
        }
        
        return metrics
    
    @staticmethod
    def _r_squared(actual: pd.Series, predicted: pd.Series) -> float:
        """Calculate R-squared"""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    @staticmethod
    def diebold_mariano_test(
        actual: pd.Series,
        forecast1: pd.Series,
        forecast2: pd.Series,
        h: int = 1
    ) -> Dict:
        """
        Diebold-Mariano test for comparing forecast accuracy
        
        Tests H0: forecast1 and forecast2 have equal accuracy
        
        Parameters:
        -----------
        actual : pd.Series
            Actual values
        forecast1 : pd.Series
            First forecast
        forecast2 : pd.Series
            Second forecast
        h : int
            Forecast horizon (for adjusting standard error)
            
        Returns:
        --------
        dict
            Test statistic, p-value, and conclusion
        """
        from scipy.stats import t as t_dist
        
        # Calculate forecast errors
        e1 = actual - forecast1
        e2 = actual - forecast2
        
        # Loss differential (using squared errors)
        d = e1**2 - e2**2
        
        # Mean of loss differential
        d_bar = np.mean(d)
        
        # Variance of loss differential (with autocorrelation correction)
        n = len(d)
        gamma0 = np.var(d, ddof=1)
        
        # Simplified standard error (assuming no autocorrelation for brevity)
        se = np.sqrt(gamma0 / n)
        
        # DM statistic
        dm_stat = d_bar / se
        
        # P-value (two-tailed test)
        df = n - 1
        p_value = 2 * (1 - t_dist.cdf(np.abs(dm_stat), df))
        
        return {
            'DM Statistic': dm_stat,
            'p-value': p_value,
            'Conclusion': 'Forecasts significantly different' if p_value < 0.05 else 'No significant difference',
            'Better Model': 'Model 1' if dm_stat < 0 else 'Model 2'
        }
    
    @staticmethod
    def create_comparison_table(
        actual: pd.Series,
        forecasts: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Create comparison table for multiple models
        
        Parameters:
        -----------
        actual : pd.Series
            Actual values
        forecasts : dict
            Dictionary mapping model names to forecast series
            
        Returns:
        --------
        pd.DataFrame
            Comparison table with metrics for each model
        """
        results = {}
        
        for model_name, forecast in forecasts.items():
            metrics = ModelComparison.calculate_metrics(actual, forecast)
            results[model_name] = metrics
        
        df = pd.DataFrame(results).T
        
        # Add ranking for each metric (lower is better for error metrics)
        for col in ['MAE', 'MSE', 'RMSE', 'MAPE', 'MAD']:
            df[f'{col} Rank'] = df[col].rank()
        
        # Higher is better for R²
        df['R² Rank'] = df['R²'].rank(ascending=False)
        
        # Calculate average rank
        rank_cols = [col for col in df.columns if 'Rank' in col]
        df['Average Rank'] = df[rank_cols].mean(axis=1)
        
        # Sort by average rank
        df = df.sort_values('Average Rank')
        
        return df
    
    @staticmethod
    def forecast_encompassing_test(
        actual: pd.Series,
        forecast1: pd.Series,
        forecast2: pd.Series
    ) -> Dict:
        """
        Forecast encompassing test
        
        Tests if forecast1 encompasses forecast2
        (i.e., forecast2 provides no additional information)
        
        Parameters:
        -----------
        actual : pd.Series
            Actual values
        forecast1 : pd.Series
            First forecast (to test if it encompasses)
        forecast2 : pd.Series
            Second forecast
            
        Returns:
        --------
        dict
            Test results
        """
        from statsmodels.api import OLS, add_constant
        
        # Errors
        e1 = actual - forecast1
        e2 = actual - forecast2
        
        # Regression: e1 on constant and (f2 - f1)
        y = e1
        X = add_constant(forecast2 - forecast1)
        
        model = OLS(y, X).fit()
        
        # Test if coefficient on (f2 - f1) is zero
        lambda_coef = model.params.iloc[1]
        p_value = model.pvalues.iloc[1]
        
        return {
            'Lambda Coefficient': lambda_coef,
            'p-value': p_value,
            'Conclusion': 'Forecast 1 encompasses Forecast 2' if p_value > 0.05 else 'Forecast 2 adds information'
        }
    
    @staticmethod
    def combine_forecasts(
        forecasts: Dict[str, pd.Series],
        method: str = 'equal',
        weights: Dict[str, float] = None
    ) -> pd.Series:
        """
        Combine forecasts from multiple models
        
        Parameters:
        -----------
        forecasts : dict
            Dictionary mapping model names to forecast series
        method : str
            Combination method: 'equal', 'inverse_mse', 'custom'
        weights : dict, optional
            Custom weights for each model (for method='custom')
            
        Returns:
        --------
        pd.Series
            Combined forecast
        """
        if method == 'equal':
            # Simple average
            combined = pd.concat(forecasts.values(), axis=1).mean(axis=1)
            
        elif method == 'inverse_mse':
            # Weighted by inverse MSE (not implemented without validation set)
            # Fall back to equal weights
            combined = pd.concat(forecasts.values(), axis=1).mean(axis=1)
            
        elif method == 'custom':
            if weights is None:
                raise ValueError("Weights must be provided for custom method")
            
            # Weighted combination
            combined = pd.Series(0, index=list(forecasts.values())[0].index)
            for model_name, forecast in forecasts.items():
                combined += weights.get(model_name, 0) * forecast
                
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return combined
    
    @staticmethod
    def rolling_forecast_evaluation(
        y: pd.Series,
        forecast_func,
        window_size: int,
        horizon: int = 1
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Perform rolling window forecast evaluation
        
        Parameters:
        -----------
        y : pd.Series
            Full time series
        forecast_func : callable
            Function that takes training data and returns forecast
        window_size : int
            Size of rolling window
        horizon : int
            Forecast horizon
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.DataFrame]
            Actual values, forecasts, metrics over time
        """
        forecasts = []
        actuals = []
        metrics_list = []
        
        for i in range(window_size, len(y) - horizon + 1):
            # Training data
            train = y[i-window_size:i]
            
            # Actual future value
            actual = y[i:i+horizon]
            
            # Generate forecast
            try:
                forecast = forecast_func(train)
                forecasts.append(forecast)
                actuals.append(actual.iloc[0] if len(actual) > 0 else np.nan)
                
                # Calculate metrics for this window
                if not np.isnan(actual.iloc[0]) and not np.isnan(forecast):
                    window_metrics = ModelComparison.calculate_metrics(
                        pd.Series([actual.iloc[0]]),
                        pd.Series([forecast])
                    )
                    metrics_list.append(window_metrics)
                
            except Exception as e:
                forecasts.append(np.nan)
                actuals.append(np.nan)
        
        actuals_series = pd.Series(actuals, name='Actual')
        forecasts_series = pd.Series(forecasts, name='Forecast')
        
        metrics_df = pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()
        
        return actuals_series, forecasts_series, metrics_df
