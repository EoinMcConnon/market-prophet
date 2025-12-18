"""
OLS (Ordinary Least Squares) Model Implementation
For trend analysis and regression-based forecasting
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Dict, Optional


class OLSForecaster:
    """
    OLS-based time series forecasting with trend analysis
    """
    
    def __init__(self):
        self.model = None
        self.results = None
        self.trend_type = None
        
    def fit(
        self,
        y: pd.Series,
        trend: str = 'linear',
        exog: Optional[pd.DataFrame] = None
    ) -> 'OLSForecaster':
        """
        Fit OLS model with specified trend
        
        Parameters:
        -----------
        y : pd.Series
            Target variable (prices or returns)
        trend : str
            Type of trend: 'linear', 'quadratic', 'exponential'
        exog : pd.DataFrame, optional
            Exogenous variables (e.g., market returns, interest rates)
            
        Returns:
        --------
        self
        """
        self.trend_type = trend
        n = len(y)
        
        # Create time index
        t = np.arange(n)
        
        # Build design matrix based on trend type
        if trend == 'linear':
            X = np.column_stack([np.ones(n), t])
            self.feature_names = ['Intercept', 'Time']
            
        elif trend == 'quadratic':
            X = np.column_stack([np.ones(n), t, t**2])
            self.feature_names = ['Intercept', 'Time', 'Time²']
            
        elif trend == 'exponential':
            # For exponential, we use log(y) if possible
            if (y > 0).all():
                y_log = np.log(y)
                X = np.column_stack([np.ones(n), t])
                self.feature_names = ['Intercept', 'Time']
                y = y_log
            else:
                # Fall back to linear if negative values
                X = np.column_stack([np.ones(n), t])
                self.feature_names = ['Intercept', 'Time']
        
        # Add exogenous variables if provided
        if exog is not None:
            X = np.column_stack([X, exog.values])
            self.feature_names.extend(exog.columns.tolist())
        
        # Fit OLS model
        self.model = sm.OLS(y.values, X)
        self.results = self.model.fit()
        
        return self
    
    def predict(
        self,
        steps: int,
        exog_future: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate forecasts
        
        Parameters:
        -----------
        steps : int
            Number of steps ahead to forecast
        exog_future : pd.DataFrame, optional
            Future values of exogenous variables
            
        Returns:
        --------
        pd.Series
            Forecasted values
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get the last time index from training
        n_train = len(self.results.fittedvalues)
        t_future = np.arange(n_train, n_train + steps)
        
        # Build design matrix for forecasts
        if self.trend_type == 'linear':
            X_future = np.column_stack([np.ones(steps), t_future])
            
        elif self.trend_type == 'quadratic':
            X_future = np.column_stack([np.ones(steps), t_future, t_future**2])
            
        elif self.trend_type == 'exponential':
            X_future = np.column_stack([np.ones(steps), t_future])
        
        # Add exogenous variables if provided
        if exog_future is not None:
            X_future = np.column_stack([X_future, exog_future.values])
        
        # Generate predictions
        forecasts = self.results.predict(X_future)
        
        # If exponential trend, transform back
        if self.trend_type == 'exponential':
            forecasts = np.exp(forecasts)
        
        return pd.Series(forecasts, name='Forecast')
    
    def get_fitted_values(self) -> pd.Series:
        """Get fitted values from the model"""
        if self.results is None:
            raise ValueError("Model not fitted")
        
        fitted = self.results.fittedvalues
        
        # Transform back if exponential
        if self.trend_type == 'exponential':
            fitted = np.exp(fitted)
        
        return pd.Series(fitted, name='Fitted')
    
    def get_residuals(self) -> pd.Series:
        """Get model residuals"""
        if self.results is None:
            raise ValueError("Model not fitted")
        
        return pd.Series(self.results.resid, name='Residuals')
    
    def get_summary(self) -> Dict:
        """
        Get model summary statistics
        
        Returns:
        --------
        dict
            Dictionary with model statistics
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        return {
            'R²': self.results.rsquared,
            'Adjusted R²': self.results.rsquared_adj,
            'F-statistic': self.results.fvalue,
            'F p-value': self.results.f_pvalue,
            'AIC': self.results.aic,
            'BIC': self.results.bic,
            'Coefficients': dict(zip(self.feature_names, self.results.params)),
            'Std Errors': dict(zip(self.feature_names, self.results.bse)),
            't-statistics': dict(zip(self.feature_names, self.results.tvalues)),
            'p-values': dict(zip(self.feature_names, self.results.pvalues))
        }
    
    def get_coefficient_table(self) -> pd.DataFrame:
        """Get nicely formatted coefficient table"""
        if self.results is None:
            raise ValueError("Model not fitted")
        
        coef_table = pd.DataFrame({
            'Coefficient': self.results.params,
            'Std Error': self.results.bse,
            't-statistic': self.results.tvalues,
            'p-value': self.results.pvalues,
            'CI Lower': self.results.conf_int()[0],
            'CI Upper': self.results.conf_int()[1]
        }, index=self.feature_names)
        
        return coef_table
    
    def diagnostic_tests(self) -> Dict:
        """
        Run diagnostic tests on residuals
        
        Returns:
        --------
        dict
            Test results
        """
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.stattools import durbin_watson
        from scipy import stats
        
        if self.results is None:
            raise ValueError("Model not fitted")
        
        residuals = self.results.resid
        
        # Breusch-Pagan test for heteroskedasticity
        bp_test = het_breuschpagan(residuals, self.model.exog)
        
        # Durbin-Watson for autocorrelation
        dw_stat = durbin_watson(residuals)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        
        return {
            'Heteroskedasticity': {
                'Breusch-Pagan Statistic': bp_test[0],
                'p-value': bp_test[1],
                'Conclusion': 'Homoskedastic' if bp_test[1] > 0.05 else 'Heteroskedastic'
            },
            'Autocorrelation': {
                'Durbin-Watson': dw_stat,
                'Conclusion': 'No autocorrelation' if 1.5 < dw_stat < 2.5 else 'Autocorrelation present'
            },
            'Normality': {
                'Jarque-Bera Statistic': jb_stat,
                'p-value': jb_pvalue,
                'Conclusion': 'Normal' if jb_pvalue > 0.05 else 'Not normal'
            }
        }
    
    def predict_interval(
        self,
        steps: int,
        alpha: float = 0.05
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate prediction intervals
        
        Parameters:
        -----------
        steps : int
            Forecast horizon
        alpha : float
            Significance level (default 0.05 for 95% CI)
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            Point forecast, lower bound, upper bound
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        # Get point forecasts
        forecasts = self.predict(steps)
        
        # Calculate prediction intervals
        n_train = len(self.results.fittedvalues)
        t_future = np.arange(n_train, n_train + steps)
        
        if self.trend_type == 'linear':
            X_future = np.column_stack([np.ones(steps), t_future])
        elif self.trend_type == 'quadratic':
            X_future = np.column_stack([np.ones(steps), t_future, t_future**2])
        else:
            X_future = np.column_stack([np.ones(steps), t_future])
        
        # Get prediction with standard errors
        predictions = self.results.get_prediction(X_future)
        pred_summary = predictions.summary_frame(alpha=alpha)
        
        lower = pred_summary['obs_ci_lower']
        upper = pred_summary['obs_ci_upper']
        
        # Transform back if exponential
        if self.trend_type == 'exponential':
            lower = np.exp(lower)
            upper = np.exp(upper)
        
        return forecasts, pd.Series(lower.values), pd.Series(upper.values)
