"""
ARIMA Model Implementation
For time series forecasting with autoregressive and moving average components
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional


class ARIMAForecaster:
    """
    ARIMA time series forecasting
    """
    
    def __init__(self):
        self.model = None
        self.results = None
        self.order = None
        
    def auto_select_order(
        self,
        y: pd.Series,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        ic: str = 'aic'
    ) -> Tuple[int, int, int]:
        """
        Automatically select ARIMA order using information criteria
        
        Parameters:
        -----------
        y : pd.Series
            Time series data
        max_p : int
            Maximum AR order to test
        max_d : int
            Maximum differencing order to test
        max_q : int
            Maximum MA order to test
        ic : str
            Information criterion ('aic' or 'bic')
            
        Returns:
        --------
        Tuple[int, int, int]
            Selected (p, d, q) order
        """
        best_ic = np.inf
        best_order = (0, 0, 0)
        
        # Determine d using ADF test
        d = 0
        y_diff = y.copy()
        while d < max_d:
            adf_result = adfuller(y_diff.dropna())
            if adf_result[1] < 0.05:  # p-value < 0.05, stationary
                break
            y_diff = y_diff.diff().dropna()
            d += 1
        
        # Grid search over p and q
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(y, order=(p, d, q))
                    results = model.fit()
                    
                    current_ic = results.aic if ic == 'aic' else results.bic
                    
                    if current_ic < best_ic:
                        best_ic = current_ic
                        best_order = (p, d, q)
                        
                except:
                    continue
        
        return best_order
    
    def fit(
        self,
        y: pd.Series,
        order: Optional[Tuple[int, int, int]] = None,
        auto: bool = False
    ) -> 'ARIMAForecaster':
        """
        Fit ARIMA model
        
        Parameters:
        -----------
        y : pd.Series
            Time series data
        order : Tuple[int, int, int], optional
            (p, d, q) order. If None and auto=False, uses (1,1,1)
        auto : bool
            If True, automatically select best order
            
        Returns:
        --------
        self
        """
        if auto:
            self.order = self.auto_select_order(y)
        elif order is not None:
            self.order = order
        else:
            self.order = (1, 1, 1)  # Default
        
        # Fit model
        self.model = ARIMA(y, order=self.order)
        self.results = self.model.fit()
        
        return self
    
    def predict(self, steps: int) -> pd.Series:
        """
        Generate forecasts
        
        Parameters:
        -----------
        steps : int
            Number of steps ahead to forecast
            
        Returns:
        --------
        pd.Series
            Forecasted values
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecasts = self.results.forecast(steps=steps)
        return pd.Series(forecasts, name='Forecast')
    
    def get_fitted_values(self) -> pd.Series:
        """Get fitted values from the model"""
        if self.results is None:
            raise ValueError("Model not fitted")
        
        return pd.Series(self.results.fittedvalues, name='Fitted')
    
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
        
        summary = {
            'Order (p,d,q)': self.order,
            'AIC': self.results.aic,
            'BIC': self.results.bic,
            'HQIC': self.results.hqic,
            'Log Likelihood': self.results.llf,
        }
        
        # Add parameter estimates
        params = {}
        for name, value in self.results.params.items():
            params[name] = {
                'Coefficient': value,
                'Std Error': self.results.bse[name],
                'z-statistic': value / self.results.bse[name],
                'p-value': self.results.pvalues[name]
            }
        summary['Parameters'] = params
        
        return summary
    
    def get_coefficient_table(self) -> pd.DataFrame:
        """Get nicely formatted coefficient table"""
        if self.results is None:
            raise ValueError("Model not fitted")
        
        coef_table = pd.DataFrame({
            'Coefficient': self.results.params,
            'Std Error': self.results.bse,
            'z-statistic': self.results.tvalues,
            'p-value': self.results.pvalues,
        })
        
        return coef_table
    
    def diagnostic_tests(self) -> Dict:
        """
        Run diagnostic tests on residuals
        
        Returns:
        --------
        dict
            Test results including Ljung-Box test
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from scipy import stats
        
        if self.results is None:
            raise ValueError("Model not fitted")
        
        residuals = self.results.resid
        
        # Ljung-Box test for autocorrelation in residuals
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(residuals.dropna())
        
        return {
            'Ljung-Box Test': {
                'Statistic': lb_test['lb_stat'].iloc[-1],
                'p-value': lb_test['lb_pvalue'].iloc[-1],
                'Conclusion': 'No autocorrelation' if lb_test['lb_pvalue'].iloc[-1] > 0.05 else 'Autocorrelation present'
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
        
        # Get forecast with prediction intervals
        forecast_result = self.results.get_forecast(steps=steps)
        forecast_df = forecast_result.summary_frame(alpha=alpha)
        
        forecasts = pd.Series(forecast_df['mean'].values, name='Forecast')
        lower = pd.Series(forecast_df['mean_ci_lower'].values, name='Lower')
        upper = pd.Series(forecast_df['mean_ci_upper'].values, name='Upper')
        
        return forecasts, lower, upper
    
    def plot_diagnostics(self, figsize=(12, 8)):
        """
        Plot diagnostic plots for ARIMA model
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        fig = self.results.plot_diagnostics(figsize=figsize)
        return fig
    
    @staticmethod
    def plot_acf_pacf(y: pd.Series, lags: int = 40, figsize=(12, 5)):
        """
        Plot ACF and PACF for order selection
        
        Parameters:
        -----------
        y : pd.Series
            Time series data
        lags : int
            Number of lags to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        plot_acf(y.dropna(), lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')
        
        plot_pacf(y.dropna(), lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def check_stationarity(y: pd.Series) -> Dict:
        """
        Check stationarity using ADF test
        
        Parameters:
        -----------
        y : pd.Series
            Time series data
            
        Returns:
        --------
        dict
            Test results
        """
        result = adfuller(y.dropna())
        
        return {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Used Lag': result[2],
            'Number of Observations': result[3],
            'Critical Values': result[4],
            'Is Stationary': result[1] < 0.05,
            'Conclusion': 'Stationary' if result[1] < 0.05 else 'Non-stationary (differencing needed)'
        }
    
    @staticmethod
    def difference_series(y: pd.Series, order: int = 1) -> pd.Series:
        """
        Difference the series
        
        Parameters:
        -----------
        y : pd.Series
            Original series
        order : int
            Order of differencing
            
        Returns:
        --------
        pd.Series
            Differenced series
        """
        result = y.copy()
        for _ in range(order):
            result = result.diff()
        return result.dropna()
