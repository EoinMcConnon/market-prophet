"""
GARCH Model Implementation
For volatility forecasting and modeling conditional heteroskedasticity
"""

import numpy as np
import pandas as pd
from arch import arch_model
from typing import Tuple, Dict, Optional


class GARCHForecaster:
    """
    GARCH volatility forecasting
    """
    
    def __init__(self):
        self.model = None
        self.results = None
        self.order = None
        self.vol_type = None
        
    def fit(
        self,
        returns: pd.Series,
        p: int = 1,
        q: int = 1,
        vol: str = 'GARCH',
        dist: str = 'normal',
        mean: str = 'constant'
    ) -> 'GARCHForecaster':
        """
        Fit GARCH model
        
        Parameters:
        -----------
        returns : pd.Series
            Return series (not prices!)
        p : int
            GARCH order (lag of squared residuals)
        q : int
            ARCH order (lag of conditional variance)
        vol : str
            Volatility model: 'GARCH', 'EGARCH', 'TARCH'
        dist : str
            Error distribution: 'normal', 't', 'skewt'
        mean : str
            Mean model: 'constant', 'AR', 'Zero'
            
        Returns:
        --------
        self
        """
        self.order = (p, q)
        self.vol_type = vol
        
        # Scale returns to percentage for better convergence
        returns_scaled = returns * 100
        
        # Create and fit model
        self.model = arch_model(
            returns_scaled,
            p=p,
            q=q,
            vol=vol,
            dist=dist,
            mean=mean
        )
        
        self.results = self.model.fit(disp='off')
        
        return self
    
    def predict_volatility(
        self,
        horizon: int = 1,
        method: str = 'analytic'
    ) -> pd.Series:
        """
        Forecast conditional volatility
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon
        method : str
            Forecasting method: 'analytic' or 'simulation'
            
        Returns:
        --------
        pd.Series
            Forecasted volatility (as percentage)
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get forecasts
        forecasts = self.results.forecast(horizon=horizon, method=method)
        vol_forecast = np.sqrt(forecasts.variance.values[-1, :])
        
        return pd.Series(vol_forecast, name='Volatility Forecast')
    
    def get_conditional_volatility(self) -> pd.Series:
        """Get fitted conditional volatility"""
        if self.results is None:
            raise ValueError("Model not fitted")
        
        return pd.Series(
            self.results.conditional_volatility,
            index=self.results.resid.index,
            name='Conditional Volatility'
        )
    
    def get_standardized_residuals(self) -> pd.Series:
        """Get standardized residuals"""
        if self.results is None:
            raise ValueError("Model not fitted")
        
        return pd.Series(
            self.results.std_resid,
            index=self.results.resid.index,
            name='Standardized Residuals'
        )
    
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
            'Model': f'{self.vol_type}({self.order[0]}, {self.order[1]})',
            'Log Likelihood': self.results.loglikelihood,
            'AIC': self.results.aic,
            'BIC': self.results.bic,
            'Number of Observations': self.results.nobs,
        }
        
        # Add parameter estimates
        params = {}
        for name in self.results.params.index:
            params[name] = {
                'Coefficient': self.results.params[name],
                'Std Error': self.results.std_err[name],
                't-statistic': self.results.tvalues[name],
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
            'Std Error': self.results.std_err,
            't-statistic': self.results.tvalues,
            'p-value': self.results.pvalues,
        })
        
        return coef_table
    
    def diagnostic_tests(self) -> Dict:
        """
        Run diagnostic tests on standardized residuals
        
        Returns:
        --------
        dict
            Test results
        """
        from scipy import stats
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        if self.results is None:
            raise ValueError("Model not fitted")
        
        std_resid = self.results.std_resid
        
        # Ljung-Box test on standardized residuals
        lb_test = acorr_ljungbox(std_resid, lags=10, return_df=True)
        
        # Ljung-Box test on squared standardized residuals (for remaining ARCH effects)
        lb_test_sq = acorr_ljungbox(std_resid**2, lags=10, return_df=True)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(std_resid.dropna())
        
        return {
            'Ljung-Box (Residuals)': {
                'Statistic': lb_test['lb_stat'].iloc[-1],
                'p-value': lb_test['lb_pvalue'].iloc[-1],
                'Conclusion': 'No autocorrelation' if lb_test['lb_pvalue'].iloc[-1] > 0.05 else 'Autocorrelation present'
            },
            'Ljung-Box (Squared Residuals)': {
                'Statistic': lb_test_sq['lb_stat'].iloc[-1],
                'p-value': lb_test_sq['lb_pvalue'].iloc[-1],
                'Conclusion': 'No ARCH effects' if lb_test_sq['lb_pvalue'].iloc[-1] > 0.05 else 'ARCH effects remain'
            },
            'Normality': {
                'Jarque-Bera Statistic': jb_stat,
                'p-value': jb_pvalue,
                'Conclusion': 'Normal' if jb_pvalue > 0.05 else 'Not normal'
            }
        }
    
    def plot_diagnostics(self):
        """
        Plot diagnostic plots
        
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        fig = self.results.plot(annualize='D')
        return fig
    
    @staticmethod
    def test_arch_effects(returns: pd.Series, lags: int = 10) -> Dict:
        """
        Test for ARCH effects using Ljung-Box test on squared returns
        
        Parameters:
        -----------
        returns : pd.Series
            Return series
        lags : int
            Number of lags for test
            
        Returns:
        --------
        dict
            Test results
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        # Test on squared returns
        lb_test = acorr_ljungbox(returns**2, lags=lags, return_df=True)
        
        return {
            'Test': 'Ljung-Box on Squared Returns',
            'Statistic': lb_test['lb_stat'].iloc[-1],
            'p-value': lb_test['lb_pvalue'].iloc[-1],
            'ARCH Effects Present': lb_test['lb_pvalue'].iloc[-1] < 0.05,
            'Conclusion': 'ARCH effects present - GARCH appropriate' if lb_test['lb_pvalue'].iloc[-1] < 0.05 else 'No ARCH effects detected'
        }
    
    def value_at_risk(
        self,
        confidence: float = 0.05,
        horizon: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Parameters:
        -----------
        confidence : float
            Confidence level (e.g., 0.05 for 95% VaR)
        horizon : int
            Forecast horizon
            
        Returns:
        --------
        float
            VaR estimate
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        # Get volatility forecast
        vol_forecast = self.predict_volatility(horizon=horizon)
        
        # Calculate VaR (assuming normal distribution)
        from scipy.stats import norm
        z_score = norm.ppf(confidence)
        var = z_score * vol_forecast.iloc[0]
        
        return var
    
    def expected_shortfall(
        self,
        confidence: float = 0.05,
        horizon: int = 1
    ) -> float:
        """
        Calculate Expected Shortfall (ES) / Conditional VaR
        
        Parameters:
        -----------
        confidence : float
            Confidence level
        horizon : int
            Forecast horizon
            
        Returns:
        --------
        float
            ES estimate
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        # Get volatility forecast
        vol_forecast = self.predict_volatility(horizon=horizon)
        
        # Calculate ES (assuming normal distribution)
        from scipy.stats import norm
        z_score = norm.ppf(confidence)
        es = vol_forecast.iloc[0] * norm.pdf(z_score) / confidence
        
        return es
    
    def annualized_volatility(
        self,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized volatility from latest conditional volatility
        
        Parameters:
        -----------
        periods_per_year : int
            Number of periods per year (252 for daily, 12 for monthly)
            
        Returns:
        --------
        float
            Annualized volatility
        """
        if self.results is None:
            raise ValueError("Model not fitted")
        
        latest_vol = self.results.conditional_volatility.iloc[-1]
        annual_vol = latest_vol * np.sqrt(periods_per_year)
        
        return annual_vol
