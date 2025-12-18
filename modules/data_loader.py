"""
Data Loading and Preprocessing Module
Handles data fetching from Yahoo Finance and CSV uploads
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple


class DataLoader:
    """Handle data loading from various sources"""
    
    def __init__(self):
        self.data = None
        self.returns = None
        self.log_returns = None
        
    def fetch_yahoo_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., 'AAPL', 'SPY')
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        frequency : str
            'daily', 'weekly', or 'monthly'
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with price data
        """
        try:
            # Download data
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Use Adj Close if available, otherwise Close
            if 'Adj Close' in data.columns:
                prices = data[['Adj Close']].copy()
                prices.columns = ['Close']
            else:
                prices = data[['Close']].copy()
            
            # Resample if needed
            if frequency == "weekly":
                prices = prices.resample('W').last().dropna()
            elif frequency == "monthly":
                prices = prices.resample('M').last().dropna()
            
            self.data = prices
            self._calculate_returns()
            
            return prices
            
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")
    
    def load_csv(self, file_path: str, date_column: str = None, 
                 value_column: str = None) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Parameters:
        -----------
        file_path : str
            Path to CSV file
        date_column : str, optional
            Name of date column
        value_column : str, optional
            Name of value column
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        try:
            data = pd.read_csv(file_path)
            
            # Try to identify date column
            if date_column is None:
                date_cols = [col for col in data.columns 
                           if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    date_column = date_cols[0]
            
            if date_column:
                data[date_column] = pd.to_datetime(data[date_column])
                data.set_index(date_column, inplace=True)
            
            # If value column specified, keep only that
            if value_column:
                data = data[[value_column]]
                data.columns = ['Close']
            
            self.data = data
            self._calculate_returns()
            
            return data
            
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    
    def _calculate_returns(self):
        """Calculate simple and log returns"""
        if self.data is not None and 'Close' in self.data.columns:
            # Simple returns
            self.returns = self.data['Close'].pct_change().dropna()
            
            # Log returns
            self.log_returns = np.log(self.data['Close']).diff().dropna()
    
    def get_train_test_split(
        self,
        test_size: float = 0.2
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            Training and testing data
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        n = len(self.data)
        train_size = int(n * (1 - test_size))
        
        train = self.data['Close'][:train_size]
        test = self.data['Close'][train_size:]
        
        return train, test
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics of the data"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        stats = pd.DataFrame()
        
        # Price statistics
        stats['Prices'] = self.data['Close'].describe()
        
        # Returns statistics if available
        if self.returns is not None:
            stats['Returns'] = self.returns.describe()
        
        return stats
    
    def check_stationarity(self) -> dict:
        """
        Check if series is stationary using ADF test
        
        Returns:
        --------
        dict
            Test results
        """
        from statsmodels.tsa.stattools import adfuller
        
        if self.data is None:
            raise ValueError("No data loaded")
        
        result = adfuller(self.data['Close'].dropna())
        
        return {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4],
            'Is Stationary': result[1] < 0.05
        }


class PreloadedDatasets:
    """Manage pre-loaded interesting datasets"""
    
    DATASETS = {
        '2008 Financial Crisis (S&P 500)': {
            'ticker': '^GSPC',
            'start': '2007-01-01',
            'end': '2009-12-31',
            'description': 'S&P 500 during the 2008 financial crisis'
        },
        'COVID-19 Crash (2020)': {
            'ticker': '^GSPC',
            'start': '2019-01-01',
            'end': '2021-12-31',
            'description': 'Market crash and recovery during COVID-19'
        },
        'Bitcoin Bull Run (2020-2021)': {
            'ticker': 'BTC-USD',
            'start': '2020-01-01',
            'end': '2021-12-31',
            'description': 'Bitcoin explosive growth period'
        },
        'Tech Bubble Burst (2000-2002)': {
            'ticker': '^IXIC',
            'start': '2000-01-01',
            'end': '2002-12-31',
            'description': 'Dot-com bubble burst'
        },
        'Oil Price Collapse (2014-2016)': {
            'ticker': 'CL=F',
            'start': '2014-01-01',
            'end': '2016-12-31',
            'description': 'Crude oil price collapse'
        }
    }
    
    @classmethod
    def get_dataset_names(cls):
        """Get list of available dataset names"""
        return list(cls.DATASETS.keys())
    
    @classmethod
    def load_dataset(cls, name: str) -> Tuple[pd.DataFrame, str]:
        """
        Load a preloaded dataset
        
        Parameters:
        -----------
        name : str
            Dataset name
            
        Returns:
        --------
        Tuple[pd.DataFrame, str]
            Data and description
        """
        if name not in cls.DATASETS:
            raise ValueError(f"Dataset {name} not found")
        
        config = cls.DATASETS[name]
        loader = DataLoader()
        
        data = loader.fetch_yahoo_data(
            ticker=config['ticker'],
            start_date=config['start'],
            end_date=config['end']
        )
        
        return data, config['description']
