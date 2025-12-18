"""
MARKET PROPHET - Complete Working Version
Financial Econometrics Time Series Forecasting Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.data_loader import DataLoader, PreloadedDatasets
from modules.ols_model import OLSForecaster
from modules.arima_model import ARIMAForecaster
from modules.garch_model import GARCHForecaster
from modules.model_comparison import ModelComparison

# Page configuration
st.set_page_config(
    page_title="Market Prophet",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.markdown("# 🔮 MARKET PROPHET")
st.markdown("### Advanced Time Series Forecasting with OLS, ARIMA & GARCH")
st.markdown("---")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_fitted' not in st.session_state:
    st.session_state.models_fitted = False

# Sidebar - Data Selection
st.sidebar.header("📊 Data Configuration")

data_source = st.sidebar.radio(
    "Data Source",
    ["Yahoo Finance", "Preloaded Datasets"],
    help="Choose where to load your time series data"
)

loader = DataLoader()

# Data loading
if data_source == "Yahoo Finance":
    st.sidebar.subheader("Yahoo Finance Settings")
    
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=730))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    if st.sidebar.button("📥 Load Data", type="primary"):
        with st.spinner("Fetching data..."):
            try:
                data = loader.fetch_yahoo_data(
                    ticker=ticker,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    frequency='daily'
                )
                st.session_state.data = data
                st.session_state.loader = loader
                st.session_state.data_loaded = True
                st.session_state.description = f"{ticker} from {start_date} to {end_date}"
                st.sidebar.success(f"✅ Loaded {len(data)} observations")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

else:  # Preloaded Datasets
    st.sidebar.subheader("Historical Periods")
    
    dataset_names = PreloadedDatasets.get_dataset_names()
    selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_names)
    
    if st.sidebar.button("📥 Load Dataset", type="primary"):
        with st.spinner(f"Loading {selected_dataset}..."):
            try:
                data, description = PreloadedDatasets.load_dataset(selected_dataset)
                temp_loader = DataLoader()
                temp_loader.data = data
                temp_loader._calculate_returns()
                st.session_state.data = data
                st.session_state.loader = temp_loader
                st.session_state.data_loaded = True
                st.session_state.description = description
                st.sidebar.success(f"✅ Loaded {len(data)} observations")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

# Main content
if st.session_state.data_loaded:
    data = st.session_state.data
    loader = st.session_state.loader
    description = st.session_state.description
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Data Overview",
        "🔧 Model Configuration",
        "🎯 Forecasts & Results",
        "📊 Model Comparison",
        "📚 About"
    ])
    
    # TAB 1: Data Overview
    with tab1:
        st.header("Data Overview")
        st.info(f"📝 {description}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Observations", len(data))
        with col2:
            st.metric("Start Date", str(data.index[0])[:10])
        with col3:
            st.metric("End Date", str(data.index[-1])[:10])
        with col4:
            current_price = float(data['Close'].iloc[-1])
            st.metric("Latest Price", f"${current_price:.2f}")
        
        st.subheader("Price Series")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data.index, data['Close'].values, linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title("Historical Prices")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        if loader.returns is not None and len(loader.returns) > 0:
            st.subheader("Return Series")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(loader.returns.index, loader.returns.values, linewidth=1, alpha=0.7)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            ax.set_xlabel("Date")
            ax.set_ylabel("Returns")
            ax.set_title("Simple Returns")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Summary Statistics")
            try:
                stats = loader.get_summary_statistics()
                st.dataframe(stats)
            except:
                st.write("Price statistics:")
                st.write(data['Close'].describe())
        
        with col2:
            st.subheader("Stationarity Test")
            try:
                stationarity = loader.check_stationarity()
                st.write(f"**ADF Statistic:** {stationarity['ADF Statistic']:.4f}")
                st.write(f"**p-value:** {stationarity['p-value']:.4f}")
                if stationarity['Is Stationary']:
                    st.success("✅ Series is stationary")
                else:
                    st.warning("⚠️ Series is non-stationary")
            except Exception as e:
                st.error(f"Could not perform test: {str(e)}")
    
    # TAB 2: Model Configuration
    with tab2:
        st.header("Model Configuration & Training")
        
        st.subheader("Data Split")
        test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        train_data, test_data = loader.get_train_test_split(test_size=test_size)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Set", len(train_data))
        with col2:
            st.metric("Test Set", len(test_data))
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔵 OLS Model")
            ols_enabled = st.checkbox("Enable OLS", value=True)
            if ols_enabled:
                ols_trend = st.selectbox("Trend Type", ["linear", "quadratic"], key="ols_trend")
        
        with col2:
            st.subheader("🟢 ARIMA Model")
            arima_enabled = st.checkbox("Enable ARIMA", value=True)
            if arima_enabled:
                arima_auto = st.checkbox("Auto-select order", value=True, key="arima_auto")
                if not arima_auto:
                    arima_p = st.number_input("p", 0, 5, 1, key="p")
                    arima_d = st.number_input("d", 0, 2, 1, key="d")
                    arima_q = st.number_input("q", 0, 5, 1, key="q")
        
        with col3:
            st.subheader("🟡 GARCH Model")
            garch_enabled = st.checkbox("Enable GARCH", value=True)
            if garch_enabled:
                garch_p = st.number_input("p (GARCH)", 1, 3, 1, key="gp")
                garch_q = st.number_input("q (ARCH)", 1, 3, 1, key="gq")
        
        st.markdown("---")
        forecast_horizon = st.slider("Forecast Horizon", 1, 30, 10)
        
        if st.button("🚀 Train All Models", type="primary", use_container_width=True):
            with st.spinner("Training models..."):
                models = {}
                
                if ols_enabled:
                    try:
                        ols_model = OLSForecaster()
                        ols_model.fit(train_data, trend=ols_trend)
                        models['OLS'] = ols_model
                        st.success("✅ OLS trained")
                    except Exception as e:
                        st.error(f"OLS failed: {str(e)}")
                
                if arima_enabled:
                    try:
                        arima_model = ARIMAForecaster()
                        if arima_auto:
                            arima_model.fit(train_data, auto=True)
                        else:
                            arima_model.fit(train_data, order=(arima_p, arima_d, arima_q))
                        models['ARIMA'] = arima_model
                        st.success(f"✅ ARIMA{arima_model.order} trained")
                    except Exception as e:
                        st.error(f"ARIMA failed: {str(e)}")
                
                if garch_enabled:
                    try:
                        train_returns = loader.returns[:len(train_data)-1]
                        garch_model = GARCHForecaster()
                        garch_model.fit(train_returns, p=garch_p, q=garch_q)
                        models['GARCH'] = garch_model
                        st.success(f"✅ GARCH({garch_p},{garch_q}) trained")
                    except Exception as e:
                        st.error(f"GARCH failed: {str(e)}")
                
                st.session_state.models = models
                st.session_state.train_data = train_data
                st.session_state.test_data = test_data
                st.session_state.forecast_horizon = forecast_horizon
                st.session_state.models_fitted = True
                
                st.success(f"🎉 Trained {len(models)} model(s)!")
    
    # TAB 3: Forecasts & Results
    with tab3:
        st.header("Forecasts & Results")
        
        if st.session_state.models_fitted:
            models = st.session_state.models
            train_data = st.session_state.train_data
            test_data = st.session_state.test_data
            
            # Add option to extend forecast
            st.subheader("⚙️ Forecast Settings")
            forecast_extension = st.slider(
                "Additional Future Periods (beyond test set)", 
                0, 60, 30,
                help="Forecast this many periods into the future beyond your test data"
            )
            
            for model_name, model in models.items():
                st.subheader(f"{model_name} Model")
                
                try:
                    if model_name != 'GARCH':
                        # Price forecasting models
                        fitted = model.get_fitted_values()
                        
                        st.write("**In-Sample Fit**")
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.plot(train_data.index, train_data.values, label='Actual', linewidth=2)
                        ax.plot(train_data.index, fitted.values, label='Fitted', linestyle='--', linewidth=2)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.set_title(f"{model_name} In-Sample Fit")
                        st.pyplot(fig)
                        plt.close()
                        
                        st.write("**Out-of-Sample Forecast**")
                        # Forecast test set + additional periods
                        total_forecast_steps = len(test_data) + forecast_extension
                        forecast_full = model.predict(steps=total_forecast_steps)
                        
                        # Split into test period and future period
                        forecast_test = forecast_full[:len(test_data)]
                        forecast_future = forecast_full[len(test_data):]
                        
                        # Create future dates
                        if len(forecast_future) > 0:
                            last_date = test_data.index[-1]
                            freq = pd.infer_freq(test_data.index)
                            if freq is None:
                                freq = 'D'  # Default to daily
                            future_dates = pd.date_range(
                                start=last_date + pd.Timedelta(days=1),
                                periods=len(forecast_future),
                                freq=freq
                            )
                        
                        fig, ax = plt.subplots(figsize=(14, 5))
                        
                        # Plot historical test data
                        ax.plot(test_data.index, test_data.values, 
                               label='Actual (Test Period)', linewidth=2, color='black')
                        
                        # Plot forecast for test period
                        ax.plot(test_data.index, forecast_test.values, 
                               label=f'{model_name} Forecast (Test)', 
                               linestyle='--', linewidth=2, color='red')
                        
                        # Plot extended forecast
                        if len(forecast_future) > 0:
                            ax.plot(future_dates, forecast_future.values, 
                                   label=f'{model_name} Future Forecast', 
                                   linestyle=':', linewidth=2.5, color='orange')
                            ax.axvline(x=test_data.index[-1], color='green', 
                                      linestyle='--', alpha=0.5, label='End of Known Data')
                        
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.set_title(f"{model_name} Forecast (Test + {forecast_extension} Future Periods)")
                        st.pyplot(fig)
                        plt.close()
                        
                        # Show metrics for test period only
                        metrics = ModelComparison.calculate_metrics(test_data, forecast_test)
                        st.write("**Accuracy Metrics (Test Period)**")
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(metrics_df.style.format("{:.4f}"))
                        
                        # Show future forecast values
                        if len(forecast_future) > 0:
                            with st.expander("📊 View Future Forecast Values"):
                                future_df = pd.DataFrame({
                                    'Date': future_dates,
                                    'Forecast': forecast_future.values
                                })
                                st.dataframe(future_df)
                    
                    else:  # GARCH
                        st.write("**Conditional Volatility**")
                        cond_vol = model.get_conditional_volatility()
                        
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.plot(cond_vol.index, cond_vol.values, linewidth=2)
                        ax.set_title("Conditional Volatility (GARCH)")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                        
                        # Extended volatility forecast
                        total_vol_forecast = st.session_state.forecast_horizon + forecast_extension
                        vol_forecast = model.predict_volatility(horizon=total_vol_forecast)
                        st.write(f"**Volatility Forecast (Next {total_vol_forecast} Periods)**")
                        st.write(vol_forecast)
                    
                    st.markdown("---")
                
                except Exception as e:
                    st.error(f"Error with {model_name}: {str(e)}")
        else:
            st.warning("⚠️ Train models first in Model Configuration tab")
    
    # TAB 4: Model Comparison
    with tab4:
        st.header("Model Comparison")
        
        if st.session_state.models_fitted:
            models = st.session_state.models
            test_data = st.session_state.test_data
            
            forecasts = {}
            for name, model in models.items():
                if name != 'GARCH':
                    try:
                        forecast = model.predict(steps=len(test_data))
                        forecasts[name] = forecast
                    except Exception as e:
                        st.warning(f"Could not forecast {name}: {str(e)}")
            
            if len(forecasts) > 0:
                st.subheader("Accuracy Comparison Table")
                comparison = ModelComparison.create_comparison_table(test_data, forecasts)
                st.dataframe(comparison[['MAE', 'RMSE', 'MAPE', 'R²']].style.format("{:.4f}"))
                
                best_model = comparison.index[0]
                st.success(f"🏆 **Best Model:** {best_model}")
                
                st.subheader("Visual Comparison")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(test_data.index, test_data.values, 'k-', label='Actual', linewidth=2)
                
                colors = {'OLS': 'blue', 'ARIMA': 'green'}
                for name, forecast in forecasts.items():
                    ax.plot(test_data.index, forecast.values, '--', 
                           label=f'{name} Forecast', color=colors.get(name, 'gray'), linewidth=2)
                
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_title("Model Comparison: Forecasts vs Actual")
                st.pyplot(fig)
                plt.close()
                
                if len(forecasts) >= 2:
                    st.subheader("Diebold-Mariano Test")
                    model_names = list(forecasts.keys())
                    try:
                        dm_test = ModelComparison.diebold_mariano_test(
                            test_data, forecasts[model_names[0]], forecasts[model_names[1]]
                        )
                        st.write(f"**Comparing:** {model_names[0]} vs {model_names[1]}")
                        st.write(f"**DM Statistic:** {dm_test['DM Statistic']:.4f}")
                        st.write(f"**p-value:** {dm_test['p-value']:.4f}")
                        st.write(f"**Result:** {dm_test['Conclusion']}")
                    except Exception as e:
                        st.error(f"DM test failed: {str(e)}")
            else:
                st.warning("No forecasts available for comparison")
        else:
            st.warning("⚠️ Train models first")
    
    # TAB 5: About
    with tab5:
        st.header("About Market Prophet")
        
        st.markdown("""
        ### 🔮 What is Market Prophet?
        
        Market Prophet is a comprehensive time series forecasting platform that implements
        three powerful econometric models:
        
        **1. OLS (Ordinary Least Squares) 📏**
        - Captures linear and quadratic trends
        - Best for: Trending markets
        - Output: Price forecasts
        
        **2. ARIMA (AutoRegressive Integrated Moving Average) 🔄**
        - Models autocorrelation in time series
        - Best for: Pattern-based forecasting
        - Output: Price forecasts with confidence intervals
        
        **3. GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) 📊**
        - Models volatility clustering
        - Best for: Risk management, VaR calculations
        - Output: Volatility forecasts
        
        ### 📚 How to Use
        
        1. **Load Data:** Choose Yahoo Finance or preloaded datasets
        2. **Configure Models:** Select which models to train
        3. **Train:** Click "Train All Models"
        4. **View Results:** See forecasts in tabs 3 & 4
        5. **Compare:** See which model performs best
        
        ### 🎓 Educational Value
        
        This platform demonstrates:
        - Real-world application of econometric models
        - Model comparison and evaluation
        - Forecast accuracy testing
        - Professional software development practices
        
        ### 👥 Created By
        
        [Eoin McConnon, Rian O Gorman, Luke Monahan]
        
        **Course:** FIN41660 Financial Econometrics  
        **Institution:** University College Dublin  
        **Year:** 2025/2026
        
        ---
        
        **💡 Tip:** Use the standalone script for detailed analysis:
        ```
        python standalone_analysis.py --ticker AAPL
        ```
        """)

else:
    st.info("👈 Please load data from the sidebar to begin")
    
    st.markdown("""
    ### Getting Started
    
    1. Choose a data source in the sidebar
    2. Select Yahoo Finance or a preloaded dataset
    3. Click "Load Data"
    4. Navigate through the tabs to explore!
    
    ### Quick Examples
    
    **Yahoo Finance Tickers:**
    - `AAPL` - Apple Inc.
    - `MSFT` - Microsoft
    - `GOOGL` - Google
    - `BTC-USD` - Bitcoin
    - `^GSPC` - S&P 500
    
    **Preloaded Datasets:**
    - 2008 Financial Crisis
    - COVID-19 Market Crash
    - Bitcoin Bull Run
    - And more!
    """)
