# 🔮 MARKET PROPHET

**Advanced Time Series Forecasting Platform**

**Course:** FIN41660 Financial Econometrics  
**Institution:** University College Dublin  
**Academic Year:** 2025/2026

---

## 🌐 LIVE DEMO

**Access Market Prophet directly in your browser:**

### **🔗 https://market-prophet-rvb2wk6x2jqr7vj5gemw2g.streamlit.app/**

No installation required! Try it now with any stock ticker or preloaded dataset.

---

## 📋 PROJECT OVERVIEW

Market Prophet is a comprehensive time series forecasting platform implementing three econometric models:
- **OLS (Ordinary Least Squares)** - Trend analysis
- **ARIMA (AutoRegressive Integrated Moving Average)** - Time series forecasting
- **GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)** - Volatility modeling

The platform provides both an interactive web dashboard and a standalone command-line tool for analyzing financial time series data.

---

## 🚀 QUICK START

### **Option 1: Use the Live Demo**

Simply visit: **https://market-prophet-rvb2wk6x2jqr7vj5gemw2g.streamlit.app/**

1. Choose a data source in the sidebar
2. Load data (try ticker: AAPL)
3. Navigate to "Model Configuration" tab
4. Click "Train All Models"
5. View results in tabs 3 & 4

### **Option 2: Run Locally**

**Prerequisites:**
- Python 3.8 or higher
- Internet connection (for Yahoo Finance data)

**Installation:**

1. Extract the ZIP file
2. Open Command Prompt/Terminal
3. Navigate to the project folder:
   ```bash
   cd path/to/market_prophet
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Run the Interactive Dashboard:**

```bash
python -m streamlit run app.py
```

Your browser will open automatically with the Market Prophet dashboard.

**Run the Standalone Analysis Script:**

```bash
python standalone_analysis.py --ticker AAPL --start 2023-01-01 --end 2023-12-31
```

This runs a complete analysis and saves plots automatically.

---

## 📁 PROJECT STRUCTURE

```
market_prophet/
│
├── app.py                          # Interactive Streamlit dashboard
├── standalone_analysis.py          # Command-line analysis tool
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── modules/                        # Core model implementations
│   ├── __init__.py
│   ├── data_loader.py             # Data fetching & preprocessing
│   ├── ols_model.py               # OLS regression model
│   ├── arima_model.py             # ARIMA time series model
│   ├── garch_model.py             # GARCH volatility model
│   └── model_comparison.py        # Model evaluation & comparison
```

---

## 🎯 FEATURES

### **Data Sources**
- Yahoo Finance API (real-time stock, crypto, forex, commodity data)
- Preloaded Historical Datasets (2008 Crisis, COVID-19, Bitcoin Bull Run, etc.)
- CSV file upload support
- Multiple frequencies (daily, weekly, monthly)

### **Models Implemented**
1. **OLS Model**
   - Linear and quadratic trends
   - R², adjusted R², F-statistics
   - Diagnostic tests (heteroskedasticity, autocorrelation, normality)

2. **ARIMA Model**
   - Automatic order selection using AIC/BIC
   - Stationarity testing (ADF test)
   - ACF/PACF plots
   - Prediction intervals

3. **GARCH Model**
   - Volatility clustering
   - Conditional variance forecasts
   - VaR and Expected Shortfall calculations
   - Annualized volatility

### **Model Comparison**
- Multiple accuracy metrics (MAE, RMSE, MAPE, R²)
- Diebold-Mariano test for forecast comparison
- Forecast encompassing test
- Automatic ranking and best model selection

---

## 📊 USAGE EXAMPLES

### **Example 1: Analyze Apple Stock**

```bash
python standalone_analysis.py --ticker AAPL --start 2023-01-01 --end 2023-12-31
```

**Output:**
- Model training results
- Forecast accuracy comparison
- Saved plot: `analysis_AAPL_2023-01-01_2023-12-31.png`

### **Example 2: Compare Multiple Stocks**

```bash
python standalone_analysis.py --ticker AAPL --start 2023-01-01 --end 2023-12-31
python standalone_analysis.py --ticker MSFT --start 2023-01-01 --end 2023-12-31
python standalone_analysis.py --ticker GOOGL --start 2023-01-01 --end 2023-12-31
```

### **Example 3: Live Web Interface**

Visit: **https://market-prophet-rvb2wk6x2jqr7vj5gemw2g.streamlit.app/**

Or run locally:
```bash
python -m streamlit run app.py
```

Then in the browser:
1. Select "Yahoo Finance" data source
2. Enter ticker (e.g., AAPL, BTC-USD, ^GSPC)
3. Click "Load Data"
4. Go to "Model Configuration" tab
5. Click "Train All Models"
6. View results in tabs 3 & 4

---

## 🧪 TESTING

To verify the installation works:

```bash
# Test with Apple stock
python standalone_analysis.py --ticker AAPL --start 2023-01-01 --end 2023-12-31

# You should see:
# ✓ Data loaded successfully
# ✓ OLS trained
# ✓ ARIMA trained
# ✓ GARCH trained
# ✓ Comparison table
# ✓ Plot saved
```

---

## 📦 DEPENDENCIES

All dependencies are listed in `requirements.txt`:

**Core:**
- numpy (≥1.24.0)
- pandas (≥2.0.0)
- scipy (≥1.11.0)

**Models:**
- statsmodels (≥0.14.0) - ARIMA
- arch (≥6.2.0) - GARCH
- scikit-learn (≥1.3.0) - Metrics

**Data:**
- yfinance (≥0.2.28) - Yahoo Finance API

**Visualization:**
- matplotlib (≥3.7.0)
- seaborn (≥0.12.0)
- plotly (≥5.17.0)

**Dashboard:**
- streamlit (≥1.28.0)

---

## 🎓 ACADEMIC CONTEXT

This project demonstrates:
- Implementation of econometric models from theory to practice
- Model comparison and validation techniques
- Software engineering best practices
- Real-world application of financial econometrics

**Key Learning Outcomes:**
- OLS regression for trend analysis
- ARIMA for time series forecasting
- GARCH for volatility modeling
- Statistical testing (Diebold-Mariano, ADF, Ljung-Box)
- Model evaluation metrics

---

## 🌐 DEPLOYMENT

**Production URL:** https://market-prophet-rvb2wk6x2jqr7vj5gemw2g.streamlit.app/

The platform is deployed on Streamlit Cloud, providing:
- Free hosting
- Automatic updates from GitHub
- 24/7 availability
- No installation required for end users

To deploy your own version:
1. Fork/upload code to GitHub
2. Visit https://share.streamlit.io
3. Connect your repository
4. Deploy with one click

---

## 🔧 TROUBLESHOOTING

### **"pip is not recognized"**
```bash
python -m pip install -r requirements.txt
```

### **"streamlit is not recognized"**
```bash
python -m streamlit run app.py
```

### **"ModuleNotFoundError"**
Ensure you're in the `market_prophet` folder and ran:
```bash
pip install -r requirements.txt
```

### **GARCH convergence issues**
Try simpler parameters or longer time series.

### **Live demo not loading**
The app may be "sleeping" - wait 30 seconds for it to wake up on first visit.

---

## 📖 DOCUMENTATION

For detailed information about each model:

1. **OLS Model**: See `modules/ols_model.py`
2. **ARIMA Model**: See `modules/arima_model.py`
3. **GARCH Model**: See `modules/garch_model.py`
4. **Model Comparison**: See `modules/model_comparison.py`

Each module contains comprehensive docstrings explaining the methodology.

---

## 🎥 VIDEO DEMONSTRATION

See the accompanying video presentation for:
- Live demonstration of the dashboard and live URL
- Walkthrough of model training
- Explanation of results interpretation
- Comparison of forecasting accuracy

---

## 👥 PROJECT TEAM

Eoin McConnon  
Rian O Gorman  
Luke Monahan

FIN41660 Financial Econometrics  
University College Dublin  
December 2024

---

## 🙏 ACKNOWLEDGMENTS

- **Yahoo Finance** for providing free financial data API
- **Statsmodels** and **Arch** libraries for econometric implementations
- **Streamlit** for the interactive dashboard framework
- **Streamlit Cloud** for free hosting
- **Professor Alessia Paccagnini** for course guidance

---

## 📧 SUPPORT

For questions about this project, please refer to:
1. This README file
2. Code comments in each module
3. The live demo at https://market-prophet-rvb2wk6x2jqr7vj5gemw2g.streamlit.app/
4. The accompanying report (PDF)
5. The video presentation

---

**Happy Forecasting! 🔮📈**

**Try it now:** https://market-prophet-rvb2wk6x2jqr7vj5gemw2g.streamlit.app/
