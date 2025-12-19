# 🔮 Market Prophet

**Advanced Time Series Forecasting Platform for Financial Econometrics**

A comprehensive application implementing OLS, ARIMA, and GARCH models for forecasting financial time series data. Built for the FIN41660 Financial Econometrics course at University College Dublin.

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Requirements](#requirements)
- [Contributors](#contributors)

---

## ✨ Features

### 📊 **Data Sources**
- **Yahoo Finance Integration**: Real-time stock, crypto, forex, and commodity data
- **Preloaded Datasets**: Historical crisis periods (2008, COVID-19, Tech Bubble, etc.)
- **CSV Upload**: Bring your own time series data

### 🔧 **Three Powerful Models**
1. **OLS (Ordinary Least Squares)**: Trend analysis with linear, quadratic, and exponential trends
2. **ARIMA**: Autoregressive Integrated Moving Average for time series forecasting
3. **GARCH**: Generalized Autoregressive Conditional Heteroskedasticity for volatility modeling

### 📈 **Analysis Features**
- Model comparison with accuracy metrics (MAE, RMSE, MAPE, R²)
- Forecast visualization with confidence intervals
- Diagnostic tests (stationarity, autocorrelation, heteroskedasticity)
- Rolling window validation
- Diebold-Mariano test for forecast comparison

### 🎮 **Educational Mode**
- Interactive quiz game based on econometrics concepts
- Learn when to use each model
- Gamified learning experience

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download
```bash
# If using Git
git clone <https://market-prophet-rvb2wk6x2jqr7vj5gemw2g.streamlit.app/>
cd market_prophet

# Or download and extract the ZIP file>
cd market_prophet

# Or download and extract the ZIP file>
cd market_prophet

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment 
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Quick Start

### Run the Interactive Dashboard
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Run the Standalone Script
```bash
python standalone_analysis.py --ticker AAPL --start 2020-01-01 --end 2023-12-31
```

---

## 📖 Usage Guide

### 1. **Loading Data**

#### Option A: Yahoo Finance
1. Select "Yahoo Finance" in the sidebar
2. Enter a ticker symbol (e.g., AAPL, MSFT, ^GSPC, BTC-USD)
3. Choose date range and frequency
4. Click "📥 Load Data"

#### Option B: Preloaded Datasets
1. Select "Preloaded Datasets"
2. Choose from interesting historical periods:
   - 2008 Financial Crisis
   - COVID-19 Crash
   - Bitcoin Bull Run
   - Tech Bubble Burst
   - Oil Price Collapse
3. Click "📥 Load Dataset"

#### Option C: Upload CSV
1. Select "Upload CSV"
2. Upload your CSV file
3. Select date and value columns
4. Click "📥 Load CSV"

### 2. **Configuring Models**

Navigate to the **"🔧 Model Configuration"** tab:

1. **Set Train/Test Split**: Adjust slider (default 20% test set)

2. **Configure Each Model**:
   - **OLS**: Choose trend type (linear/quadratic/exponential)
   - **ARIMA**: Auto-select order or manually specify (p,d,q)
   - **GARCH**: Set p and q parameters

3. **Set Forecast Horizon**: Number of periods to forecast

4. Click **"🚀 Train All Models"**

### 3. **Viewing Results**

#### Data Overview Tab
- View price and return charts
- See summary statistics
- Check stationarity (ADF test)

#### Forecasts & Results Tab
- See individual model forecasts
- View confidence intervals
- Examine diagnostic tests
- Check residual plots

#### Model Comparison Tab
- Compare accuracy metrics across models
- View comparison table with rankings
- See Diebold-Mariano test results
- Analyze forecast plots side-by-side

### 4. **Educational Mode**

Test your knowledge with:
- Theory questions on econometric concepts
- Calculation exercises
- Progressive difficulty levels
- Real-time scoring

---

## 📁 Project Structure

```
market_prophet/
│
├── app.py                          # Main Streamlit application
├── standalone_analysis.py          # Standalone .py script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── modules/
│   ├── __init__.py
│   ├── data_loader.py             # Data fetching & preprocessing
│   ├── ols_model.py               # OLS implementation
│   ├── arima_model.py             # ARIMA implementation
│   ├── garch_model.py             # GARCH implementation
│   ├── model_comparison.py        # Accuracy metrics & tests
│   └── visualizations.py          # Plotting functions
│
├── educational/
│   ├── quiz_engine.py             # Quiz functionality
│   └── quiz_questions.py          # Question database
│
└── data/
    └── sample_datasets/           # Preloaded datasets
```

---

## 🧮 Models Implemented

### 1. OLS (Ordinary Least Squares)
- **Purpose**: Trend analysis and regression-based forecasting
- **Trends**: Linear, Quadratic, Exponential
- **Outputs**: Fitted values, forecasts, prediction intervals
- **Diagnostics**: R², F-test, heteroskedasticity tests

### 2. ARIMA (AutoRegressive Integrated Moving Average)
- **Purpose**: Time series forecasting with autocorrelation
- **Features**: Auto-order selection, ADF test, ACF/PACF plots
- **Outputs**: Point forecasts, confidence intervals
- **Diagnostics**: Ljung-Box test, residual analysis

### 3. GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
- **Purpose**: Volatility modeling and forecasting
- **Variants**: GARCH, EGARCH, TARCH
- **Outputs**: Conditional volatility, VaR, Expected Shortfall
- **Diagnostics**: ARCH effects test, standardized residuals

---

## 📊 Accuracy Metrics

The application calculates multiple forecast accuracy metrics:

- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)

Plus statistical tests:
- **Diebold-Mariano Test**: Compare forecast accuracy between models
- **Forecast Encompassing Test**: Test if one forecast contains all information from another

---

## 📦 Requirements

Core dependencies:
```
numpy >= 1.24.0
pandas >= 2.0.0
statsmodels >= 0.14.0
arch >= 6.2.0
scikit-learn >= 1.3.0
yfinance >= 0.2.28
matplotlib >= 3.7.0
seaborn >= 0.12.0
streamlit >= 1.28.0
```

See `requirements.txt` for complete list.

---



## 🎓 Educational Value

This platform is designed to help students:
1. **Understand** when to use each forecasting model
2. **Practice** with real financial data
3. **Compare** model performance objectively
4. **Learn** through interactive quizzes
5. **Apply** econometric concepts to real problems

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'streamlit'`
- **Solution**: Ensure virtual environment is activated and dependencies installed

**Issue**: Yahoo Finance download fails
- **Solution**: Check internet connection, verify ticker symbol is correct

**Issue**: GARCH model fails to converge
- **Solution**: Try different p,q parameters or scale your returns

**Issue**: "No module named 'modules'"
- **Solution**: Ensure you're running from the `market_prophet` directory

---

## 📝 Usage Tips

1. **Start with preloaded datasets** to see how models work
2. **Use auto-order selection** for ARIMA initially
3. **Compare all three models** for best results
4. **Check diagnostic tests** before trusting forecasts
5. **Try different train/test splits** to validate robustness

---

## 🤝 Contributors

[Eoin McConnon, Rian O'Gorman, Luke Monahan]

**Course**: FIN41660 Financial Econometrics  
**Institution**: University College Dublin  
**Instructor**:  Professor Alessia Paccagnini  
**Academic Year**: 2025/2026

---

## 📄 License

This project is created for educational purposes as part of the FIN41660 course.

---

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free financial data API
- **Statsmodels** and **Arch** libraries for econometric implementations
- **Streamlit** for the interactive dashboard framework


---

## 📧 Support

For questions or issues:
1. Check this README
2. Review code comments
3. Consult course materials
4. Contact group members

---

**Happy Forecasting! 🔮📈**
