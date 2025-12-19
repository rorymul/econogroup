# 💰 Precious Metals Forecasting Dashboard

An interactive financial econometrics application analyzing precious metals (Gold, Silver, Platinum, Palladium) using advanced time series models.

**FIN41660 - Financial Econometrics Group Project**  
University College Dublin  
Due: December 21, 2025

---

## 📊 Overview

This dashboard provides comprehensive analysis of precious metals as safe-haven assets during market volatility using:
- **OLS Regression**: Analyzes metal returns against market stress indicators (VIX, USD Index, WTI Oil, Treasury Yields)
- **ARIMA Forecasting**: Forecasts future returns using AutoRegressive Integrated Moving Average models
- **GARCH Volatility**: Models and forecasts volatility clustering using GARCH(1,1)

---

## 🚀 Features

### 1. OLS Regression Analysis
- Multiple regression of metal returns against market indicators
- Coefficient significance testing
- Model diagnostics (Residuals plot, Q-Q plot)
- Durbin-Watson statistic for autocorrelation testing
- Interpretation of volatility response (weak/moderate/strong)

### 2. ARIMA Forecasting
- Automatic model selection from 16 ARIMA(p,0,q) specifications
- Models ranked by AIC (lower is better)
- Adjustable forecast horizon (5-30 business days)
- Interactive forecast visualization
- Detailed forecast table with daily predictions

### 3. GARCH Volatility Analysis
- GARCH(1,1) volatility modeling
- Conditional volatility visualization
- Adjustable volatility forecast horizon
- Annualized volatility calculation (252 trading days)
- Volatility change metrics

---

## 🎨 Optimal Viewing Conditions

### ✅ **Recommended Setup**
- **Device**: Desktop or laptop computer
- **Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **Display**: Standard resolution (1920x1080 or higher recommended)
- **Theme**: Light mode for best experience

### 📱 **Mobile/Tablet Viewing**
- Dashboard is mobile-responsive and works on iPad/iPhone
- **Important**: For best results on mobile devices, please use **light mode**
- Some table headers may appear dark on iOS devices in dark mode
- All functionality remains intact regardless of device theme

### 🌐 **Browser Compatibility**
- Chrome (recommended)
- Firefox
- Safari
- Microsoft Edge
- Brave

---

## 📦 Installation & Setup

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- `streamlit>=1.28.0` - Web application framework
- `yfinance>=0.2.28` - Financial data from Yahoo Finance
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `plotly>=5.17.0` - Interactive visualizations
- `statsmodels>=0.14.0` - Statistical models
- `arch>=6.2.0` - GARCH models
- `scipy>=1.10.0` - Scientific computing

---

## 🏃 Running the Dashboard

1. **Navigate to project directory**:
   ```bash
   cd /path/to/metals-dashboard
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run metals_dashboard.py
   ```

3. **Open your browser**:
   - Streamlit will automatically open your browser
   - If not, navigate to: `http://localhost:8501`

4. **Start analyzing**:
   - Select a precious metal from the sidebar
   - Choose your analysis model (OLS, ARIMA, or GARCH)
   - Adjust parameters as needed

---

## 📈 Data Sources

### Metals Data
- **Gold** (GC=F): Gold Futures
- **Silver** (SI=F): Silver Futures
- **Platinum** (PL=F): Platinum Futures
- **Palladium** (PA=F): Palladium Futures

### Market Indicators
- **VIX** (^VIX): CBOE Volatility Index
- **USD Index** (DX-Y.NYB): U.S. Dollar Index
- **WTI Oil** (CL=F): West Texas Intermediate Crude Oil
- **US 10Y Yield** (^TNX): 10-Year Treasury Yield
- **US 2Y Yield** (^IRX): 2-Year Treasury Yield

**Data Provider**: Yahoo Finance via `yfinance` library  
**Period**: Last 10 years of daily data  
**Frequency**: Business days (weekdays only)

---

## 🎮 How to Use

### Step 1: Select Metal
Use the sidebar dropdown to choose from:
- 🥇 Gold
- 🥈 Silver
- ⚪ Platinum
- ⚫ Palladium

### Step 2: Choose Model
Select your analysis type:
- **OLS Regression**: Understand how metals respond to market stress
- **ARIMA Forecasting**: Predict future returns
- **GARCH Volatility**: Analyze and forecast volatility

### Step 3: Adjust Parameters
- **ARIMA**: Adjust forecast horizon (5-30 days)
- **GARCH**: Adjust volatility forecast horizon (5-30 days)

### Step 4: Interpret Results
- Read the Key Insights section
- Examine coefficient tables and significance
- Review diagnostic plots
- Analyze forecast visualizations

---

## 📊 Understanding the Models

### OLS Regression
**Purpose**: Quantify the relationship between metal returns and market indicators

**Interpretation**:
- **Positive VIX coefficient**: Metal moves with volatility (potential safe haven)
- **Negative VIX coefficient**: Metal moves against volatility (not a safe haven)
- **Significance (p < 0.05)**: Relationship is statistically reliable
- **R-squared**: Proportion of variance explained by the model

### ARIMA(p,0,q)
**Purpose**: Forecast future returns based on past patterns

**Components**:
- **AR(p)**: AutoRegressive terms (uses p past values)
- **I(0)**: No differencing (returns are already stationary)
- **MA(q)**: Moving Average terms (uses q past errors)

**Model Selection**: Lowest AIC indicates best fit

### GARCH(1,1)
**Purpose**: Model time-varying volatility (volatility clustering)

**Parameters**:
- **omega (ω)**: Constant term
- **alpha (α)**: ARCH term (yesterday's shock effect)
- **beta (β)**: GARCH term (volatility persistence)

**Key Insight**: High beta means volatility is persistent

---

## 🎓 Academic Context

### Assignment Requirements
This dashboard fulfills the requirements for:
- Interactive data visualization ✓
- Multiple econometric models ✓
- Professional presentation ✓
- Interpretable results ✓

### Methodology
- **Data Processing**: Log returns for normality
- **Statistical Testing**: Significance tests, diagnostic checks
- **Model Validation**: AIC/BIC for model selection, Q-Q plots for normality
- **Forecasting**: Out-of-sample predictions with adjustable horizons

---

## ⚠️ Important Notes

### Data Limitations
- Data is fetched in real-time from Yahoo Finance
- Historical data availability varies by asset
- Palladium may have limited historical data
- Market holidays result in missing data points

### Statistical Assumptions
- Log returns assumed approximately normal
- GARCH assumes volatility clustering
- ARIMA assumes stationarity of returns
- OLS assumes linear relationships

### Disclaimer
> **For educational purposes only**. This dashboard is a student project for FIN41660 Financial Econometrics. Results should not be used for actual trading or investment decisions. Past performance does not guarantee future results.

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Charts not displaying  
**Solution**: Refresh the page, check internet connection

**Issue**: "Not enough data" error  
**Solution**: Try selecting Gold or Silver (most reliable data)

**Issue**: Slow loading  
**Solution**: Initial data download takes 10-20 seconds; cached after first load

**Issue**: Dark mode table headers on iPad  
**Solution**: Switch device to light mode for optimal viewing

---

## 📁 Project Structure

```
metals-dashboard/
├── metals_dashboard.py    # Main application file
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## 👥 Contributors

**FIN41660 Group Project**  
University College Dublin  
MSc in Quantitative Finance

---

## 📝 License

This project is submitted as academic coursework for University College Dublin.  
© 2025 - For educational use only.

---

## 🙏 Acknowledgments

- **Data Source**: Yahoo Finance
- **Framework**: Streamlit
- **Statistical Libraries**: statsmodels, arch
- **Visualization**: Plotly
- **Course**: FIN41660 Financial Econometrics, UCD

---

## 📞 Support

For technical issues or questions about the dashboard:
1. Check this README first
2. Review the code comments in `metals_dashboard.py`
3. Consult course materials for econometric methodology

---

**Last Updated**: December 19, 2025  
**Version**: 1.0  
**Status**: ✅ Ready for Submission
