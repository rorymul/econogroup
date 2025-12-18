"""
Precious Metals Forecasting Dashboard
Interactive Financial Econometrics Application
FIN41660 - Group Project
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Metals Forecasting Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Force light mode */
    :root {
        color-scheme: light !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: white !important;
    }
    
    [data-testid="stHeader"] {
        background-color: white !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Ensure all text is readable in light mode */
    .stMarkdown, .stText, p, span, div {
        color: #262730 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #FFD700;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        border: 3px solid #FFA500;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #FFA500;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("# 💰 Precious Metals Forecasting Dashboard")
st.markdown("### *Advanced Time Series Analysis with OLS, ARIMA & GARCH*")
st.markdown("---")

# Data loading with caching
@st.cache_data(ttl=3600)
def load_data():
    """Load precious metals and market indicator data"""
    
    tickers = {'gold': 'GC=F', 'silver': 'SI=F', 'platinum': 'PL=F', 'palladium': 'PA=F'}
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=10*365)
    
    try:
        # Download metals data
        data = yf.download(
            list(tickers.values()),
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )
        
        # Handle MultiIndex columns for multiple tickers
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close'].copy()
        else:
            # Single ticker case
            prices = pd.DataFrame(data['Close'])
            prices.columns = ['Close']
        
        inverse_tickers = {v: k for k, v in tickers.items()}
        prices = prices.rename(columns=inverse_tickers)
        prices = prices.dropna(how='all')
        
        # Download market indicators
        other_tickers = {
            'vix': '^VIX',
            'us2y_yield': '^IRX',
            'usd_index': 'DX-Y.NYB',
            'us10y_yield': '^TNX',
            'wti_oil': 'CL=F'
        }
        
        other_data = yf.download(
            list(other_tickers.values()),
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )
        
        # Handle MultiIndex columns
        if isinstance(other_data.columns, pd.MultiIndex):
            df1 = other_data['Close'].copy()
        else:
            df1 = pd.DataFrame(other_data['Close'])
            df1.columns = ['Close']
        
        inverse_other = {v: k for k, v in other_tickers.items()}
        df1 = df1.rename(columns=inverse_other)
        df1 = df1.dropna(how='all')
        
        # Combine data with outer join first to see what we have
        df = prices.join(df1, how='outer')
        
        # Forward fill missing values (common for some indices) - up to 5 days
        df = df.fillna(method='ffill', limit=5).fillna(method='bfill', limit=5)
        
        # Now drop any rows that are still completely empty
        df = df.dropna(how='all')
        
        # For metals specifically, don't forward fill - only keep actual trading days
        # But for indices, we already forward filled above
        
        # Calculate log returns
        metals = ['gold', 'silver', 'platinum', 'palladium']
        for metal in metals:
            if metal in df.columns:
                df[f'{metal}_lr'] = np.log(df[metal] / df[metal].shift(1))
        
        if 'vix' in df.columns:
            df['vix_lr'] = np.log(df['vix'] / df['vix'].shift(1))
        if 'usd_index' in df.columns:
            df['usd_index_lr'] = np.log(df['usd_index'] / df['usd_index'].shift(1))
        if 'wti_oil' in df.columns:
            df['wti_oil_lr'] = np.log(df['wti_oil'] / df['wti_oil'].shift(1))
        if 'us10y_yield' in df.columns:
            df['us10y_yield_change'] = df['us10y_yield'] - df['us10y_yield'].shift(1)
        if 'us2y_yield' in df.columns:
            df['us2y_yield_change'] = df['us2y_yield'] - df['us2y_yield'].shift(1)
        
        # Drop infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df, prices
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty dataframes to prevent crashes
        return pd.DataFrame(), pd.DataFrame()

# Load data
with st.spinner("🔄 Loading market data..."):
    df, prices = load_data()
    
# Check if data loaded successfully
if df.empty:
    st.error("Failed to load data from Yahoo Finance. Please try again later or check your internet connection.")
    st.stop()

# Sidebar
st.sidebar.markdown("## 🎮 Control Panel")
st.sidebar.success(f"✅ Data loaded: {len(df)} trading days")
st.sidebar.info(f"📅 **Period:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
st.sidebar.markdown("---")

# Metal selection
metal_choice = st.sidebar.selectbox(
    "🎯 Select Precious Metal:",
    ['gold', 'silver', 'platinum', 'palladium'],
    format_func=lambda x: f"{x.title()} {'🥇' if x=='gold' else '🥈' if x=='silver' else '⚪' if x=='platinum' else '⚫'}"
)

# Model selection
st.sidebar.markdown("---")
model_choice = st.sidebar.radio(
    "🔧 Select Model:",
    ['OLS Regression', 'ARIMA Forecasting', 'GARCH Volatility']
)

# Main content
if model_choice == 'OLS Regression':
    st.markdown(f"## 📊 OLS Regression Analysis: {metal_choice.upper()}")
    st.info("💡 **Objective:** Analyze how this metal responds to market stress indicators (VIX, USD, Oil, Yields)")
    
    # Prepare data
    data_metal = df[[f'{metal_choice}_lr', 'vix_lr', 'usd_index_lr', 'wti_oil_lr', 
                     'us10y_yield_change', 'us2y_yield_change']].dropna()
    
    # Check if we have enough data
    if len(data_metal) < 50:
        st.error(f"Not enough data points for {metal_choice}. Need at least 50 observations, have {len(data_metal)}.")
        
        # Show debugging info
        with st.expander("🔍 Debug Information"):
            st.write(f"**Total rows in main dataframe:** {len(df)}")
            st.write(f"**Rows with {metal_choice} returns:** {df[f'{metal_choice}_lr'].notna().sum()}")
            st.write(f"**Missing values after selecting columns:**")
            missing = df[[f'{metal_choice}_lr', 'vix_lr', 'usd_index_lr', 'wti_oil_lr', 
                         'us10y_yield_change', 'us2y_yield_change']].isnull().sum()
            st.dataframe(missing)
            st.info("💡 Try selecting a different metal (Gold or Silver typically have the most data)")
        st.stop()
    
    # Fit OLS model with error handling
    try:
        formula = f'{metal_choice}_lr ~ vix_lr + usd_index_lr + wti_oil_lr + us10y_yield_change + us2y_yield_change'
        model_ols = smf.ols(formula=formula, data=data_metal).fit()
    except Exception as e:
        st.error(f"Error fitting OLS model: {str(e)}")
        st.info(f"Data shape: {data_metal.shape}")
        st.info(f"Missing values:\n{data_metal.isnull().sum()}")
        st.stop()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 R-squared", f"{model_ols.rsquared:.4f}")
    with col2:
        st.metric("🎯 Adj R-squared", f"{model_ols.rsquared_adj:.4f}")
    with col3:
        st.metric("📉 AIC", f"{model_ols.aic:.2f}")
    with col4:
        st.metric("📈 F-statistic", f"{model_ols.fvalue:.2f}")
    
    st.markdown("---")
    
    # Coefficients table
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Regression Coefficients")
        
        # Create mapping for cleaner variable names
        variable_names = {
            'Intercept': 'Intercept',
            'vix_lr': 'VIX Returns',
            'usd_index_lr': 'USD Index Returns',
            'wti_oil_lr': 'WTI Oil Returns',
            'us10y_yield_change': 'Change in 10Y Yield',
            'us2y_yield_change': 'Change in 2Y Yield'
        }
        
        coef_df = pd.DataFrame({
            'Variable': [variable_names.get(var, var) for var in model_ols.params.index],
            'Coefficient': model_ols.params.values,
            'Std Error': model_ols.bse.values,
            'P-value': model_ols.pvalues.values,
            'Significant': ['✅ Yes' if p < 0.05 else '❌ No' for p in model_ols.pvalues.values]
        })
        
        # Format
        coef_df['Coefficient'] = coef_df['Coefficient'].apply(lambda x: f"{x:.6f}")
        coef_df['Std Error'] = coef_df['Std Error'].apply(lambda x: f"{x:.6f}")
        coef_df['P-value'] = coef_df['P-value'].apply(lambda x: f"{x:.4f}")
        
        # Style
        def highlight_significant(row):
            if '✅' in row['Significant']:
                return ['background-color: #d4edda'] * len(row)
            else:
                return ['background-color: #f8d7da'] * len(row)
        
        styled_df = coef_df.style.apply(highlight_significant, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 🧠 Key Insights")
        vix_coef = model_ols.params['vix_lr']
        vix_pval = model_ols.pvalues['vix_lr']
        
        if vix_pval < 0.05:
            # Calculate 10% VIX impact
            impact_10pct = vix_coef * 10
            
            if vix_coef > 0:
                st.success(f"**Positive Response to Volatility** ✅\n\n{metal_choice.title()} moves with market volatility.\n\n**VIX Coefficient:** {vix_coef:.4f}\n\n**Interpretation:** When VIX rises by 10%, {metal_choice} returns typically change by {impact_10pct:+.3f}%.")
            else:
                st.warning(f"**Negative Response to Volatility** ⚠️\n\n{metal_choice.title()} moves against market volatility.\n\n**VIX Coefficient:** {vix_coef:.4f}\n\n**Interpretation:** When VIX rises by 10%, {metal_choice} returns typically change by {impact_10pct:+.3f}%.")
        else:
            st.info(f"**No Significant Response to Volatility**\n\n{metal_choice.title()} does not show a statistically significant relationship with market volatility.\n\n**VIX Coefficient:** {vix_coef:.4f}\n**P-value:** {vix_pval:.4f}\n\n*Changes in VIX do not significantly explain {metal_choice} returns.*")
        
        st.metric("🎲 Durbin-Watson", f"{sm.stats.stattools.durbin_watson(model_ols.resid):.3f}", 
                 help="Tests for autocorrelation. ~2.0 is ideal")
    
    # Diagnostic plots
    st.markdown("### 📈 Model Diagnostics")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residuals Over Time', 'Q-Q Plot'),
        horizontal_spacing=0.1
    )
    
    # Residuals over time
    fig.add_trace(
        go.Scatter(
            x=model_ols.resid.index,
            y=model_ols.resid.values,
            mode='lines',
            line=dict(color='steelblue', width=1),
            name='Residuals',
            hovertemplate='Date: %{x}<br>Residual: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2, row=1, col=1)
    
    # Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(model_ols.resid, dist="norm")
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode='markers',
            marker=dict(color='steelblue', size=5, opacity=0.6),
            name='Sample Quantiles',
            hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=slope * osm + intercept,
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Reference Line',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1, showgrid=True, gridcolor='LightGray')
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2, showgrid=True, gridcolor='LightGray')
    fig.update_yaxes(title_text="Residuals", row=1, col=1, showgrid=True, gridcolor='LightGray')
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2, showgrid=True, gridcolor='LightGray')
    
    fig.update_layout(
        height=400,
        showlegend=True,
        template='plotly_white',
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif model_choice == 'ARIMA Forecasting':
    st.markdown(f"## 🔮 ARIMA Forecasting: {metal_choice.upper()}")
    st.info("💡 **Objective:** Forecast future returns using AutoRegressive Integrated Moving Average models")
    
    # Prepare data
    returns = df[f'{metal_choice}_lr'].dropna()
    
    # Check if we have data
    if len(returns) < 100:
        st.error(f"Not enough data for {metal_choice}. Need at least 100 observations, have {len(returns)}.")
        st.stop()
    
    try:
        returns = returns.asfreq('B')
    except Exception as e:
        st.error(f"Error setting business day frequency: {str(e)}")
        st.info("Attempting to continue without frequency setting...")
        pass
    
    # Forecast settings
    col1, col2 = st.columns([3, 1])
    with col1:
        forecast_days = st.slider("📅 Forecast Horizon (business days):", 5, 30, 10)
    with col2:
        st.metric("📊 Data Points", f"{len(returns):,}")
    
    st.markdown("---")
    
    # Model selection
    with st.spinner("🤖 Testing ARIMA models..."):
        arma_results = []
        for p in range(0, 4):
            for q in range(0, 4):
                try:
                    model = ARIMA(returns, order=(p, 0, q))
                    fitted = model.fit()
                    arma_results.append({
                        'AR(p)': p,
                        'MA(q)': q,
                        'AIC': fitted.aic,
                        'BIC': fitted.bic,
                        'Log-Likelihood': fitted.llf
                    })
                except:
                    pass
        
        arma_df = pd.DataFrame(arma_results).sort_values('AIC')
    
    # Display model comparison
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏆 ARIMA Model Comparison")
        st.markdown("*Models ranked by AIC (lower is better)*")
        
        # Style the table with color gradient
        display_df = arma_df.head(10).copy()
        display_df['AIC'] = display_df['AIC'].apply(lambda x: f"{x:.2f}")
        display_df['BIC'] = display_df['BIC'].apply(lambda x: f"{x:.2f}")
        display_df['Log-Likelihood'] = display_df['Log-Likelihood'].apply(lambda x: f"{x:.2f}")
        
        # Add rank column
        display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
        
        # Highlight best model
        def highlight_best(row):
            if row['Rank'] == 1:
                return ['background-color: #90EE90; font-weight: bold'] * len(row)
            elif row['Rank'] <= 3:
                return ['background-color: #E8F5E9'] * len(row)
            else:
                return [''] * len(row)
        
        styled_arma = display_df.style.apply(highlight_best, axis=1)
        st.dataframe(styled_arma, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 🎯 Best Model")
        best_p = int(arma_df.iloc[0]['AR(p)'])
        best_q = int(arma_df.iloc[0]['MA(q)'])
        
        st.success(f"**ARMA({best_p}, {best_q})**")
        st.metric("AIC", f"{arma_df.iloc[0]['AIC']:.2f}")
        st.metric("BIC", f"{arma_df.iloc[0]['BIC']:.2f}")
        
        st.markdown("---")
        st.info(f"**AR({best_p})**: Uses {best_p} lag(s)\n\n**MA({best_q})**: Uses {best_q} error term(s)")
    
    # Fit best model and forecast
    best_model = ARIMA(returns, order=(best_p, 0, best_q))
    best_fitted = best_model.fit()
    forecast = best_fitted.forecast(steps=forecast_days)
    
    st.markdown("---")
    
    # Forecast visualization
    st.markdown("### 📊 Forecast Visualization")
    
    fig = go.Figure()
    
    # Historical data (last 252 days)
    historical = returns.iloc[-252:]
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical.values * 100,  # Convert to percentage
        mode='lines',
        name='Historical Returns',
        line=dict(color='steelblue', width=2),
        hovertemplate='Date: %{x}<br>Return: %{y:.3f}%<extra></extra>'
    ))
    
    # Forecast
    future_dates = pd.date_range(returns.index[-1] + timedelta(days=1), periods=forecast_days, freq='B')
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast.values * 100,  # Convert to percentage
        mode='lines+markers',
        name='Forecast',
        line=dict(color='orange', width=3, dash='dash'),
        marker=dict(size=8, color='orange', symbol='diamond'),
        hovertemplate='Date: %{x}<br>Forecast: %{y:.3f}%<extra></extra>'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title=dict(
            text=f'{metal_choice.title()} Returns Forecast - ARMA({best_p},{best_q})',
            font=dict(size=18, color='#333')
        ),
        xaxis_title='Date',
        yaxis_title='Return (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridcolor='LightGray', zeroline=True)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    st.markdown("### 📋 Detailed Forecast")
    
    forecast_df = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m-%d'),
        'Day': range(1, forecast_days + 1),
        'Predicted Return (%)': (forecast.values * 100).round(4)
    })
    
    # Color code the returns
    def color_forecast_returns(val):
        if isinstance(val, (int, float)):
            color = '#d4edda' if val > 0 else '#f8d7da' if val < 0 else 'white'
            return f'background-color: {color}; font-weight: bold'
        return ''
    
    styled_forecast = forecast_df.style.applymap(color_forecast_returns, subset=['Predicted Return (%)'])
    st.dataframe(styled_forecast, use_container_width=True, hide_index=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    avg_forecast = forecast.mean() * 100
    with col1:
        st.metric("📊 Average Forecast", f"{avg_forecast:.3f}%")
    with col2:
        st.metric("📈 Max Forecast", f"{(forecast.max() * 100):.3f}%")
    with col3:
        st.metric("📉 Min Forecast", f"{(forecast.min() * 100):.3f}%")
    with col4:
        cumulative = ((1 + forecast).prod() - 1) * 100
        st.metric("🎯 Cumulative Return", f"{cumulative:.3f}%")
    
    if avg_forecast > 0:
        st.success(f"📈 **Bullish Signal!** Model predicts positive returns averaging {avg_forecast:.3f}% per day")
    else:
        st.warning(f"📉 **Bearish Signal!** Model predicts negative returns averaging {avg_forecast:.3f}% per day")

else:  # GARCH
    st.markdown(f"## ⚡ GARCH Volatility Analysis: {metal_choice.upper()}")
    st.info("💡 **Objective:** Model and forecast volatility clustering using GARCH(1,1)")
    
    # Prepare data
    returns = df[f'{metal_choice}_lr'].dropna() * 100
    
    # Check if we have enough data
    if len(returns) < 100:
        st.error(f"Not enough data for {metal_choice}. Need at least 100 observations, have {len(returns)}.")
        st.stop()
    
    # Forecast settings
    col1, col2 = st.columns([3, 1])
    with col1:
        forecast_days = st.slider("📅 Volatility Forecast Horizon:", 5, 30, 10)
    with col2:
        st.metric("📊 Data Points", f"{len(returns):,}")
    
    st.markdown("---")
    
    # Fit GARCH
    try:
        with st.spinner("⚙️ Fitting GARCH(1,1) model..."):
            garch_model = arch_model(returns, vol='GARCH', p=1, q=1)
            garch_fitted = garch_model.fit(disp='off')
    except Exception as e:
        st.error(f"Error fitting GARCH model: {str(e)}")
        st.info(f"This can happen with certain data patterns. Try a different metal or check your data.")
        st.stop()
    
    # Model summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 GARCH Parameters")
        
        params_df = pd.DataFrame({
            'Parameter': garch_fitted.params.index,
            'Value': garch_fitted.params.values,
            'Std Error': garch_fitted.std_err.values,
            'T-Statistic': garch_fitted.tvalues.values
        })
        
        params_df['Value'] = params_df['Value'].apply(lambda x: f"{x:.6f}")
        params_df['Std Error'] = params_df['Std Error'].apply(lambda x: f"{x:.6f}")
        params_df['T-Statistic'] = params_df['T-Statistic'].apply(lambda x: f"{x:.3f}")
        
        styled_params = params_df.style.set_properties(**{
            'background-color': '#f0f2f6',
            'border': '1px solid #ddd',
            'font-weight': 'bold'
        })
        
        st.dataframe(styled_params, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📊 Model Fit")
        st.metric("Log-Likelihood", f"{garch_fitted.loglikelihood:.2f}")
        st.metric("AIC", f"{garch_fitted.aic:.2f}")
        st.metric("BIC", f"{garch_fitted.bic:.2f}")
        
        st.markdown("---")
        st.info("**GARCH(1,1)**\n\nCaptures volatility clustering and persistence")
    
    # Generate forecast
    forecast_vol = garch_fitted.forecast(horizon=forecast_days)
    variance_forecast = forecast_vol.variance.values[-1]
    
    st.markdown("---")
    
    # Volatility visualization
    st.markdown("### 📈 Volatility Analysis")
    
    cond_vol = garch_fitted.conditional_volatility
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Historical Conditional Volatility', f'Volatility Forecast ({forecast_days} Days)'),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )
    
    # Historical volatility
    fig.add_trace(
        go.Scatter(
            x=cond_vol.index[-252:],
            y=cond_vol.values[-252:],
            mode='lines',
            name='Conditional Volatility',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)',
            hovertemplate='Date: %{x}<br>Volatility: %{y:.3f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Forecast
    future_dates = pd.date_range(returns.index[-1] + timedelta(days=1), periods=forecast_days, freq='D')
    forecast_vol_vals = np.sqrt(variance_forecast)
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=forecast_vol_vals,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange', width=3, dash='dash'),
            marker=dict(size=10, color='orange', symbol='diamond'),
            hovertemplate='Date: %{x}<br>Forecast: %{y:.3f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1, showgrid=True, gridcolor='LightGray')
    fig.update_xaxes(title_text="Date", row=2, col=1, showgrid=True, gridcolor='LightGray')
    fig.update_yaxes(title_text="Volatility (%)", row=1, col=1, showgrid=True, gridcolor='LightGray')
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1, showgrid=True, gridcolor='LightGray')
    
    fig.update_layout(
        height=700,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatility metrics
    st.markdown("### 📊 Volatility Statistics")
    
    current_vol = cond_vol.iloc[-1]
    avg_forecast_vol = np.sqrt(variance_forecast).mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Volatility", f"{current_vol:.3f}%")
    with col2:
        st.metric("Avg Forecast Vol", f"{avg_forecast_vol:.3f}%")
    with col3:
        change = ((avg_forecast_vol - current_vol) / current_vol) * 100
        st.metric("Expected Change", f"{change:+.1f}%")
    with col4:
        st.metric("Annual Vol (252d)", f"{current_vol * np.sqrt(252):.2f}%")
    
    if avg_forecast_vol > current_vol:
        st.warning("⚠️ **Volatility Expected to INCREASE** - Higher risk ahead!")
    else:
        st.success("✅ **Volatility Expected to DECREASE** - Market calming down!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white;'>
        <h4>Financial Econometrics Project - FIN41660</h4>
        <p>Built using Streamlit | Data from Yahoo Finance</p>
        <p><em>For educational purposes only - Not financial advice</em></p>
    </div>
""", unsafe_allow_html=True)

