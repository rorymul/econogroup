"""
🏆 Precious Metals vs Market Volatility Dashboard
Interactive Financial Econometrics Application
FIN41660 - Group Project
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Page config
st.set_page_config(
    page_title="💰 Metals Forecasting Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fun styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        background-color: #FFD700;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        border: 3px solid #FFA500;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FFA500;
        transform: scale(1.05);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    h1 {
        color: #FFD700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h2, h3 {
        color: #FFA500 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title with emojis
st.markdown("# 💰 Precious Metals Forecasting Lab 📈")
st.markdown("### 🔬 *Analyze • Forecast • Predict Market Volatility*")

# Sidebar
st.sidebar.markdown("## 🎮 Control Panel")
st.sidebar.markdown("---")

# Data loading with caching
@st.cache_data(ttl=3600)
def load_data():
    """Load precious metals and market indicator data"""
    
    # Define tickers
    tickers = {'gold': 'GC=F', 'silver': 'SI=F', 'platinum': 'PL=F', 'palladium': 'PA=F'}
    
    # Date range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=10*365)
    
    # Download metals data
    data = yf.download(
        list(tickers.values()),
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False
    )
    
    inverse_tickers = {v: k for k, v in tickers.items()}
    prices = data['Close'].copy()
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
    
    inverse_other = {v: k for k, v in other_tickers.items()}
    df1 = other_data['Close'].copy()
    df1 = df1.rename(columns=inverse_other)
    df1 = df1.dropna(how='all')
    
    # Combine data
    df = prices.join(df1, how='inner')
    
    # Calculate log returns
    metals = ['gold', 'silver', 'platinum', 'palladium']
    for metal in metals:
        df[f'{metal}_lr'] = np.log(df[metal] / df[metal].shift(1))
    
    df['vix_lr'] = np.log(df['vix'] / df['vix'].shift(1))
    df['usd_index_lr'] = np.log(df['usd_index'] / df['usd_index'].shift(1))
    df['wti_oil_lr'] = np.log(df['wti_oil'] / df['wti_oil'].shift(1))
    df['us10y_yield_change'] = df['us10y_yield'] - df['us10y_yield'].shift(1)
    df['us2y_yield_change'] = df['us2y_yield'] - df['us2y_yield'].shift(1)
    
    return df, prices

# Load data with spinner
with st.spinner("🔄 Loading market data from Yahoo Finance..."):
    df, prices = load_data()

st.sidebar.success(f"✅ Data loaded: {len(df)} trading days")
st.sidebar.markdown(f"📅 **Period:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

# Metal selection
st.sidebar.markdown("---")
st.sidebar.markdown("## 🎯 Select Your Metal")
metal_choice = st.sidebar.selectbox(
    "Choose a precious metal:",
    ['gold', 'silver', 'platinum', 'palladium'],
    format_func=lambda x: f"{'🥇' if x=='gold' else '🥈' if x=='silver' else '⚪' if x=='platinum' else '⚫'} {x.title()}"
)

# Model selection
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔧 Choose Your Model")
model_choice = st.sidebar.radio(
    "Select forecasting method:",
    ['📊 OLS Regression', '📈 ARIMA Forecasting', '📉 GARCH Volatility'],
    help="Each model reveals different market insights!"
)

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["📊 Model Analysis", "📈 Interactive Charts", "🎲 Quick Stats"])

with tab1:
    st.markdown(f"## 🔍 Analyzing: **{metal_choice.upper()}** {' 🥇' if metal_choice=='gold' else '🥈' if metal_choice=='silver' else '⚪' if metal_choice=='platinum' else '⚫'}")
    
    if model_choice == '📊 OLS Regression':
        st.markdown("### 🎯 OLS: Safe Haven Analysis")
        st.info("💡 **What does this tell us?** How does this metal respond to market stress indicators like VIX, USD strength, and oil prices?")
        
        # Prepare data
        data_metal = df[[f'{metal_choice}_lr', 'vix_lr', 'usd_index_lr', 'wti_oil_lr', 
                         'us10y_yield_change', 'us2y_yield_change']].dropna()
        
        # Fit OLS model
        formula = f'{metal_choice}_lr ~ vix_lr + usd_index_lr + wti_oil_lr + us10y_yield_change + us2y_yield_change'
        model_ols = smf.ols(formula=formula, data=data_metal).fit()
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 R-squared", f"{model_ols.rsquared:.4f}", 
                     help="How much variance is explained by the model")
        with col2:
            st.metric("🎯 Adj R-squared", f"{model_ols.rsquared_adj:.4f}")
        with col3:
            st.metric("📉 AIC", f"{model_ols.aic:.2f}")
        
        # Coefficients
        st.markdown("#### 🔢 Regression Coefficients")
        coef_df = pd.DataFrame({
            'Variable': model_ols.params.index,
            'Coefficient': model_ols.params.values,
            'P-value': model_ols.pvalues.values,
            'Significant': ['✅ Yes' if p < 0.05 else '❌ No' for p in model_ols.pvalues.values]
        })
        
        # Format the dataframe nicely
        coef_df['Coefficient'] = coef_df['Coefficient'].apply(lambda x: f"{x:.6f}")
        coef_df['P-value'] = coef_df['P-value'].apply(lambda x: f"{x:.4f}")
        
        # Style the dataframe
        def highlight_significant(row):
            if '✅' in row['Significant']:
                return ['background-color: #d4edda'] * len(row)
            else:
                return ['background-color: #f8d7da'] * len(row)
        
        styled_df = coef_df.style.apply(highlight_significant, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Interpretation
        st.markdown("#### 🧠 Key Insights:")
        vix_coef = model_ols.params['vix_lr']
        if vix_coef > 0:
            st.success(f"✅ **Safe Haven Confirmed!** When VIX rises by 1%, {metal_choice} returns increase by {vix_coef:.4f}%")
        else:
            st.warning(f"⚠️ **Not a Safe Haven!** When VIX rises by 1%, {metal_choice} returns decrease by {abs(vix_coef):.4f}%")
        
        # Residuals plot
        st.markdown("#### 📉 Model Diagnostics")
        
        # Create Plotly subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Residuals Over Time', 'Q-Q Plot')
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
                marker=dict(color='steelblue', size=4),
                name='Residuals',
                hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=slope * osm + intercept,
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Reference Line'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1, showgrid=True, gridcolor='LightGray')
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2, showgrid=True, gridcolor='LightGray')
        fig.update_yaxes(title_text="Residuals", row=1, col=1, showgrid=True, gridcolor='LightGray')
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2, showgrid=True, gridcolor='LightGray')
        
        fig.update_layout(
            height=400,
            showlegend=False,
            template='plotly_white',
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif model_choice == '📈 ARIMA Forecasting':
        st.markdown("### 🔮 ARIMA: Price Return Forecasting")
        st.info("💡 **What does this tell us?** Predicts future returns based on past patterns (autoregressive + moving average)")
        
        # Prepare data
        returns = df[f'{metal_choice}_lr'].dropna()
        returns = returns.asfreq('B')  # Business day frequency
        
        # Allow user to select forecast horizon
        forecast_days = st.slider("📅 Forecast Horizon (business days):", 1, 30, 10)
        
        # ARIMA model selection
        with st.spinner("🤖 Finding best ARIMA model..."):
            arma_results = []
            for p in range(0, 3):
                for q in range(0, 3):
                    try:
                        model = ARIMA(returns, order=(p, 0, q))
                        fitted = model.fit()
                        arma_results.append({
                            'AR(p)': p,
                            'MA(q)': q,
                            'AIC': fitted.aic,
                            'BIC': fitted.bic
                        })
                    except:
                        pass
            
            arma_df = pd.DataFrame(arma_results).sort_values('AIC')
        
        st.markdown("#### 🏆 Best ARIMA Models (by AIC)")
        
        # Style the table
        styled_arma = arma_df.head(5).style.background_gradient(
            cmap='RdYlGn_r',
            subset=['AIC', 'BIC']
        ).format({'AIC': '{:.2f}', 'BIC': '{:.2f}'})
        
        st.dataframe(styled_arma, use_container_width=True, hide_index=True)
        
        # Fit best model
        best_p = int(arma_df.iloc[0]['AR(p)'])
        best_q = int(arma_df.iloc[0]['MA(q)'])
        
        best_model = ARIMA(returns, order=(best_p, 0, best_q))
        best_fitted = best_model.fit()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Best Model", f"ARMA({best_p}, {best_q})")
        with col2:
            st.metric("📊 AIC Score", f"{best_fitted.aic:.2f}")
        
        # Generate forecast
        forecast = best_fitted.forecast(steps=forecast_days)
        
        # Create interactive Plotly chart
        fig = go.Figure()
        
        # Historical returns (last 252 days)
        historical_data = returns.iloc[-252:]
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data.values,
            mode='lines',
            name='Historical Returns',
            line=dict(color='steelblue', width=2),
            hovertemplate='Date: %{x}<br>Return: %{y:.4f}%<extra></extra>'
        ))
        
        # Forecast
        future_dates = pd.date_range(returns.index[-1] + timedelta(days=1), periods=forecast_days, freq='B')
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast.values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange', width=3, dash='dash'),
            marker=dict(size=8, color='orange', symbol='circle'),
            hovertemplate='Date: %{x}<br>Forecast: %{y:.4f}%<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color="red", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{metal_choice.title()} Returns Forecast - ARMA({best_p},{best_q})',
                font=dict(size=18, color='#333', family='Arial Black')
            ),
            xaxis_title='Date',
            yaxis_title='Log Returns',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary
        st.markdown("#### 🎲 Forecast Summary")
        forecast_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Predicted Return (%)': (forecast.values * 100).round(4)
        })
        
        # Style the dataframe
        def color_returns(val):
            if isinstance(val, (int, float)):
                color = '#d4edda' if val > 0 else '#f8d7da' if val < 0 else 'white'
                return f'background-color: {color}'
            return ''
        
        styled_forecast = forecast_df.style.applymap(color_returns, subset=['Predicted Return (%)'])
        st.dataframe(styled_forecast, use_container_width=True, hide_index=True)
        
        avg_forecast = forecast.mean() * 100
        if avg_forecast > 0:
            st.success(f"📈 **Bullish Signal!** Average predicted return: +{avg_forecast:.3f}%")
        else:
            st.warning(f"📉 **Bearish Signal!** Average predicted return: {avg_forecast:.3f}%")
        
    else:  # GARCH
        st.markdown("### ⚡ GARCH: Volatility Forecasting")
        st.info("💡 **What does this tell us?** Models volatility clustering - periods of high volatility tend to follow high volatility")
        
        # Prepare data
        returns = df[f'{metal_choice}_lr'].dropna() * 100  # Scale to percentage
        
        # Forecast horizon
        forecast_days = st.slider("📅 Volatility Forecast Horizon:", 1, 30, 10)
        
        # Fit GARCH(1,1)
        with st.spinner("⚙️ Fitting GARCH(1,1) model..."):
            garch_model = arch_model(returns, vol='GARCH', p=1, q=1)
            garch_fitted = garch_model.fit(disp='off')
        
        # Display model summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Model", "GARCH(1,1)")
        with col2:
            st.metric("🎯 Log-Likelihood", f"{garch_fitted.loglikelihood:.2f}")
        with col3:
            st.metric("📉 AIC", f"{garch_fitted.aic:.2f}")
        
        # Parameters
        st.markdown("#### 🔢 Model Parameters")
        params_df = pd.DataFrame({
            'Parameter': garch_fitted.params.index,
            'Value': garch_fitted.params.values,
            'Std Error': garch_fitted.std_err.values
        })
        
        # Format and style
        params_df['Value'] = params_df['Value'].apply(lambda x: f"{x:.6f}")
        params_df['Std Error'] = params_df['Std Error'].apply(lambda x: f"{x:.6f}")
        
        styled_params = params_df.style.set_properties(**{
            'background-color': '#f0f2f6',
            'border': '1px solid #ddd'
        })
        
        st.dataframe(styled_params, use_container_width=True, hide_index=True)
        
        # Generate forecast
        forecast_vol = garch_fitted.forecast(horizon=forecast_days)
        variance_forecast = forecast_vol.variance.values[-1]
        
        # Plot volatility
        st.markdown("#### 📊 Volatility Analysis")
        
        # Conditional volatility
        cond_vol = garch_fitted.conditional_volatility
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=cond_vol.index[-252:],
            y=cond_vol.values[-252:],
            mode='lines',
            name='Conditional Volatility',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)',
            hovertemplate='Date: %{x}<br>Volatility: %{y:.3f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'{metal_choice.title()} Conditional Volatility (Last Year)',
                font=dict(size=16, color='#333', family='Arial Black')
            ),
            xaxis_title='Date',
            yaxis_title='Volatility (%)',
            template='plotly_white',
            height=400,
            hovermode='x unified',
            xaxis=dict(showgrid=True, gridcolor='LightGray'),
            yaxis=dict(showgrid=True, gridcolor='LightGray')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volatility forecast
        st.markdown("#### 🔮 Volatility Forecast")
        future_dates = pd.date_range(returns.index[-1] + timedelta(days=1), periods=forecast_days, freq='D')
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=future_dates,
            y=np.sqrt(variance_forecast),
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange', width=3, dash='dash'),
            marker=dict(size=10, color='orange', symbol='diamond'),
            hovertemplate='Date: %{x}<br>Forecast Vol: %{y:.3f}%<extra></extra>'
        ))
        
        fig2.update_layout(
            title=dict(
                text=f'Volatility Forecast (Next {forecast_days} Days)',
                font=dict(size=16, color='#333', family='Arial Black')
            ),
            xaxis_title='Date',
            yaxis_title='Volatility (%)',
            template='plotly_white',
            height=400,
            hovermode='x unified',
            xaxis=dict(showgrid=True, gridcolor='LightGray'),
            yaxis=dict(showgrid=True, gridcolor='LightGray')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Volatility stats
        st.markdown("#### 📊 Volatility Statistics")
        current_vol = cond_vol.iloc[-1]
        avg_forecast_vol = np.sqrt(variance_forecast).mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Volatility", f"{current_vol:.3f}%")
        with col2:
            st.metric("Avg Forecast Vol", f"{avg_forecast_vol:.3f}%")
        with col3:
            change = ((avg_forecast_vol - current_vol) / current_vol) * 100
            st.metric("Expected Change", f"{change:+.1f}%")
        
        if avg_forecast_vol > current_vol:
            st.warning("⚠️ **Volatility Expected to INCREASE** - Market uncertainty rising!")
        else:
            st.success("✅ **Volatility Expected to DECREASE** - Market calming down!")

with tab2:
    st.markdown("## 📊 Interactive Price Charts")
    
    # Time period selector
    period = st.selectbox("Select Time Period:", 
                         ['1 Month', '3 Months', '6 Months', '1 Year', '5 Years', 'All Data'])
    
    period_map = {
        '1 Month': 30,
        '3 Months': 90,
        '6 Months': 180,
        '1 Year': 252,
        '5 Years': 252*5,
        'All Data': len(df)
    }
    
    days = period_map[period]
    
    # Price chart
    metal_colors = {
        'gold': '#FFD700',
        'silver': '#C0C0C0',
        'platinum': '#E5E4E2',
        'palladium': '#CED0DD'
    }
    
    fig = go.Figure()
    
    price_data = df[metal_choice].iloc[-days:]
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data.values,
        mode='lines',
        name=metal_choice.title(),
        line=dict(color=metal_colors.get(metal_choice, 'steelblue'), width=2.5),
        fill='tozeroy',
        fillcolor=f'rgba({int(metal_colors.get(metal_choice, "#4682B4")[1:3], 16)}, {int(metal_colors.get(metal_choice, "#4682B4")[3:5], 16)}, {int(metal_colors.get(metal_choice, "#4682B4")[5:7], 16)}, 0.1)',
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'{metal_choice.title()} Price - Last {period}',
            font=dict(size=20, color='#333', family='Arial Black')
        ),
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white',
        height=500,
        hovermode='x unified',
        xaxis=dict(showgrid=True, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridcolor='LightGray')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Returns distribution
    st.markdown("### 📊 Returns Distribution")
    returns_subset = df[f'{metal_choice}_lr'].dropna().iloc[-days:] * 100
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Histogram(
        x=returns_subset,
        nbinsx=50,
        name='Returns',
        marker=dict(
            color='steelblue',
            line=dict(color='darkblue', width=1)
        ),
        hovertemplate='Return Range: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ))
    
    # Add mean line
    mean_return = returns_subset.mean()
    fig2.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text=f'Mean: {mean_return:.3f}%',
        annotation_position="top right"
    )
    
    fig2.update_layout(
        title=dict(
            text=f'{metal_choice.title()} Daily Returns Distribution',
            font=dict(size=18, color='#333', family='Arial Black')
        ),
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridcolor='LightGray')
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("## 🎲 Quick Statistics Dashboard")
    
    # Calculate statistics
    returns = df[f'{metal_choice}_lr'].dropna() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Mean Return", f"{returns.mean():.3f}%")
    with col2:
        st.metric("📉 Volatility", f"{returns.std():.3f}%")
    with col3:
        st.metric("📈 Max Return", f"{returns.max():.3f}%")
    with col4:
        st.metric("📉 Min Return", f"{returns.min():.3f}%")
    
    # Correlation heatmap
    st.markdown("### 🔥 Correlation Heatmap")
    corr_cols = ['gold_lr', 'silver_lr', 'platinum_lr', 'palladium_lr', 'vix_lr', 'usd_index_lr']
    corr_data = df[corr_cols].dropna()
    corr_matrix = corr_data.corr()
    
    # Create nicer labels
    labels = ['Gold', 'Silver', 'Platinum', 'Palladium', 'VIX', 'USD Index']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=[[f'{val:.3f}' for val in row] for row in corr_matrix.values],
        texttemplate='%{text}',
        textfont={"size": 14},
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
        showscale=True
    ))
    
    fig.update_layout(
        title=dict(
            text='Asset Correlation Matrix',
            font=dict(size=18, color='#333')
        ),
        template='plotly_white',
        height=600,
        xaxis=dict(side='bottom'),
        yaxis=dict(side='left', autorange='reversed'),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent performance
    st.markdown("### 📅 Recent Performance (Last 30 Days)")
    recent_prices = df[metal_choice].iloc[-30:]
    perf = ((recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1) * 100
    
    if perf > 0:
        st.success(f"📈 {metal_choice.title()} is UP {perf:.2f}% over the last 30 days!")
    else:
        st.error(f"📉 {metal_choice.title()} is DOWN {abs(perf):.2f}% over the last 30 days!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white;'>
        <h4>Financial Econometrics Project - FIN41660</h4>
        <p>Built using Streamlit | Data from Yahoo Finance</p>
        <p><em>For educational purposes only - Not financial advice</em></p>
    </div>
""", unsafe_allow_html=True)
