
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import logging
from datetime import datetime, timedelta
import io
import base64
from typing import Tuple, Dict, Any
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌡️ Temperature Forecast Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Custom Font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Forecast Summary Card */
    .forecast-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .forecast-summary h3 {
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    
    /* Performance Metrics */
    .performance-card {
        background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .performance-card:hover {
        border-color: #667eea;
        transform: scale(1.02);
    }
    
    /* Sidebar Enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Alert Styles */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Enhancements */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
    }
    
    /* Loading Spinner */
    .stSpinner {
        color: #667eea;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 10px;
        border: 1px solid #e9ecef;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Data Frame Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Plotly Chart Container */
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Status Indicators */
    .status-success {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-error {
        background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .metric-card {
            margin: 0.25rem 0;
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class TemperatureForecastApp:
    """Production-ready Temperature Forecasting Application"""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.forecast_results = None
        
    @st.cache_data
    def load_data(_self, file_path: str = 'daily-min-temperatures.csv') -> pd.DataFrame:
        """Load and validate temperature data with caching"""
        try:
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            
            # Data validation
            if df.empty:
                raise ValueError("Dataset is empty")
            
            if 'Temp' not in df.columns:
                raise ValueError("Temperature column 'Temp' not found")
            
            # Handle missing values
            df['Temp'] = df['Temp'].interpolate(method='linear')
            
            # Remove outliers using IQR method
            Q1 = df['Temp'].quantile(0.25)
            Q3 = df['Temp'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df['Temp'] < lower_bound) | (df['Temp'] > upper_bound)
            df.loc[outliers_mask, 'Temp'] = df['Temp'].median()
            
            logger.info(f"Data loaded successfully: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def perform_stationarity_test(self, data: pd.Series) -> Dict[str, Any]:
        """Perform Augmented Dickey-Fuller test for stationarity"""
        try:
            result = adfuller(data.dropna())
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        except Exception as e:
            logger.error(f"Error in stationarity test: {str(e)}")
            return None
    
    def find_best_arima_order(self, data: pd.Series) -> Tuple[int, int, int]:
        """Find best ARIMA order using AIC/BIC criteria"""
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        # Test different combinations
        for p in range(0, 4):
            for d in range(0, 3):
                for q in range(0, 4):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    @st.cache_data
    def fit_arima_model(_self, data: pd.Series, seasonal: bool = True, m: int = 365) -> Any:
        """Fit ARIMA model with caching and error handling"""
        try:
            with st.spinner("Training ARIMA model... This may take a few minutes."):
                # Find best order
                best_order = _self.find_best_arima_order(data)
                
                # Fit the model
                if seasonal and len(data) > 2 * m:
                    # Use SARIMAX for seasonal data
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    model = SARIMAX(
                        data, 
                        order=best_order,
                        seasonal_order=(1, 1, 1, m),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                else:
                    # Use regular ARIMA
                    model = ARIMA(data, order=best_order)
                
                fitted_model = model.fit()
                logger.info(f"ARIMA model fitted: {best_order}")
                return fitted_model
                
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            # Fallback to simple ARIMA(1,1,1)
            try:
                model = ARIMA(data, order=(1, 1, 1))
                fitted_model = model.fit()
                st.warning("Using fallback ARIMA(1,1,1) model")
                return fitted_model
            except Exception as e2:
                logger.error(f"Fallback model also failed: {str(e2)}")
                st.error(f"Error fitting model: {str(e)}")
                return None
    
    def generate_forecast(self, model: Any, n_periods: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Generate forecast with confidence intervals and metrics"""
        try:
            # Generate forecast using statsmodels
            forecast_result = model.get_forecast(steps=n_periods)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            # Create forecast dataframe
            last_date = self.data.index[-1]
            forecast_index = pd.date_range(
                last_date + pd.Timedelta(days=1), 
                periods=n_periods, 
                freq='D'
            )
            
            forecast_df = pd.DataFrame({
                'Date': forecast_index,
                'Forecast': forecast.values,
                'Lower_CI': conf_int.iloc[:, 0].values,
                'Upper_CI': conf_int.iloc[:, 1].values,
                'Confidence_Width': conf_int.iloc[:, 1].values - conf_int.iloc[:, 0].values
            })
            forecast_df.set_index('Date', inplace=True)
            
            # Calculate forecast metrics
            metrics = {
                'mean_forecast': np.mean(forecast),
                'std_forecast': np.std(forecast),
                'min_forecast': np.min(forecast),
                'max_forecast': np.max(forecast),
                'avg_confidence_width': np.mean(forecast_df['Confidence_Width'])
            }
            
            return forecast_df, metrics
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            st.error(f"Error generating forecast: {str(e)}")
            return None, None
    
    def calculate_model_performance(self, model: Any, data: pd.Series, test_size: int = 365) -> Dict[str, float]:
        """Calculate model performance metrics using walk-forward validation"""
        try:
            if len(data) < test_size + 30:
                test_size = max(30, len(data) // 4)
            
            train_data = data[:-test_size]
            test_data = data[-test_size:]
            
            # Fit model on training data
            best_order = self.find_best_arima_order(train_data)
            train_model = ARIMA(train_data, order=best_order)
            fitted_train_model = train_model.fit()
            
            # Generate predictions
            forecast_result = fitted_train_model.get_forecast(steps=len(test_data))
            predictions = forecast_result.predicted_mean
            
            # Calculate metrics
            mse = mean_squared_error(test_data, predictions)
            mae = mean_absolute_error(test_data, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(test_data, predictions)
            mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
            
            return {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance: {str(e)}")
            return {}
    
    def create_interactive_plots(self, forecast_df: pd.DataFrame, metrics: Dict[str, float]):
        """Create enhanced interactive Plotly visualizations"""
        
        # Main forecast plot with enhanced styling
        fig_main = go.Figure()
        
        # Historical data with gradient effect
        fig_main.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['Temp'],
            mode='lines',
            name='📈 Historical Temperature',
            line=dict(color='#667eea', width=2.5),
            hovertemplate='<b>📅 Date:</b> %{x}<br><b>🌡️ Temperature:</b> %{y:.1f}°C<extra></extra>',
            opacity=0.8
        ))
        
        # Forecast line with enhanced styling
        fig_main.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Forecast'],
            mode='lines+markers',
            name='🔮 Temperature Forecast',
            line=dict(color='#ff6b6b', width=4, dash='solid'),
            marker=dict(size=6, color='#ff6b6b', symbol='circle'),
            hovertemplate='<b>📅 Date:</b> %{x}<br><b>🔮 Forecast:</b> %{y:.1f}°C<extra></extra>'
        ))
        
        # Enhanced confidence interval
        fig_main.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Upper_CI'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_main.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Lower_CI'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 107, 107, 0.2)',
            name='📊 95% Confidence Interval',
            hovertemplate='<b>📅 Date:</b> %{x}<br><b>📊 Range:</b> %{y:.1f}°C<extra></extra>'
        ))
        
        # Add vertical line to separate historical from forecast
        last_historical_date = self.data.index[-1]
        fig_main.add_vline(
            x=last_historical_date,
            line_dash="dash",
            line_color="rgba(128, 128, 128, 0.5)",
            annotation_text="📍 Forecast Start",
            annotation_position="top"
        )
        
        fig_main.update_layout(
            title={
                'text': '🌡️ Temperature Forecast Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Inter, sans-serif', 'color': '#2c3e50'}
            },
            xaxis_title='📅 Date',
            yaxis_title='🌡️ Temperature (°C)',
            hovermode='x unified',
            template='plotly_white',
            height=650,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            ),
            plot_bgcolor='rgba(248, 249, 250, 0.8)',
            paper_bgcolor='white'
        )
        
        # Add range selector
        fig_main.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=30, label="30D", step="day", stepmode="backward"),
                        dict(count=90, label="90D", step="day", stepmode="backward"),
                        dict(count=365, label="1Y", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Enhanced additional analysis plots
        st.markdown("### 📊 Additional Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribution", "📅 Seasonal Patterns", "📊 Trend Analysis", "🎯 Forecast Details"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced temperature distribution
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=self.data['Temp'],
                    nbinsx=40,
                    name='Historical Distribution',
                    marker_color='rgba(102, 126, 234, 0.7)',
                    hovertemplate='<b>Temperature Range:</b> %{x}°C<br><b>Frequency:</b> %{y}<extra></extra>'
                ))
                
                # Add forecast distribution
                fig_dist.add_trace(go.Histogram(
                    x=forecast_df['Forecast'],
                    nbinsx=20,
                    name='Forecast Distribution',
                    marker_color='rgba(255, 107, 107, 0.7)',
                    hovertemplate='<b>Forecast Range:</b> %{x}°C<br><b>Frequency:</b> %{y}<extra></extra>'
                ))
                
                fig_dist.update_layout(
                    title='📊 Temperature Distribution Comparison',
                    xaxis_title='Temperature (°C)',
                    yaxis_title='Frequency',
                    template='plotly_white',
                    height=400,
                    barmode='overlay'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Box plot comparison
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=self.data['Temp'],
                    name='Historical',
                    marker_color='rgba(102, 126, 234, 0.7)',
                    boxpoints='outliers'
                ))
                fig_box.add_trace(go.Box(
                    y=forecast_df['Forecast'],
                    name='Forecast',
                    marker_color='rgba(255, 107, 107, 0.7)',
                    boxpoints='outliers'
                ))
                
                fig_box.update_layout(
                    title='📦 Temperature Range Comparison',
                    yaxis_title='Temperature (°C)',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced seasonal patterns
                monthly_avg = self.data.groupby(self.data.index.month)['Temp'].mean()
                monthly_std = self.data.groupby(self.data.index.month)['Temp'].std()
                
                fig_seasonal = go.Figure()
                fig_seasonal.add_trace(go.Bar(
                    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    y=monthly_avg.values,
                    error_y=dict(type='data', array=monthly_std.values),
                    name='Monthly Average',
                    marker_color='rgba(102, 126, 234, 0.8)',
                    hovertemplate='<b>Month:</b> %{x}<br><b>Avg Temp:</b> %{y:.1f}°C<extra></extra>'
                ))
                
                fig_seasonal.update_layout(
                    title='📅 Seasonal Temperature Patterns',
                    xaxis_title='Month',
                    yaxis_title='Average Temperature (°C)',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_seasonal, use_container_width=True)
            
            with col2:
                # Day of year analysis
                if len(self.data) > 365:
                    daily_avg = self.data.groupby(self.data.index.dayofyear)['Temp'].mean()
                    
                    fig_daily = go.Figure()
                    fig_daily.add_trace(go.Scatter(
                        x=daily_avg.index,
                        y=daily_avg.values,
                        mode='lines',
                        name='Daily Average',
                        line=dict(color='rgba(102, 126, 234, 0.8)', width=2),
                        hovertemplate='<b>Day of Year:</b> %{x}<br><b>Avg Temp:</b> %{y:.1f}°C<extra></extra>'
                    ))
                    
                    fig_daily.update_layout(
                        title='📈 Daily Temperature Cycle',
                        xaxis_title='Day of Year',
                        yaxis_title='Average Temperature (°C)',
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_daily, use_container_width=True)
                else:
                    st.info("📊 Need more than 1 year of data for daily cycle analysis")
        
        with tab3:
            # Trend analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Moving averages
                ma_7 = self.data['Temp'].rolling(window=7).mean()
                ma_30 = self.data['Temp'].rolling(window=30).mean()
                
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=self.data.index,
                    y=self.data['Temp'],
                    mode='lines',
                    name='Daily Temperature',
                    line=dict(color='rgba(102, 126, 234, 0.3)', width=1),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Temperature:</b> %{y:.1f}°C<extra></extra>'
                ))
                
                fig_trend.add_trace(go.Scatter(
                    x=ma_7.index,
                    y=ma_7.values,
                    mode='lines',
                    name='7-Day Moving Average',
                    line=dict(color='rgba(255, 107, 107, 0.8)', width=2)
                ))
                
                fig_trend.add_trace(go.Scatter(
                    x=ma_30.index,
                    y=ma_30.values,
                    mode='lines',
                    name='30-Day Moving Average',
                    line=dict(color='rgba(76, 175, 80, 0.8)', width=3)
                ))
                
                fig_trend.update_layout(
                    title='📈 Temperature Trends',
                    xaxis_title='Date',
                    yaxis_title='Temperature (°C)',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Year-over-year comparison (if multiple years available)
                if len(self.data) > 365:
                    yearly_avg = self.data.groupby(self.data.index.year)['Temp'].mean()
                    
                    fig_yearly = go.Figure()
                    fig_yearly.add_trace(go.Bar(
                        x=yearly_avg.index,
                        y=yearly_avg.values,
                        name='Yearly Average',
                        marker_color='rgba(102, 126, 234, 0.8)',
                        hovertemplate='<b>Year:</b> %{x}<br><b>Avg Temp:</b> %{y:.1f}°C<extra></extra>'
                    ))
                    
                    fig_yearly.update_layout(
                        title='📊 Year-over-Year Comparison',
                        xaxis_title='Year',
                        yaxis_title='Average Temperature (°C)',
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_yearly, use_container_width=True)
                else:
                    st.info("📊 Need multiple years of data for year-over-year analysis")
        
        with tab4:
            # Forecast details
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence interval width over time
                fig_conf = go.Figure()
                fig_conf.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['Confidence_Width'],
                    mode='lines+markers',
                    name='Confidence Width',
                    line=dict(color='rgba(255, 193, 7, 0.8)', width=3),
                    marker=dict(size=6),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Confidence Width:</b> %{y:.1f}°C<extra></extra>'
                ))
                
                fig_conf.update_layout(
                    title='🎯 Forecast Confidence Over Time',
                    xaxis_title='Date',
                    yaxis_title='Confidence Width (°C)',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_conf, use_container_width=True)
            
            with col2:
                # Forecast vs historical comparison
                recent_historical = self.data['Temp'].tail(len(forecast_df))
                
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Scatter(
                    x=list(range(len(recent_historical))),
                    y=recent_historical.values,
                    mode='lines',
                    name='Recent Historical Pattern',
                    line=dict(color='rgba(102, 126, 234, 0.8)', width=2)
                ))
                
                fig_compare.add_trace(go.Scatter(
                    x=list(range(len(forecast_df))),
                    y=forecast_df['Forecast'].values,
                    mode='lines',
                    name='Forecast Pattern',
                    line=dict(color='rgba(255, 107, 107, 0.8)', width=2)
                ))
                
                fig_compare.update_layout(
                    title='🔄 Pattern Comparison',
                    xaxis_title='Days',
                    yaxis_title='Temperature (°C)',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_compare, use_container_width=True)
    
    def create_forecast_summary(self, forecast_df: pd.DataFrame, metrics: Dict[str, float]):
        """Create enhanced forecast summary with key insights"""
        st.markdown("""
        <div class="forecast-summary fade-in">
            <h2 style="margin-bottom: 2rem; text-align: center; font-weight: 600;">📈 Forecast Summary & Key Insights</h2>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                <h3 style="color: white; margin-bottom: 0.5rem;">🌡️</h3>
                <h2 style="color: white; margin: 0;">{:.1f}°C</h2>
                <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">Average Forecast</p>
                <p style="color: rgba(255,255,255,0.6); margin: 0; font-size: 0.9rem;">±{:.1f}°C std dev</p>
            </div>
            """.format(metrics['mean_forecast'], metrics['std_forecast']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                <h3 style="color: white; margin-bottom: 0.5rem;">📏</h3>
                <h4 style="color: white; margin: 0; font-size: 1.2rem;">{:.1f}°C - {:.1f}°C</h4>
                <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">Temperature Range</p>
                <p style="color: rgba(255,255,255,0.6); margin: 0; font-size: 0.9rem;">{:.1f}°C spread</p>
            </div>
            """.format(
                metrics['min_forecast'], 
                metrics['max_forecast'],
                metrics['max_forecast'] - metrics['min_forecast']
            ), unsafe_allow_html=True)
        
        with col3:
            temp_trend = "Rising 📈" if forecast_df['Forecast'].iloc[-1] > forecast_df['Forecast'].iloc[0] else "Falling 📉"
            trend_delta = forecast_df['Forecast'].iloc[-1] - forecast_df['Forecast'].iloc[0]
            trend_color = "#4CAF50" if trend_delta > 0 else "#f44336"
            
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                <h3 style="color: white; margin-bottom: 0.5rem;">📊</h3>
                <h4 style="color: {}; margin: 0;">{}</h4>
                <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">Overall Trend</p>
                <p style="color: rgba(255,255,255,0.6); margin: 0; font-size: 0.9rem;">{:+.1f}°C change</p>
            </div>
            """.format(trend_color, temp_trend, trend_delta), unsafe_allow_html=True)
        
        with col4:
            confidence_width = metrics['avg_confidence_width'] / 2
            confidence_quality = "High" if confidence_width < 2 else "Medium" if confidence_width < 4 else "Low"
            confidence_color = "#4CAF50" if confidence_width < 2 else "#FF9800" if confidence_width < 4 else "#f44336"
            
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                <h3 style="color: white; margin-bottom: 0.5rem;">🎯</h3>
                <h4 style="color: {}; margin: 0;">±{:.1f}°C</h4>
                <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">Confidence Width</p>
                <p style="color: rgba(255,255,255,0.6); margin: 0; font-size: 0.9rem;">{} precision</p>
            </div>
            """.format(confidence_color, confidence_width, confidence_quality), unsafe_allow_html=True)
        
        # Add insights section
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
            <h4 style="color: white; margin-bottom: 1rem;">🔍 Key Insights</h4>
        """, unsafe_allow_html=True)
        
        # Generate insights based on data
        insights = []
        
        if abs(trend_delta) > 2:
            insights.append(f"📊 Significant temperature {'increase' if trend_delta > 0 else 'decrease'} of {abs(trend_delta):.1f}°C expected over the forecast period")
        
        if confidence_width < 2:
            insights.append("🎯 High forecast confidence - model predictions are very reliable")
        elif confidence_width > 4:
            insights.append("⚠️ Lower forecast confidence - consider using more historical data")
        
        if metrics['std_forecast'] < 2:
            insights.append("🌡️ Stable temperature pattern expected with low variability")
        else:
            insights.append("🌡️ Variable temperature pattern expected - prepare for fluctuations")
        
        for insight in insights:
            st.markdown(f'<p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;">{insight}</p>', unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    def export_results(self, forecast_df: pd.DataFrame):
        """Export forecast results to CSV"""
        try:
            # Prepare export data
            export_df = forecast_df.copy()
            export_df.reset_index(inplace=True)
            export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d')
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            # Create download button
            st.download_button(
                label="📥 Download Forecast Data (CSV)",
                data=csv_data,
                file_name=f"temperature_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
    
    def create_hero_section(self):
        """Create an attractive hero section"""
        st.markdown("""
        <div class="fade-in">
            <h1 class="main-header">🌡️ Temperature Forecast Dashboard</h1>
            <div style="text-align: center; margin-bottom: 2rem;">
                <p style="font-size: 1.2rem; color: #6c757d; font-weight: 400;">
                    Advanced Time Series Forecasting with ARIMA Models
                </p>
                <p style="color: #6c757d;">
                    📊 Analyze historical patterns • 🔮 Predict future temperatures • 📈 Validate model performance
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <h3 style="color: #667eea; margin-bottom: 0.5rem;">🎯</h3>
                <h4 style="margin-bottom: 0.5rem;">Accurate</h4>
                <p style="color: #6c757d; font-size: 0.9rem;">ARIMA-based forecasting with confidence intervals</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <h3 style="color: #667eea; margin-bottom: 0.5rem;">⚡</h3>
                <h4 style="margin-bottom: 0.5rem;">Fast</h4>
                <p style="color: #6c757d; font-size: 0.9rem;">Optimized algorithms with smart caching</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <h3 style="color: #667eea; margin-bottom: 0.5rem;">📱</h3>
                <h4 style="margin-bottom: 0.5rem;">Interactive</h4>
                <p style="color: #6c757d; font-size: 0.9rem;">Dynamic charts with hover details and zoom</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <h3 style="color: #667eea; margin-bottom: 0.5rem;">🔧</h3>
                <h4 style="margin-bottom: 0.5rem;">Flexible</h4>
                <p style="color: #6c757d; font-size: 0.9rem;">Upload custom data and configure parameters</p>
            </div>
            """, unsafe_allow_html=True)

    def create_enhanced_sidebar(self):
        """Create an enhanced sidebar with better organization"""
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   border-radius: 15px; margin-bottom: 2rem; color: white;">
            <h2 style="margin: 0; font-weight: 600;">⚙️ Control Panel</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Configure your forecast settings</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Input Section
        st.sidebar.markdown('<h3 class="sub-header" style="color: #667eea; border-left: 3px solid #667eea; padding-left: 0.5rem;">📁 Data Input</h3>', unsafe_allow_html=True)
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload Temperature Data",
            type=['csv'],
            help="Upload a CSV file with 'Date' and 'Temp' columns",
            label_visibility="collapsed"
        )
        
        if uploaded_file is None:
            st.sidebar.info("💡 Using default Melbourne temperature data")
        
        st.sidebar.markdown("---")
        
        # Model Configuration
        st.sidebar.markdown('<h3 class="sub-header" style="color: #667eea; border-left: 3px solid #667eea; padding-left: 0.5rem;">🤖 Model Settings</h3>', unsafe_allow_html=True)
        
        forecast_days = st.sidebar.slider(
            "📅 Forecast Period",
            min_value=7,
            max_value=365,
            value=30,
            step=7,
            help="Number of days to forecast into the future"
        )
        
        seasonal_toggle = st.sidebar.toggle(
            "🌊 Seasonal Modeling",
            value=True,
            help="Include seasonal patterns in the model"
        )
        
        show_performance = st.sidebar.toggle(
            "📊 Model Validation",
            value=True,
            help="Display model performance metrics"
        )
        
        advanced_options = st.sidebar.expander("🔧 Advanced Options")
        with advanced_options:
            confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
            auto_optimize = st.checkbox("Auto-optimize parameters", value=True)
            remove_outliers = st.checkbox("Remove outliers", value=True)
        
        st.sidebar.markdown("---")
        
        # Quick Actions
        st.sidebar.markdown('<h3 class="sub-header" style="color: #667eea; border-left: 3px solid #667eea; padding-left: 0.5rem;">⚡ Quick Actions</h3>', unsafe_allow_html=True)
        
        return uploaded_file, forecast_days, seasonal_toggle, show_performance

    def run_app(self):
        """Main application runner with enhanced UI"""
        
        # Hero Section
        self.create_hero_section()
        
        # Enhanced Sidebar
        uploaded_file, forecast_days, seasonal_toggle, show_performance = self.create_enhanced_sidebar()
        

        
        # Load data
        try:
            if uploaded_file is not None:
                self.data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
                st.sidebar.success("✅ Custom data loaded successfully!")
            else:
                self.data = self.load_data()
            
            if self.data is None:
                st.error("❌ Failed to load temperature data. Please check your data file.")
                st.stop()
            
            # Enhanced Data Overview
            st.markdown('<h2 class="sub-header">📋 Data Overview</h2>', unsafe_allow_html=True)
            
            # Create tabs for better organization
            tab1, tab2, tab3 = st.tabs(["📊 Summary Statistics", "📈 Data Preview", "🔍 Data Quality"])
            
            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("""
                    <div class="performance-card">
                        <h3 style="color: #667eea; margin-bottom: 0.5rem;">📝</h3>
                        <h2 style="margin: 0;">{:,}</h2>
                        <p style="color: #6c757d; margin: 0;">Total Records</p>
                    </div>
                    """.format(len(self.data)), unsafe_allow_html=True)
                
                with col2:
                    date_range = f"{self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}"
                    st.markdown("""
                    <div class="performance-card">
                        <h3 style="color: #667eea; margin-bottom: 0.5rem;">📅</h3>
                        <h4 style="margin: 0; font-size: 0.9rem;">{}</h4>
                        <p style="color: #6c757d; margin: 0;">Date Range</p>
                    </div>
                    """.format(date_range), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="performance-card">
                        <h3 style="color: #667eea; margin-bottom: 0.5rem;">🌡️</h3>
                        <h2 style="margin: 0;">{:.1f}°C</h2>
                        <p style="color: #6c757d; margin: 0;">Average Temperature</p>
                    </div>
                    """.format(self.data['Temp'].mean()), unsafe_allow_html=True)
                
                with col4:
                    temp_range = f"{self.data['Temp'].min():.1f}°C - {self.data['Temp'].max():.1f}°C"
                    st.markdown("""
                    <div class="performance-card">
                        <h3 style="color: #667eea; margin-bottom: 0.5rem;">📏</h3>
                        <h4 style="margin: 0; font-size: 0.9rem;">{}</h4>
                        <p style="color: #6c757d; margin: 0;">Temperature Range</p>
                    </div>
                    """.format(temp_range), unsafe_allow_html=True)
            
            with tab2:
                st.markdown("### 📋 Recent Data Points")
                st.dataframe(
                    self.data.tail(10).round(2),
                    use_container_width=True,
                    height=300
                )
                
                # Quick visualization
                fig_preview = px.line(
                    self.data.tail(100), 
                    y='Temp',
                    title="📈 Last 100 Days Temperature Trend",
                    labels={'Temp': 'Temperature (°C)', 'Date': 'Date'}
                )
                fig_preview.update_layout(
                    template='plotly_white',
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_preview, use_container_width=True)
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    missing_count = self.data['Temp'].isnull().sum()
                    missing_pct = (missing_count / len(self.data)) * 100
                    
                    if missing_count == 0:
                        st.success(f"✅ No missing values detected")
                    else:
                        st.warning(f"⚠️ {missing_count} missing values ({missing_pct:.1f}%)")
                    
                    # Data quality metrics
                    std_dev = self.data['Temp'].std()
                    st.info(f"📊 Standard Deviation: {std_dev:.2f}°C")
                
                with col2:
                    # Outlier detection
                    Q1 = self.data['Temp'].quantile(0.25)
                    Q3 = self.data['Temp'].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((self.data['Temp'] < (Q1 - 1.5 * IQR)) | 
                               (self.data['Temp'] > (Q3 + 1.5 * IQR))).sum()
                    
                    if outliers == 0:
                        st.success(f"✅ No outliers detected")
                    else:
                        st.warning(f"⚠️ {outliers} potential outliers detected")
                    
                    # Data completeness
                    completeness = ((len(self.data) - missing_count) / len(self.data)) * 100
                    st.info(f"📈 Data Completeness: {completeness:.1f}%")
            
            # Enhanced Stationarity Test Section
            st.markdown('<h2 class="sub-header">🔍 Statistical Analysis</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("🔍 Run Stationarity Test", type="secondary", use_container_width=True):
                    with st.spinner("Performing stationarity test..."):
                        stationarity_result = self.perform_stationarity_test(self.data['Temp'])
                        st.session_state.stationarity_result = stationarity_result
            
            with col2:
                if hasattr(st.session_state, 'stationarity_result') and st.session_state.stationarity_result:
                    result = st.session_state.stationarity_result
                    
                    # Create beautiful result display
                    if result['is_stationary']:
                        st.markdown("""
                        <div class="metric-card" style="border-left: 4px solid #28a745;">
                            <h4 style="color: #28a745; margin-bottom: 1rem;">✅ Data is Stationary</h4>
                            <div style="display: flex; justify-content: space-between;">
                                <div>
                                    <p><strong>ADF Statistic:</strong> {:.4f}</p>
                                    <p><strong>P-value:</strong> {:.6f}</p>
                                </div>
                                <div>
                                    <p style="color: #28a745; font-weight: 600;">Ready for ARIMA modeling</p>
                                </div>
                            </div>
                        </div>
                        """.format(result['adf_statistic'], result['p_value']), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="metric-card" style="border-left: 4px solid #ffc107;">
                            <h4 style="color: #ffc107; margin-bottom: 1rem;">⚠️ Data is Non-Stationary</h4>
                            <div style="display: flex; justify-content: space-between;">
                                <div>
                                    <p><strong>ADF Statistic:</strong> {:.4f}</p>
                                    <p><strong>P-value:</strong> {:.6f}</p>
                                </div>
                                <div>
                                    <p style="color: #ffc107; font-weight: 600;">Will apply differencing</p>
                                </div>
                            </div>
                        </div>
                        """.format(result['adf_statistic'], result['p_value']), unsafe_allow_html=True)
            
            # Enhanced Forecast Generation Section
            st.markdown('<h2 class="sub-header">🚀 Generate Forecast</h2>', unsafe_allow_html=True)
            
            # Create a prominent forecast button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                forecast_button = st.button(
                    "🚀 Generate Temperature Forecast", 
                    type="primary", 
                    use_container_width=True,
                    help="Click to start the forecasting process"
                )
            
            if forecast_button:
                try:
                    # Fit model
                    self.model = self.fit_arima_model(
                        self.data['Temp'], 
                        seasonal=seasonal_toggle,
                        m=365 if seasonal_toggle else 1
                    )
                    
                    if self.model is None:
                        st.error("❌ Failed to fit ARIMA model.")
                        st.stop()
                    
                    # Generate forecast
                    forecast_df, forecast_metrics = self.generate_forecast(self.model, forecast_days)
                    
                    if forecast_df is None:
                        st.error("❌ Failed to generate forecast.")
                        st.stop()
                    
                    # Success message with animation
                    st.markdown("""
                    <div class="fade-in" style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                                border-radius: 15px; color: white; margin: 2rem 0;">
                        <h2 style="margin: 0;">🎉 Forecast Generated Successfully!</h2>
                        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Your temperature forecast is ready for analysis</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Enhanced Model Information
                    st.markdown('<h2 class="sub-header">🤖 Model Information</h2>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        <div class="performance-card">
                            <h4 style="color: #667eea; margin-bottom: 1rem;">📊 Model Type</h4>
                            <h3 style="margin: 0;">ARIMA{}</h3>
                            <p style="color: #6c757d; margin: 0.5rem 0 0 0;">Auto-selected order</p>
                        </div>
                        """.format(self.model.order), unsafe_allow_html=True)
                    
                    with col2:
                        aic_value = self.model.aic()
                        st.markdown("""
                        <div class="performance-card">
                            <h4 style="color: #667eea; margin-bottom: 1rem;">📈 AIC Score</h4>
                            <h3 style="margin: 0;">{:.2f}</h3>
                            <p style="color: #6c757d; margin: 0.5rem 0 0 0;">Model fit quality</p>
                        </div>
                        """.format(aic_value), unsafe_allow_html=True)
                    
                    with col3:
                        bic_value = self.model.bic()
                        st.markdown("""
                        <div class="performance-card">
                            <h4 style="color: #667eea; margin-bottom: 1rem;">📊 BIC Score</h4>
                            <h3 style="margin: 0;">{:.2f}</h3>
                            <p style="color: #6c757d; margin: 0.5rem 0 0 0;">Model complexity</p>
                        </div>
                        """.format(bic_value), unsafe_allow_html=True)
                    
                    # Forecast summary
                    self.create_forecast_summary(forecast_df, forecast_metrics)
                    
                    # Enhanced Interactive plots
                    st.markdown('<h2 class="sub-header">📈 Forecast Visualization</h2>', unsafe_allow_html=True)
                    self.create_interactive_plots(forecast_df, forecast_metrics)
                    
                    # Enhanced Model Performance
                    if show_performance:
                        st.markdown('<h2 class="sub-header">🎯 Model Performance</h2>', unsafe_allow_html=True)
                        
                        with st.spinner("🔄 Calculating model performance metrics..."):
                            performance_metrics = self.calculate_model_performance(self.model, self.data['Temp'])
                            
                            if performance_metrics:
                                # Create performance dashboard
                                col1, col2, col3, col4, col5 = st.columns(5)
                                
                                with col1:
                                    rmse_val = performance_metrics.get('RMSE', 0)
                                    st.markdown("""
                                    <div class="performance-card">
                                        <h4 style="color: #667eea; margin-bottom: 0.5rem;">📏 RMSE</h4>
                                        <h2 style="margin: 0; color: #2c3e50;">{:.2f}</h2>
                                        <p style="color: #6c757d; margin: 0; font-size: 0.8rem;">Root Mean Square Error</p>
                                    </div>
                                    """.format(rmse_val), unsafe_allow_html=True)
                                
                                with col2:
                                    mae_val = performance_metrics.get('MAE', 0)
                                    st.markdown("""
                                    <div class="performance-card">
                                        <h4 style="color: #667eea; margin-bottom: 0.5rem;">📊 MAE</h4>
                                        <h2 style="margin: 0; color: #2c3e50;">{:.2f}</h2>
                                        <p style="color: #6c757d; margin: 0; font-size: 0.8rem;">Mean Absolute Error</p>
                                    </div>
                                    """.format(mae_val), unsafe_allow_html=True)
                                
                                with col3:
                                    r2_val = performance_metrics.get('R²', 0)
                                    r2_color = "#28a745" if r2_val > 0.7 else "#ffc107" if r2_val > 0.5 else "#dc3545"
                                    st.markdown("""
                                    <div class="performance-card">
                                        <h4 style="color: #667eea; margin-bottom: 0.5rem;">🎯 R²</h4>
                                        <h2 style="margin: 0; color: {};">{:.3f}</h2>
                                        <p style="color: #6c757d; margin: 0; font-size: 0.8rem;">Coefficient of Determination</p>
                                    </div>
                                    """.format(r2_color, r2_val), unsafe_allow_html=True)
                                
                                with col4:
                                    mape_val = performance_metrics.get('MAPE', 0)
                                    mape_color = "#28a745" if mape_val < 10 else "#ffc107" if mape_val < 20 else "#dc3545"
                                    st.markdown("""
                                    <div class="performance-card">
                                        <h4 style="color: #667eea; margin-bottom: 0.5rem;">📈 MAPE</h4>
                                        <h2 style="margin: 0; color: {};">{:.1f}%</h2>
                                        <p style="color: #6c757d; margin: 0; font-size: 0.8rem;">Mean Absolute Percentage Error</p>
                                    </div>
                                    """.format(mape_color, mape_val), unsafe_allow_html=True)
                                
                                with col5:
                                    accuracy = max(0, 100 - performance_metrics.get('MAPE', 100))
                                    acc_color = "#28a745" if accuracy > 80 else "#ffc107" if accuracy > 60 else "#dc3545"
                                    st.markdown("""
                                    <div class="performance-card">
                                        <h4 style="color: #667eea; margin-bottom: 0.5rem;">✅ Accuracy</h4>
                                        <h2 style="margin: 0; color: {};">{:.1f}%</h2>
                                        <p style="color: #6c757d; margin: 0; font-size: 0.8rem;">Model Accuracy</p>
                                    </div>
                                    """.format(acc_color, accuracy), unsafe_allow_html=True)
                                
                                # Performance interpretation
                                st.markdown("### 📋 Performance Interpretation")
                                
                                if accuracy > 80:
                                    st.success("🎉 Excellent model performance! The forecast is highly reliable.")
                                elif accuracy > 60:
                                    st.warning("⚠️ Good model performance. The forecast is reasonably reliable.")
                                else:
                                    st.error("❌ Model performance could be improved. Consider using more data or different parameters.")
                    
                    # Enhanced Export and Data View
                    st.markdown('<h2 class="sub-header">💾 Export & Data</h2>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("""
                        <div class="metric-card" style="text-align: center;">
                            <h4 style="color: #667eea; margin-bottom: 1rem;">📥 Download Results</h4>
                            <p style="color: #6c757d; margin-bottom: 1rem;">Export your forecast data for further analysis</p>
                        </div>
                        """, unsafe_allow_html=True)
                        self.export_results(forecast_df)
                    
                    with col2:
                        st.markdown("""
                        <div class="metric-card" style="text-align: center;">
                            <h4 style="color: #667eea; margin-bottom: 1rem;">📊 Data Summary</h4>
                            <p style="color: #6c757d;">Forecast Period: <strong>{} days</strong></p>
                            <p style="color: #6c757d;">Data Points: <strong>{:,}</strong></p>
                            <p style="color: #6c757d;">Date Range: <strong>{} to {}</strong></p>
                        </div>
                        """.format(
                            forecast_days,
                            len(forecast_df),
                            forecast_df.index[0].strftime('%Y-%m-%d'),
                            forecast_df.index[-1].strftime('%Y-%m-%d')
                        ), unsafe_allow_html=True)
                    
                    # Enhanced Forecast table
                    with st.expander("📋 View Detailed Forecast Data", expanded=False):
                        st.markdown("### 📊 Complete Forecast Dataset")
                        
                        # Add search and filter options
                        col1, col2 = st.columns(2)
                        with col1:
                            show_confidence = st.checkbox("Show Confidence Intervals", value=True)
                        with col2:
                            round_decimals = st.selectbox("Decimal Places", [1, 2, 3], index=1)
                        
                        # Display filtered data
                        display_df = forecast_df.copy()
                        if not show_confidence:
                            display_df = display_df[['Forecast']]
                        
                        st.dataframe(
                            display_df.round(round_decimals),
                            use_container_width=True,
                            height=400
                        )
                
                except Exception as e:
                    st.error(f"❌ An error occurred during forecasting: {str(e)}")
                    st.error("Please check your data and try again.")
                    logger.error(f"Forecasting error: {traceback.format_exc()}")
        
        except Exception as e:
            st.error(f"❌ Application error: {str(e)}")
            logger.error(f"Application error: {traceback.format_exc()}")
        
        # Enhanced Footer
        st.markdown("""
        <div class="footer">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h3 style="margin: 0; font-weight: 600;">🌡️ Temperature Forecast Dashboard</h3>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Advanced Time Series Forecasting with ARIMA Models</p>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; font-weight: 500;">Built with ❤️ using:</p>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Streamlit • Plotly • Statsmodels • Scikit-learn</p>
                </div>
            </div>
            <hr style="margin: 1.5rem 0; opacity: 0.3;">
            <div style="text-align: center;">
                <p style="margin: 0; opacity: 0.8;">
                    💡 <strong>Pro Tip:</strong> For production use, ensure your data is clean and representative of future conditions.
                </p>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.6; font-size: 0.9rem;">
                    © 2024 Temperature Forecast Dashboard. Made for accurate weather predictions.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Application entry point
if __name__ == "__main__":
    app = TemperatureForecastApp()
    app.run_app()
