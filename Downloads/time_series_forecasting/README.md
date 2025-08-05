# 🌡️ Temperature Forecast Dashboard

A production-ready Streamlit application for time series forecasting of temperature data using ARIMA models.

## Features

### 🚀 Core Functionality
- **Interactive Time Series Forecasting** using ARIMA models
- **Seasonal Pattern Detection** with configurable seasonal modeling
- **Real-time Model Performance Metrics** (RMSE, MAE, R², MAPE)
- **Confidence Intervals** for forecast uncertainty quantification
- **Data Validation & Cleaning** with outlier detection and missing value handling

### 📊 Visualizations
- **Interactive Plotly Charts** with hover details and zoom capabilities
- **Temperature Distribution Analysis** 
- **Seasonal Pattern Visualization**
- **Forecast vs Historical Comparison**
- **95% Confidence Interval Bands**

### 🛠️ Advanced Features
- **Custom Data Upload** support for CSV files
- **Stationarity Testing** with Augmented Dickey-Fuller test
- **Model Performance Validation** using walk-forward validation
- **Export Functionality** for forecast results
- **Responsive Design** with mobile-friendly interface
- **Error Handling & Logging** for production reliability

### 🎯 Production Enhancements
- **Caching** for improved performance
- **Input Validation** and error handling
- **Memory Optimization** for large datasets
- **Configurable Parameters** via sidebar controls
- **Professional UI/UX** with custom styling

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run temperature_forecast.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

### 1. Data Input
- **Default Data**: Uses the included `daily-min-temperatures.csv` file
- **Custom Upload**: Upload your own CSV file with 'Date' and 'Temp' columns

### 2. Configuration
- **Forecast Period**: Choose 7-365 days for prediction
- **Seasonal Modeling**: Enable/disable seasonal pattern detection
- **Performance Metrics**: Toggle model validation display

### 3. Analysis
- Click "🚀 Generate Forecast" to run the analysis
- View interactive charts and performance metrics
- Export results as CSV for further analysis

### 4. Model Validation
- **Stationarity Test**: Check if your data needs differencing
- **Performance Metrics**: Evaluate model accuracy
- **Confidence Intervals**: Understand forecast uncertainty

## Data Format

Your CSV file should have the following structure:
```csv
Date,Temp
1981-01-01,20.7
1981-01-02,17.9
1981-01-03,18.8
```

- **Date**: Date column in YYYY-MM-DD format
- **Temp**: Temperature values (numeric)

## Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "temperature_forecast.py"]
```

### Heroku Deployment
1. Create a `Procfile`:
```
web: streamlit run temperature_forecast.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy using Heroku CLI or GitHub integration

## Model Details

### ARIMA Implementation
- **Auto ARIMA**: Automatically selects optimal (p,d,q) parameters
- **Seasonal Support**: Handles yearly seasonal patterns (m=365)
- **Stepwise Selection**: Efficient parameter optimization
- **Error Handling**: Robust model fitting with fallback options

### Performance Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error
- **Accuracy**: 100% - MAPE

### Validation Method
- **Walk-Forward Validation**: Tests model on recent historical data
- **Train/Test Split**: Uses last year of data for validation
- **Cross-Validation**: Ensures model generalization

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce forecast period for large datasets
3. **Model Fitting Failures**: Check data quality and stationarity

### Performance Optimization
- **Data Caching**: Automatic caching of loaded data and models
- **Efficient Plotting**: Uses Plotly for fast interactive charts
- **Memory Management**: Optimized for large time series datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the error logs in the application
3. Ensure your data format matches the requirements

---

**Built with ❤️ using Streamlit, Plotly, and ARIMA modeling**