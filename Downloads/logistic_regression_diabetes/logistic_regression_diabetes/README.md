# 🩺 Diabetes Prediction App

A machine learning web application that predicts diabetes risk using logistic regression. Built with Streamlit and scikit-learn.

## 📋 Overview

This application uses the Pima Indians Diabetes Database to predict whether a person has diabetes based on diagnostic measurements. The model achieves reliable predictions using logistic regression with standardized features.

## 🚀 Features

- **Interactive Web Interface**: Easy-to-use sliders for input parameters
- **Real-time Predictions**: Instant diabetes risk assessment
- **Probability Scores**: Shows prediction confidence
- **Data Preprocessing**: Handles missing values and feature scaling
- **Responsive Design**: Clean, medical-themed UI

## 📊 Dataset

The model is trained on the Pima Indians Diabetes Database with 8 features:
- **Pregnancies**: Number of pregnancies
- **Glucose**: Plasma glucose concentration
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **Diabetes Pedigree Function**: Diabetes pedigree function
- **Age**: Age in years

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Manupati-Suresh/ragas.git
   cd logistic_regression_diabetes
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (optional - pre-trained models included):
   ```bash
   python train_model.py
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📦 Dependencies

```
streamlit==1.28.1
pandas==2.0.3
scikit-learn==1.3.0
pickle-mixin==1.0.2
```

## 🔧 Usage

1. **Launch the app**: Run `streamlit run app.py`
2. **Adjust parameters**: Use the sliders to input patient data
3. **Get prediction**: Click "Predict" to see diabetes risk assessment
4. **Interpret results**: View prediction label and probability score

### Input Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Pregnancies | 0-17 | 3 | Number of times pregnant |
| Glucose | 0-200 | 120 | Plasma glucose concentration |
| Blood Pressure | 0-122 | 70 | Diastolic blood pressure |
| Skin Thickness | 0-99 | 20 | Triceps skin fold thickness |
| Insulin | 0-846 | 79 | 2-Hour serum insulin |
| BMI | 0.0-67.1 | 32.0 | Body mass index |
| Pedigree Function | 0.0-2.5 | 0.47 | Diabetes pedigree function |
| Age | 21-90 | 33 | Age in years |

## 🧠 Model Details

- **Algorithm**: Logistic Regression
- **Preprocessing**: StandardScaler for feature normalization
- **Missing Values**: Replaced with median values for key features
- **Train/Test Split**: 80/20 split with random_state=42
- **Solver**: liblinear (suitable for small datasets)

## 📁 Project Structure

```
logistic_regression_diabetes/
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── diabetes.csv           # Dataset
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── logistic_model.pkl    # Trained model (generated)
└── scaler.pkl           # Fitted scaler (generated)
```

## 🔍 Model Performance

The logistic regression model provides:
- Binary classification (Diabetic/Non-Diabetic)
- Probability scores for prediction confidence
- Standardized feature scaling for improved performance
- Robust handling of missing values

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment Options
- **Streamlit Cloud**: Connect your GitHub repository
- **Heroku**: Use the provided requirements.txt
- **Docker**: Containerize the application

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Manupati Suresh**
- GitHub: [@Manupati-Suresh](https://github.com/Manupati-Suresh)

## 🙏 Acknowledgments

- Pima Indians Diabetes Database from UCI Machine Learning Repository
- Streamlit for the amazing web framework
- scikit-learn for machine learning tools

## ⚠️ Disclaimer

This application is for educational and demonstration purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.