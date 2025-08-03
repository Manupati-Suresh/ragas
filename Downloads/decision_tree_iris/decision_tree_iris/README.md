# 🌸 Iris Flower Species Prediction App

A Streamlit web application that predicts the species of Iris flowers based on their physical measurements using a Decision Tree classifier.

## 🌟 Features

- Interactive sliders for inputting flower measurements
- Real-time predictions with confidence scores
- Probability distribution visualization
- Clean, user-friendly interface
- Responsive design with sidebar controls

## 🚀 Live Demo

Deploy this app on Streamlit Cloud: [Your App URL Here]

## 📊 Dataset

This app uses the famous Iris dataset, which contains measurements for 150 iris flowers from three different species:
- 🌺 Iris-setosa
- 🌸 Iris-versicolor  
- 🌼 Iris-virginica

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/iris-prediction-app.git
cd iris-prediction-app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── train_model.py           # Script to train the Decision Tree model
├── decision_tree_model.pkl  # Trained model file
├── requirements.txt         # Python dependencies
├── iris.data               # Iris dataset
├── iris.names              # Dataset description
└── README.md               # Project documentation
```

## 🔧 Model Details

- **Algorithm**: Decision Tree Classifier
- **Features**: Sepal length, sepal width, petal length, petal width
- **Target**: Iris species (setosa, versicolor, virginica)
- **Accuracy**: ~95% on test data

## 🌐 Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

## 📝 Usage

1. Adjust the sliders in the sidebar to input flower measurements
2. Click "Predict Species" to get the prediction
3. View the confidence score and probability distribution
4. Experiment with different measurements to see how predictions change

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).