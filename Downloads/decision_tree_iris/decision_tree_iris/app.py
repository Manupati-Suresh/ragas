
import streamlit as st
import pandas as pd
import pickle

st.title("🌸 Iris Flower Species Prediction")

st.markdown("""
This app predicts the species of an Iris flower based on its physical measurements.
Adjust the sliders below to input the flower's dimensions and click **Predict Species** to see the result.

**Species Types:**
- 🌺 Iris-setosa
- 🌸 Iris-versicolor  
- 🌼 Iris-virginica
""")

st.divider()

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        with open("decision_tree_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'decision_tree_model.pkl' is in the repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()

# Input sliders in sidebar
st.sidebar.header("🌿 Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, help="Length of the sepal in centimeters")
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, help="Width of the sepal in centimeters")
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4, help="Length of the petal in centimeters")
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2, help="Width of the petal in centimeters")

st.sidebar.markdown("---")
st.sidebar.markdown("**About:** This model uses a Decision Tree classifier trained on the famous Iris dataset.")

# Create input dataframe
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])

# Display current input values
st.subheader("Current Input Values:")
st.write(input_data)

if st.button("Predict Species", type="primary"):
    try:
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0].max()
        
        st.success(f"🌼 Predicted Species: **{prediction}**")
        st.info(f"Confidence: {confidence:.2%}")
        
        # Show all probabilities
        classes = model.classes_
        probabilities = model.predict_proba(input_data)[0]
        
        st.subheader("Prediction Probabilities:")
        prob_df = pd.DataFrame({
            'Species': classes,
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)
        
        st.bar_chart(prob_df.set_index('Species'))
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
