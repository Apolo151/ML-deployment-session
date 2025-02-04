import pickle
import numpy as np
import streamlit as st

# Load the pre-trained model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title("Iris Flower Prediction App")

# Sidebar for input features
st.sidebar.header("Input Features")
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 1.3)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)
    # Create a numpy array of the features
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return features

# Get user input from the sidebar
input_features = user_input_features()

# Perform prediction
prediction = model.predict(input_features)
prediction_proba = model.predict_proba(input_features)

# Map the prediction to the iris target names
iris_target_names = ['setosa', 'versicolor', 'virginica']
predicted_class = iris_target_names[prediction[0]]

# Display the results on the main page
st.write("## Prediction")
st.write(f"Predicted Iris Flower Type: **{predicted_class}**")

st.write("## Prediction Probabilities")
st.write(prediction_proba)
