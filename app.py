# app.py

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Load the pre-trained model
try:
    with open("iris_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model file not found. Please run the training script first.")

# Initialize FastAPI
app = FastAPI(title="Iris Prediction API")

# Define the request body using Pydantic
class IrisFeatures(BaseModel):
    features: List[float]  # Expecting 4 features (sepal length, sepal width, petal length, petal width)

# Define the prediction endpoint
@app.post("/predict")
async def predict(iris: IrisFeatures):
    # Validate that we received exactly 4 features
    if len(iris.features) != 4:
        raise HTTPException(status_code=400, detail="Exactly 4 features are required.")
    
    # Convert input to numpy array and reshape for the model
    input_features = np.array(iris.features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)
    
    # Map the numeric prediction to iris species names
    iris_target_names = ['setosa', 'versicolor', 'virginica']
    predicted_species = iris_target_names[prediction[0]]
    
    return {
        "predicted_species": predicted_species,
        "prediction_probability": prediction_proba.tolist()[0]
    }

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Welcome to the Iris Prediction API. Use the /predict endpoint to get predictions."}
