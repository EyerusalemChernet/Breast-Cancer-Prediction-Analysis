from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize the FastAPI app
app = FastAPI()

# File paths
file_path = "breast-cancer.csv"
model_path = "model.pkl"
scaler_path = "scaler.pkl"

# Ensure the CSV file exists before proceeding
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file '{file_path}' not found!")

# Load the trained model and scaler
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model or scaler file missing! Train and save them first.")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Pydantic model for input validation
class InputData(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

# Define an endpoint for predictions
@app.post("/predict")
def predict(data: InputData):
    try:
        # Prepare input for prediction
        input_data = np.array([[
            data.radius_mean,
            data.texture_mean,
            data.perimeter_mean,
            data.area_mean,
            data.smoothness_mean,
            data.compactness_mean,
            data.concavity_mean,
            data.concave_points_mean,
            data.symmetry_mean,
            data.fractal_dimension_mean,
            data.radius_se,
            data.texture_se,
            data.perimeter_se,
            data.area_se,
            data.smoothness_se,
            data.compactness_se,
            data.concavity_se,
            data.concave_points_se,
            data.symmetry_se,
            data.fractal_dimension_se,
            data.radius_worst,
            data.texture_worst,
            data.perimeter_worst,
            data.area_worst,
            data.smoothness_worst,
            data.compactness_worst,
            data.concavity_worst,
            data.concave_points_worst,
            data.symmetry_worst,
            data.fractal_dimension_worst
        ]])

        # Scale input data
        input_data_scaled = scaler.transform(input_data)

        # Predict diagnosis
        prediction = model.predict(input_data_scaled)
        prediction_prob = model.predict_proba(input_data_scaled)[:, 1]

        return {
            "prediction": "Malignant" if prediction[0] == 1 else "Benign",
            "probability": prediction_prob[0].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "Breast Cancer Prediction API is running!"}

# Run the app when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
