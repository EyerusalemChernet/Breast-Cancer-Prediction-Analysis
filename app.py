from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize the FastAPI app
app = FastAPI()

# Load the dataset and preprocess
file_path = "breast-cancer.csv"  # Update path as needed
df = pd.read_csv(file_path)

# Drop the 'id' column and map diagnosis to binary
df.drop(columns=['id'], inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Split the data into features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler for later use
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Load the model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
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
    # Prepare the input data for prediction
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

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict the diagnosis
    prediction = model.predict(input_data_scaled)
    prediction_prob = model.predict_proba(input_data_scaled)[:, 1]

    # Return the prediction and probability
    return {
        "prediction": "Malignant" if prediction[0] == 1 else "Benign",
        "probability": prediction_prob[0].tolist()
    }

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "Breast Cancer Prediction API is running!"}
