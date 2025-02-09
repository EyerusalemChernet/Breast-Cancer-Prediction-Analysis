📌 Breast Cancer Prediction and Analysis
🏥 Early Detection of Breast Cancer Using Machine Learning


📖 Table of Contents

Introduction
Dataset
Project Workflow
Technologies Used
How to Run the Project
API Usage
Results
Future Improvements
Author
Acknowledgments

📌 Introduction
Breast cancer is one of the most prevalent cancers affecting women worldwide. Early detection can increase survival rates significantly. This project leverages machine learning to classify tumors as benign or malignant based on medical imaging features.

This repository includes:
✅ Data Preprocessing & Visualization
✅ Machine Learning Model Training & Evaluation
✅ Deployment of the Model as an API using FastAPI
✅ Real-time Predictions via a REST API

📊 Dataset
Source: Kaggle - Breast Cancer Dataset
Size: 569 instances, 32 attributes
Target Variable:
0 → Benign (Non-cancerous)
1 → Malignant (Cancerous)
Features: Radius, Texture, Perimeter, Area, Compactness, Concavity, etc.
🛠 Project Workflow
1️⃣ Data Cleaning & Preprocessing
2️⃣ Exploratory Data Analysis (EDA)
3️⃣ Feature Engineering & Scaling
4️⃣ Model Training (Logistic Regression)
5️⃣ Model Evaluation (Accuracy, Confusion Matrix, ROC Curve)
6️⃣ Deployment using FastAPI & Render
7️⃣ Real-time Predictions using API Requests

💻 Technologies Used
Python 🐍
Pandas, NumPy (Data Processing)
Seaborn, Matplotlib (Data Visualization)
Scikit-Learn (Machine Learning)
FastAPI (API Deployment)
Uvicorn (Running the API)
Render (Cloud Deployment)
🚀 How to Run the Project
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/Jerusauce/Breast-Cancer-Prediction-Analysis  
cd Breast-Cancer-Prediction-Analysis  
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt  
3️⃣ Train the Model & Save It
Run the breastcancer.ipynb notebook to:
✅ Preprocess the dataset
✅ Train the Logistic Regression model
✅ Save the trained model (model.pkl) and scaler (scaler.pkl)

4️⃣ Start the API
Run the FastAPI server using Uvicorn:

bash
Copy
Edit
uvicorn app:app --host 0.0.0.0 --port 8000  
This will start the API locally on http://127.0.0.1:8000/

📡 API Usage
1️⃣ Health Check Endpoint
Check if the API is running:

bash
Copy
Edit
curl -X GET http://127.0.0.1:8000/  
Response:

json
Copy
Edit
{"message": "Breast Cancer Prediction API is running!"}
2️⃣ Prediction Endpoint
Send patient feature data to get a prediction:

bash
Copy
Edit
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{  
  "radius_mean": 17.99,  
  "texture_mean": 10.38,  
  "perimeter_mean": 122.8,  
  "area_mean": 1001.0,  
  "smoothness_mean": 0.11840,  
  "compactness_mean": 0.27760,  
  "concavity_mean": 0.3001,  
  "concave_points_mean": 0.1471,  
  "symmetry_mean": 0.2419,  
  "fractal_dimension_mean": 0.07871,  
  "radius_worst": 25.38,  
  "texture_worst": 17.33,  
  "perimeter_worst": 184.6,  
  "area_worst": 2019.0,  
  "smoothness_worst": 0.1622,  
  "compactness_worst": 0.6656,  
  "concavity_worst": 0.7119,  
  "concave_points_worst": 0.2654,  
  "symmetry_worst": 0.4601,  
  "fractal_dimension_worst": 0.11890  
}'
Response Example:

json
Copy
Edit
{
  "prediction": "Malignant",
  "probability": 0.98
}
📈 Results
Accuracy: 97% on test data
High precision & recall, making the model reliable for medical diagnosis
Fast real-time predictions via the API
🔮 Future Improvements
📌 Try Deep Learning Models (e.g., CNNs for medical images)
📌 Improve Feature Engineering to enhance model performance
📌 Deploy as a Web App for user-friendly interaction
👨‍💻 Author
Eyerusalem Chernet

📧 Email: Jerusalemroronoa@gmail.com

🙌 Acknowledgments
Thanks to Kaggle for the dataset and FastAPI for making API deployment easy!
