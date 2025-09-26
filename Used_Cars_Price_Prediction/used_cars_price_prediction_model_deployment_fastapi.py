from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Initialize FastAPI app with project-specific metadata
app = FastAPI(
    title="Car Price Prediction API",
    description=(
        "This API deploys a machine learning model to predict car prices based on multiple features "
        "like brand, model year, mileage, fuel type, transmission, color, and engine details. "
        "The service integrates preprocessing (encoding and scaling) with a trained deep learning model "
        "for accurate real-time predictions."
    ),
    version="1.0.0"
)

# Global variables for loaded artifacts
model = None
label_loaded = None
emb_label_loaded = None
scale_loaded = None
target_scale_loaded = None

# Load trained model and preprocessing objects
try:
    # --- Load Model ---
    model = load_model("cars.keras")  # Load saved Keras model
    
    # --- Load Preprocessors ---
    with open("label.pkl", "rb") as file:      # Load LabelEncoders for binary-like features (e.g., fuel_type)
        label_loaded = pickle.load(file)
    with open("emb_label.pkl", "rb") as file:  # Load LabelEncoders for embedding features (e.g., brand, ext_col)
        emb_label_loaded = pickle.load(file)
    with open("scale.pkl", "rb") as file:      # Load StandardScaler for numerical FEATURES (X columns)
        scale_loaded = pickle.load(file)
    # WARNING: Ensure the filename 'target_scale' matches how the scaler was saved.
    with open("target_scale", "rb") as file:    # Load StandardScaler for the TARGET variable (price)
        target_scale_loaded = pickle.load(file)
        
    print("✅ All artifacts loaded.")

except FileNotFoundError as e:
    # Important: In a production environment, failure to load artifacts should stop the service.
    print(f"Some file is not Found: {e}")

# Input schema for request body (features required for prediction)
class CarsInputs(BaseModel):
    brand: str
    model_year: int
    milage: float
    fuel_type: str
    transmission: str
    ext_col: str
    int_col: str
    accident: str
    horse_power: float
    engine_displacement: float
    engine_cylinder: float # Note: engine_cylinder was cast to int in training

# Define response schema for consistent API output
class PredictionResponse(BaseModel):
    predicted_value: str
    cars_features: dict

# Feature groups
label_cols = ["fuel_type", "accident"]  # Categorical features (LabelEncoder)
emb_label_cols = ["brand", "transmission", "ext_col", "int_col"]  # Categorical features (Embedding Encoders)
num_cols_scale = ["model_year", "milage", "horse_power", "engine_displacement", "engine_cylinder"]  # Numerical features (Scaler)
target_scale = "price"

# Function to preprocess input before prediction
def preprocess_input(data: dict):
    """
    Preprocess raw car input features into a format suitable for model prediction.

    Args:
        data (dict): Dictionary containing car features from the request.

    Returns:
        list: Transformed input features list for the Keras Model (embedding inputs + numerical input).
    """
    try:
        df = pd.DataFrame([data])  # Convert input dict to DataFrame (1 row)

        # Apply Label Encoding for specific categorical variables
        # WARNING: This code is susceptible to crashes (ValueError) if an unseen category is provided.
        for col in label_cols:
            le = label_loaded[col]
            df[col] = le.transform(df[col])

        # Apply Embedding/Ordinal Encoding for other categorical variables
        # WARNING: This code is susceptible to crashes (ValueError) if an unseen category is provided.
        for col in emb_label_cols:
            emb_le = emb_label_loaded[col]
            df[col] = emb_le.transform(df[col])

        # Scale numerical features using the fitted StandardScaler
        df[num_cols_scale] = scale_loaded.transform(df[num_cols_scale])

        # Convert final dataframe columns into the required list of NumPy arrays 
        # (Order must match Keras Model inputs: Embeddings first, then numerical block last)
        inputs = [df[col].values for col in emb_label_cols] + [df[num_cols_scale].values]
        return inputs
    
    except KeyError as e:
        # Handles missing fields in the input request
        raise HTTPException(
            status_code=400,
            detail=f"Missing a required key in input data: {str(e)}"
        )
    except ValueError as e:
        # This often occurs due to an 'unseen' categorical value in a production environment
        raise HTTPException(
            status_code=500,
            detail=f"Encoding error: {str(e)}. Check for unseen categorical values."
        )

# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint: Provides a welcome message for the API."""
    return {"message": "Welcome to Car Price Prediction API!"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def prediction_func(car: CarsInputs):
    """
    Predict car price using trained deep learning model and inverse-scale the output.
    """
    try:
        # Preprocess input, returning a list of arrays matching Keras input structure
        model_input = preprocess_input(car.dict())

        # Predict price using the model
        # Output is the SCALED prediction (e.g., 0.45), since the model was trained on scaled prices.
        prediction_scaled = model.predict(model_input)
        
        # Reshape to (n_samples, n_features) i.e. (1, 1) for scikit-learn inverse_transform
        prediction_2nd = prediction_scaled.reshape(-1, 1)
        
        # Inverse Transform: Convert the SCALED output back to the original USD price
        predicted_price = target_scale_loaded.inverse_transform(prediction_2nd)
        
        # Extract the scalar USD value
        final_prediction = float(predicted_price[0][0])

        # Return structured response
        return PredictionResponse(
            predicted_value=f"${final_prediction:.2f}",
            cars_features=car.dict()
        )
    except Exception as e:
        # Catch any unexpected prediction errors
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")

# Run FastAPI app with Uvicorn
if __name__ == "__main__":
    # Runs the API server locally
    uvicorn.run(app, host="127.0.0.1", port=8000)
