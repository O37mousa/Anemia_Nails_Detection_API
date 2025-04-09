from fastapi import FastAPI
import joblib
import pandas as pd
from utils.config import APP_NAME, APP_VERSION, SECRET_KEY_TOKEN

app = FastAPI(title=APP_NAME)

# Load the trained model
#   model = joblib.load("random_forest_model_nails.pkl")  # Ensure the model file is in the same directory

@app.get("/")
async def home():
    return {
        "app_name": APP_NAME,
        "message": "Anemia Detection API for Palm is running!"
    }



"""
@app.post("/predict/")
def predict(data: dict):
    try:
        # Convert input JSON into a DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]  # Assuming binary classification (1 = Anemic, 0 = Not Anemic)
        
        # Convert output to human-readable format
        result = "Anemic" if prediction == 1 else "Not Anemic"

        return {"anemia_detection": result}
    except Exception as e:
        return {"error": str(e)}
"""