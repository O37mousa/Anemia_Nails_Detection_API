from fastapi import FastAPI, HTTPException, UploadFile, File
# from utils.request import Request
from utils.config import (
    APP_NAME,
    APP_VERSION, 
    SECRET_KEY_TOKEN, 
    rf_model
)
from utils.inference import detect_new
import joblib
import pandas as pd

app = FastAPI(title=APP_NAME)

# Load the trained model
# model = joblib.load("random_forest_model_nails.pkl")  # Ensure the model file is in the same directory

@app.get("/", tags=['Nails Anemia Detection'])
async def home() -> dict:
    return {
        "app_name": APP_NAME,
        "message": "Anemia Detection API for Palm is running!"
    } 

@app.post("/detect/forest", tags=["Models"], description="Detection of Anemia using RF")
# async def predict_forest(data: str, image: UploadFile = File(...)) -> dict:   
async def detect_rf(image: UploadFile = File(...)) -> dict:
    try:
        # Parse the JSON string into the Request model
        # from utils.request import Request  # ensure the model is imported here
        # parsed_data = Request.parse_raw(data)

        # call the function
        response = detect_new(
            # data=parsed_data,     
            image=image,
            rf_model=rf_model
        )
        return response     
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"There's a problem in anemic detection, {str(e)}")

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