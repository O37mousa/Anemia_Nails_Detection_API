from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model = joblib.load("anemia_model.joblib")  # Ensure the model file is in the same directory

@app.get("/")
def home():
    return {"message": "Anemia Detection API is running!"}

@app.post("/predict/")
def predict(data: dict):
    try:
        # Convert input JSON into a DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]  # Assuming binary classification (1 = Anemic, 0 = Not Anemic)
        
        # Convert output to human-readable format
        result = "Anemic" if prediction == 1 else "Not Anemic"

        return {"anemia_prediction": result}
    except Exception as e:
        return {"error": str(e)}
