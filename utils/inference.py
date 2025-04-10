import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import UploadFile
# from utils.request import Request
from utils.config import rf_model
from utils.config import preprocessing

# def detect_new(data: Request, image: UploadFile, rf_model):
def detect_new( image: UploadFile, rf_model):
    """
    Inference function for the rf_model pipeline.
    """

    '''
    # -------------------------
    # Process Uploaded Image & Extract Color Features
    # -------------------------
    
    # uploaded_img = preprocessing["read_upload_file"](image)
    # processed_img = preprocessing["preprocess_uploaded_image"](uploaded_img)
    
    # color_features = preprocessing["extract_color_features"]([processed_img])
    # color_pred = color_model.predict(color_features)
    '''
    
    try:
        # Read and preprocess image
        uploaded_img = preprocessing["read_upload_file"](image)
        processed_img = preprocessing["preprocess_uploaded_image"](uploaded_img)
        color_features = preprocessing["extract_color_features"]([processed_img])

        # Make prediction
        prediction = rf_model.predict(color_features)
        prediction_value = float(prediction[0])
        
        return {
            "final_prediction": prediction_value
        }

    except Exception as e:
        return {"error": f"There's a problem in anemic detection: {str(e)}"}

    # return {
    #     "final_prediction": float(rf_model[0])
    # }
