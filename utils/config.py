import os
from dotenv import load_dotenv
import joblib
import xgboost as xgb
import cv2
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.color import rgb2lab
from skimage import img_as_float
from scipy.stats import skew, kurtosis

# Load environment variables
load_dotenv(override=True)

APP_NAME = os.getenv("APP_NAME")
APP_VERSION = os.getenv("APP_VERSION")
SECRET_KEY_TOKEN = os.getenv("SECRET_KEY_TOKEN")

# Define the base directory and artifacts folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_FOLDER_PATH = os.path.join(BASE_DIR, "artifacts")

# -------------------------
# Preprocessing Functions
# -------------------------

def preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return img

def read_upload_file(uploaded_file):
    contents = uploaded_file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")
    return img

def preprocess_uploaded_image(image, target_size=(224, 224)):
    if image is None:
        raise ValueError("No image data provided for preprocessing.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32)
    image = preprocess_input(image)
    return image

def extract_color_features(images):
    features = []
    for img in images:
        img = img_as_float(img)
        if np.isnan(img).any() or np.isinf(img).any():
            raise ValueError("Image contains NaN or infinite values.")
        
        lab = rgb2lab(img)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        l_mean, l_std, l_skew, l_kurt = np.mean(L), np.std(L), skew(L.flatten()), kurtosis(L.flatten())
        a_mean, a_std, a_skew, a_kurt = np.mean(A), np.std(A), skew(A.flatten()), kurtosis(A.flatten())
        b_mean, b_std, b_skew, b_kurt = np.mean(B), np.std(B), skew(B.flatten()), kurtosis(B.flatten())
        
        try:
            l_hist, _ = np.histogram(L.flatten(), bins=256, range=(0, 100), density=True)
            a_hist, _ = np.histogram(A.flatten(), bins=256, range=(-128, 128), density=True)
            b_hist, _ = np.histogram(B.flatten(), bins=256, range=(-128, 128), density=True)
        except RuntimeWarning as e:
            print(f"Warning during histogram calculation: {e}")
            l_hist, a_hist, b_hist = np.zeros(256), np.zeros(256), np.zeros(256)
        
        feature_vector = np.array([l_mean, l_std, l_skew, l_kurt,
                                   a_mean, a_std, a_skew, a_kurt,
                                   b_mean, b_std, b_skew, b_kurt])
        hist_features = np.concatenate([l_hist, a_hist, b_hist])
        feature_vector = np.concatenate([feature_vector, hist_features])
        
        features.append(feature_vector)
    return np.array(features)

# Group preprocessing functions into a dictionary for easy access
preprocessing = {
    "preprocess_image": preprocess_image,
    "read_upload_file": read_upload_file,
    "preprocess_uploaded_image": preprocess_uploaded_image,
    ###   "preprocess_metadata": preprocess_metadata,
    "extract_color_features": extract_color_features
}

# -------------------------
# Load Trained Models
# -------------------------
# metadata_model = joblib.load(os.path.join(ARTIFACTS_FOLDER_PATH, "metadata_model.pkl"))
color_model = joblib.load(os.path.join(ARTIFACTS_FOLDER_PATH, "color_model.pkl"))
final_model = xgb.Booster()
final_model.load_model(os.path.join(ARTIFACTS_FOLDER_PATH, "final_model.json"))
