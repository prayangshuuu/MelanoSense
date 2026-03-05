import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
from PIL import Image
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# Define paths
ML_MODELS_DIR = os.path.join(settings.BASE_DIR, 'ml_models')

cnn_model = None
meta_model = None
label_encoders = {}

def load_models():
    global cnn_model, meta_model, label_encoders
    
    # Load CNN Model
    cnn_path = os.path.join(ML_MODELS_DIR, "cnn_model.h5")
    if os.path.exists(cnn_path):
        try:
            cnn_model = tf.keras.models.load_model(cnn_path)
            logger.info("CNN model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading CNN model: {e}")
    else:
        logger.warning(f"CNN model not found at {cnn_path}")

    # Load XGBoost Model
    xgb_path = os.path.join(ML_MODELS_DIR, "xgboost_metadata_model.json")
    if os.path.exists(xgb_path):
        try:
            meta_model = xgb.XGBClassifier()
            meta_model.load_model(xgb_path)
            logger.info("XGBoost model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {e}")
    else:
        logger.warning(f"XGBoost model not found at {xgb_path}")

    # Load Label Encoders
    encoders_path = os.path.join(ML_MODELS_DIR, "meta_label_encoders.pkl")
    if os.path.exists(encoders_path):
        try:
            label_encoders = joblib.load(encoders_path)
            logger.info("Label encoders loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading label encoders: {e}")
    else:
        logger.warning(f"Label encoders not found at {encoders_path}")

def preprocess_image(image):
    try:
        # Ensure image is RGB
        img = Image.open(image)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise ValueError("Invalid image format")

def predict_image(image):
    if cnn_model is None:
        raise RuntimeError("CNN model is not loaded. Please check server logs.")
    
    try:
        img = preprocess_image(image)
        prediction = float(cnn_model.predict(img)[0][0])
        # Ensure prediction is a valid probability
        prediction = max(0.0, min(1.0, prediction))
        return prediction
    except Exception as e:
        logger.error(f"Error predicting image: {e}")
        raise RuntimeError(f"Prediction failed: {str(e)}")

def predict_metadata(data):
    """
    data: dict containing 'age', 'sex', 'localization'
    """
    if meta_model is None:
        # If model is missing, we can't predict. 
        # Option 1: Raise error. Option 2: Return None or neutral probability.
        # Given the requirement "Fix any issues immediately", raising an error is safer 
        # than giving a fake result, but we might want to allow partial prediction if CNN works.
        # However, the formula is (cnn + meta) / 2. So we need both.
        raise RuntimeError("Metadata model is not loaded. Please check server logs.")
        
    encoded = []
    feature_order = ['age', 'sex', 'localization']
    
    for feature in feature_order:
        value = data.get(feature)
        
        # If feature needs encoding
        if feature in label_encoders:
            le = label_encoders[feature]
            try:
                # Transform expects a list/array, handle single value
                value = le.transform([value])[0]
            except Exception as e:
                # Unknown or invalid category should be treated as a hard error
                logger.error(
                    "Encoding error for feature '%s' with value '%s': %s",
                    feature,
                    value,
                    e,
                )
                raise ValueError(f"Unknown category for feature '{feature}': {value}")
        
        encoded.append(value)

    encoded = np.array(encoded).reshape(1, -1)

    try:
        # predict_proba returns [prob_class_0, prob_class_1]
        pred = float(meta_model.predict_proba(encoded)[0][1])
        # Ensure prediction is a valid probability
        pred = max(0.0, min(1.0, pred))
        return pred
    except Exception as e:
        logger.error(f"Error predicting metadata: {e}")
        raise RuntimeError(f"Metadata prediction failed: {str(e)}")
