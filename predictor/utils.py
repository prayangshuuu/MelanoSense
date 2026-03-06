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
            
        # Match the trained CNN input size
        img = img.resize((128, 128))
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
        prediction = float(cnn_model.predict(img, verbose=0)[0][0])
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

def get_risk_metadata(percentage):
    """
    Centralized logic for risk level, class, and alert styling.
    """
    if percentage < 30:
        return {
            'risk_level': "Low Risk",
            'risk_class': "risk-low",
            'alert_class': "alert-success",
            'theme_color': "#059669"
        }
    elif percentage < 70:
        return {
            'risk_level': "Moderate Risk",
            'risk_class': "risk-warning",
            'alert_class': "alert-warning",
            'theme_color': "#D97706"
        }
    else:
        return {
            'risk_level': "High Risk",
            'risk_class': "risk-high",
            'alert_class': "alert-danger",
            'theme_color': "#DC2626"
        }


def hybrid_inference(image, age, sex, localization):
    """
    Run the full hybrid inference pipeline:
    - CNN image pathway
    - XGBoost metadata pathway
    - 0.7 * CNN + 0.3 * XGBoost combination

    Returns:
        prediction_label: "CANCER" or "NON_CANCER"
        hybrid_prob: float in [0, 1]
        cnn_prob: float in [0, 1]
        xgb_prob: float in [0, 1]
    """
    if cnn_model is None or meta_model is None:
        raise RuntimeError(
            "Prediction models are not fully loaded. Please check server logs."
        )

    # CNN pathway (uses the uploaded image file-like object)
    cnn_prob = predict_image(image)

    # XGBoost pathway
    meta_data = {
        'age': age,
        'sex': sex,
        'localization': localization,
    }
    xgb_prob = predict_metadata(meta_data)

    # Hybrid model: 70% CNN, 30% XGBoost
    hybrid_prob = (0.7 * cnn_prob) + (0.3 * xgb_prob)
    hybrid_prob = max(0.0, min(1.0, float(hybrid_prob)))

    # Classification threshold at 0.4
    prediction = 1 if hybrid_prob >= 0.4 else 0
    prediction_label = "CANCER" if prediction == 1 else "NON-CANCEROUS"

    # Get dynamic styling and meta data
    risk_meta = get_risk_metadata(hybrid_prob * 100)

    return {
        'percentage': f"{hybrid_prob * 100:.1f}",
        'prediction': prediction_label,
        'confidence': f"{hybrid_prob * 100:.2f}%",
        'cnn_prob': f"{cnn_prob * 100:.1f}%",
        'meta_prob': f"{xgb_prob * 100:.1f}%",
        **risk_meta
    }
