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
    Standardized to match Scan model choices: Low, Moderate, High
    """
    if percentage < 30:
        return {
            'risk_level': "Low",
            'risk_display': "Low Risk",
            'risk_class': "risk-low",
            'alert_class': "alert-success",
            'theme_color': "#059669"
        }
    elif percentage < 70:
        return {
            'risk_level': "Moderate",
            'risk_display': "Moderate Risk",
            'risk_class': "risk-warning",
            'alert_class': "alert-warning",
            'theme_color': "#D97706"
        }
    else:
        return {
            'risk_level': "High",
            'risk_display': "High Risk",
            'risk_class': "risk-high",
            'alert_class': "alert-danger",
            'theme_color': "#DC2626"
        }


def get_last_conv_layer(model):
    """
    Find the name of the last convolutional layer in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,)):
            return layer.name
    return None

def preprocess_image_path(image_path):
    """
    Load and preprocess image from path for CNN input.
    Matches the trained CNN input size (128x128).
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image at {image_path}: {e}")
        return None

def generate_gradcam_overlay(image_path, model, scan_id):
    """
    Main entry point for generating and saving Grad-CAM heatmaps.
    Returns the relative URL to the saved heatmap.
    """
    import cv2
    from django.core.files.base import ContentFile
    
    # 1. Define storage paths
    scan_folder = os.path.join(settings.MEDIA_ROOT, 'scans', str(scan_id))
    heatmap_filename = "heatmap_overlay.png"
    heatmap_path = os.path.join(scan_folder, heatmap_filename)
    heatmap_relative_url = f"{settings.MEDIA_URL}scans/{scan_id}/{heatmap_filename}"
    
    # 2. Check if already exists (Caching)
    if os.path.exists(heatmap_path):
        logger.info(f"Using existing heatmap for scan {scan_id}")
        
        # Ensure database is synced even if file exists on disk
        try:
            from .models import Scan
            scan_obj = Scan.objects.filter(id=scan_id).first()
            if scan_obj and not scan_obj.heatmap_image:
                scan_obj.heatmap_image.name = f"scans/{scan_id}/{heatmap_filename}"
                scan_obj.save(update_fields=['heatmap_image'])
        except Exception as e:
            logger.warning(f"Could not sync Scan heatmap field on cache hit: {e}")
            
        return heatmap_relative_url

    # 3. Create folder if missing
    os.makedirs(scan_folder, exist_ok=True)

    # 4. Generate Heatmap
    img_array = preprocess_image_path(image_path)
    if img_array is None:
        return None

    last_conv_layer_name = get_last_conv_layer(model)
    if not last_conv_layer_name:
        logger.error("No convolutional layer found in model.")
        return None

    heatmap = generate_gradcam(img_array, model, last_conv_layer_name)
    
    # 5. Create Overlay
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        return None
        
    heatmap_resized = cv2.resize(heatmap, (img_cv2.shape[1], img_cv2.shape[0]))
    heatmap_norm = np.uint8(255 * heatmap_resized)
    jet = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet * 0.4 + img_cv2
    
    # 6. Save as PNG
    success = cv2.imwrite(heatmap_path, superimposed_img)
    if success:
        logger.info(f"Heatmap overlay saved successfully for scan {scan_id}")
        
        # Keep the Scan object updated if it exists (Lazy update)
        try:
            from .models import Scan
            scan_obj = Scan.objects.filter(id=scan_id).first()
            if scan_obj:
                # We save the relative path to the ImageField
                scan_obj.heatmap_image.name = f"scans/{scan_id}/{heatmap_filename}"
                scan_obj.save(update_fields=['heatmap_image'])
        except Exception as e:
            logger.warning(f"Could not update Scan object heatmap field: {e}")
            
        return heatmap_relative_url
    else:
        logger.error(f"Failed to save heatmap overlay for scan {scan_id}")
        return None

def generate_gradcam(img_array, model, last_conv_layer_name):
    """
    Generate Grad-CAM heatmap using a split-model approach for Keras 3 Sequential models.
    """
    # 1. Get the last convolutional layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_idx = model.layers.index(last_conv_layer)
    
    # 2. Build sub-models
    # Model A: Input -> Last Conv Output
    model_a = tf.keras.Model(model.inputs, last_conv_layer.output)
    
    # Model B: Last Conv Output -> Final Output (Tail)
    # Re-building the tail as a Sequential model
    tail_model = tf.keras.Sequential(model.layers[last_conv_idx+1:])
    
    with tf.GradientTape() as tape:
        # Pass input through Model A
        conv_outputs = model_a(img_array)
        tape.watch(conv_outputs)
        # Pass activations through Model B
        preds = tail_model(conv_outputs)
        # We assume binary classification (single node output)
        class_score = preds[:, 0]

    # Explicitly calculate gradients of score w.r.t activations
    grads = tape.gradient(class_score, conv_outputs)
    
    if grads is None:
        # Fallback if splitting fails
        return np.zeros((7, 7)) 

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def extract_roi(original_path, heatmap):
    """
    Extract lesion ROI based on Grad-CAM activation.
    Uses thresholding and contour detection.
    """
    import cv2
    
    # Read original image
    img = cv2.imread(original_path)
    if img is None:
        return None
        
    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Threshold heatmap to get high activation region (Top 40%)
    heatmap_norm = np.uint8(255 * heatmap_resized)
    _, thresh = cv2.threshold(heatmap_norm, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None # Fallback if no high activation region found
        
    # Find the largest contour (expected to be the lesion focus)
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Add some padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    
    # Crop original image
    roi = img[y:y+h, x:x+w]
    return roi

def save_analysis_images(scan):
    """
    Backward compatible wrapper or helper if needed.
    Now uses generate_gradcam_overlay logic locally.
    """
    if not scan.image or not scan.image.original_file:
        return
    
    img_path = scan.image.original_file.path
    # Generate heatmap using the new standardized function
    generate_gradcam_overlay(img_path, cnn_model, scan.id)
    
    # For ROI, we still use the extract_roi logic
    img_array = preprocess_image_path(img_path)
    if img_array is not None:
        last_conv = get_last_conv_layer(cnn_model)
        heatmap = generate_gradcam(img_array, cnn_model, last_conv)
        roi_img = extract_roi(img_path, heatmap)
        
        if roi_img is not None:
            import cv2
            from django.core.files.base import ContentFile
            _, roi_buffer = cv2.imencode('.jpg', roi_img)
            scan.roi_image.save(f"roi_{scan.id}.jpg", ContentFile(roi_buffer.tobytes()), save=True)

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
        'percentage': round(hybrid_prob * 100, 1),
        'prediction': prediction_label,
        'confidence': round(hybrid_prob * 100, 2),
        'cnn_prob': round(cnn_prob * 100, 1),
        'meta_prob': round(xgb_prob * 100, 1),
        **risk_meta
    }
