from django.shortcuts import render
from .forms import PredictionForm
from .utils import predict_image, predict_metadata
import base64
import logging

logger = logging.getLogger(__name__)

def index(request):
    result = None
    image_base64 = None
    error_message = None
    
    if request.method == 'POST':
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Get form data
                image_file = request.FILES['image']
                age = form.cleaned_data['age']
                sex = form.cleaned_data['sex']
                localization = form.cleaned_data['localization']
                
                # Read image for display
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                # Reset file pointer for PIL processing in utils
                image_file.seek(0)
                
                # Get predictions
                # CNN Prediction
                cnn_prob = predict_image(image_file)
                
                # Metadata Prediction
                meta_data = {
                    'age': age,
                    'sex': sex,
                    'localization': localization
                }
                meta_prob = predict_metadata(meta_data)
                
                # Combine Predictions
                final_risk = (cnn_prob + meta_prob) / 2
                percentage = final_risk * 100
                
                # Interpretation
                if percentage < 30:
                    risk_level = "Low Risk"
                    risk_class = "text-success"
                    alert_class = "alert-success"
                elif percentage < 70:
                    risk_level = "Moderate Risk"
                    risk_class = "text-warning"
                    alert_class = "alert-warning"
                else:
                    risk_level = "High Risk"
                    risk_class = "text-danger"
                    alert_class = "alert-danger"
                    
                result = {
                    'percentage': f"{percentage:.1f}%",
                    'risk_level': risk_level,
                    'risk_class': risk_class,
                    'alert_class': alert_class,
                    'cnn_prob': f"{cnn_prob*100:.1f}%",
                    'meta_prob': f"{meta_prob*100:.1f}%"
                }
            except Exception as e:
                # Log full error for debugging, but show a generic message to the user
                logger.exception(f"Prediction failed: {e}")
                error_message = (
                    "We couldn't complete the analysis for this image and data. "
                    "Please verify that the image is a clear skin lesion photo and that all fields are filled correctly, then try again."
                )
                # Keep the uploaded image for display even if prediction fails
                if 'image_file' in locals() and not image_base64:
                    image_file.seek(0)
                    image_data = image_file.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')

    else:
        form = PredictionForm()
    
    return render(request, 'index.html', {
        'form': form, 
        'result': result, 
        'image_base64': image_base64,
        'error_message': error_message
    })
