from django.shortcuts import render
from .forms import PredictionForm
from .utils import hybrid_inference
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
                age = form.cleaned_data['age']
                sex = form.cleaned_data['sex']
                localization = form.cleaned_data['localization']
                
                # Check for resized image from frontend (base64)
                image_resized_data = request.POST.get('image_resized')
                
                if image_resized_data and 'base64,' in image_resized_data:
                    # Decode base64 image
                    format, imgstr = image_resized_data.split(';base64,')
                    image_data = base64.b64decode(imgstr)
                    image_base64 = imgstr # Use the one from frontend directly for display
                    
                    # Create a file-like object for the models
                    from io import BytesIO
                    image_file = BytesIO(image_data)
                else:
                    # Fallback to original file
                    image_file = request.FILES['image']
                    image_data = image_file.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    image_file.seek(0)

                # Get hybrid prediction (CNN + XGBoost)
                result = hybrid_inference(
                    image_file,
                    age=age,
                    sex=sex,
                    localization=localization,
                )
            except Exception as e:
                # Log full error for debugging, but show a generic message to the user
                logger.exception(f"Prediction failed: {e}")
                error_message = (
                    "We couldn't complete the analysis for this image and data. "
                    "Please verify that the image is a clear skin lesion photo and that all fields are filled correctly, then try again."
                )
                # Keep the uploaded image for display even if prediction fails
                if 'image_file' in locals() and not image_base64:
                    try:
                        image_file.seek(0)
                        image_data = image_file.read()
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                    except Exception:
                        image_base64 = None

    else:
        form = PredictionForm()
    
    return render(request, 'index.html', {
        'form': form,
        'result': result,
        'image_base64': image_base64,
        'error_message': error_message,
    })
