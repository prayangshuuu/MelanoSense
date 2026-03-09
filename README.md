# MelanoSense

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white" alt="Django">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/XGBoost-1E8CBE?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">
</p>

## About

MelanoSense is an advanced AI-driven screening platform designed to facilitate the early detection of melanoma, one of the most aggressive and life-threatening forms of skin cancer. The system functions as a clinical decision support tool, leveraging a hybrid deep learning pipeline to analyze dermoscopic images of skin lesions and identify patterns associated with malignant melanoma. By integrating artificial intelligence with medical imaging analysis, MelanoSense aims to assist healthcare professionals in performing rapid preliminary assessments of suspicious skin lesions.
The platform was developed as part of a digital healthcare initiative aimed at improving melanoma awareness and supporting dermatological screening in Bangladesh, where access to specialized dermatological services is often limited, particularly in rural and underserved regions. Through AI-assisted analysis, the system helps bridge the gap between early symptom recognition and timely clinical evaluation.
MelanoSense features a professional medical dashboard that allows users to securely upload dermoscopic skin lesion images and initiate automated AI-powered diagnostic analysis. The underlying model processes the image through a trained deep learning architecture capable of extracting complex visual features related to asymmetry, border irregularity, color variation, and structural abnormalities commonly associated with melanoma.
After analysis, the platform generates a comprehensive diagnostic report designed to support clinical interpretation. The report includes:
Predicted melanoma risk level (low, moderate, or high risk)
Model confidence score, indicating the reliability of the prediction
Grad-CAM heatmap visualization, which highlights the specific regions of the lesion image that influenced the model’s decision, enhancing transparency and explainability of the AI system
A downloadable clinical PDF report, which can be stored, shared, or used as part of patient documentation
By combining artificial intelligence, medical image analysis, and explainable AI techniques, MelanoSense provides an accessible, efficient, and interpretable tool that can support dermatologists and healthcare professionals in early melanoma risk assessment and screening workflows.

## Tech Stack & Frameworks

| Layer | Technology |
|---|---|
| **Backend** | [Django 6.x](https://www.djangoproject.com/) (Python) |
| **AI / Deep Learning** | [TensorFlow / Keras](https://www.tensorflow.org/) — CNN for image feature extraction |
| **Machine Learning** | [XGBoost](https://xgboost.readthedocs.io/) — Metadata-based risk classification |
| **Image Processing** | [Pillow](https://pillow.readthedocs.io/), [OpenCV](https://opencv.org/) — Image manipulation & Grad-CAM overlays |
| **PDF Reports** | [xhtml2pdf](https://xhtml2pdf.readthedocs.io/) — Clinical diagnostic report generation |
| **Frontend** | HTML5, CSS3, JavaScript, [Cropper.js](https://fengyuanchen.github.io/cropperjs/) |
| **Database** | SQLite (default) |
| **Serialization** | joblib — Model & label encoder persistence |

## Dataset

The model was trained using the **HAM10000** dataset (The Human Against Machine with 10000 training images) — a large collection of multi-source dermatoscopic images of common pigmented skin lesions.

## Model Architecture

The application utilizes a high-performance **Hybrid Model** combining:
- **CNN (Convolutional Neural Networks)**: Processes lesion images via a TensorFlow/Keras pipeline for visual feature extraction.
- **XGBoost**: Analyzes patient metadata (age, sex, localization) for optimized risk classification.
- **Hybrid Inference**: A weighted ensemble (70% CNN, 30% XGBoost) for final diagnostic confidence.

## Getting Started

### Prerequisites

- **Python 3.10+** — [Download Python](https://www.python.org/downloads/)
- **pip** — Comes bundled with Python 3.4+
- **Git** — [Download Git](https://git-scm.com/downloads)
- A virtual environment tool (recommended: built-in `venv`)

### Installation

Follow these steps to get MelanoSense running on your local machine:

**1. Clone the repository**

```bash
git clone https://github.com/prayangshuuu/MelanoSense.git
cd MelanoSense
```

**2. Create and activate a virtual environment**

This isolates project dependencies from your system Python.

```bash
# Create the virtual environment
python -m venv .venv

# Activate it
# macOS / Linux:
source .venv/bin/activate

# Windows (Command Prompt):
.venv\Scripts\activate

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

> You should see `(.venv)` at the beginning of your terminal prompt when the environment is active.

**3. Install all dependencies**

All required packages are listed in `requirements.txt`. Install them in one go:

```bash
pip install -r requirements.txt
```

<details>
<summary>What gets installed?</summary>

| Package | Purpose |
|---|---|
| `django` | Web framework powering the application |
| `tensorflow` | Deep learning engine for the CNN model |
| `xgboost` | Gradient boosting for metadata classification |
| `joblib` | Loading pre-trained label encoders |
| `pillow` | Image file handling and preprocessing |
| `numpy` | Numerical operations for model inference |
| `opencv-python` | Grad-CAM heatmap generation and image processing |
| `xhtml2pdf` | Rendering HTML templates into downloadable PDF reports |

</details>

**4. Apply database migrations**

This sets up the SQLite database schema required by the application:

```bash
python manage.py migrate
```

**5. Create a superuser (optional)**

If you want access to the Django admin panel:

```bash
python manage.py createsuperuser
```

**6. Start the development server**

```bash
python manage.py runserver
```

The application will be available at **http://127.0.0.1:8000/**

## Project Structure

```
MelanoSense/
├── MelanoSense/          # Django project settings & configuration
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── predictor/            # Core application — models, views, AI logic
│   ├── models.py         # Database models (MedicalImage, Scan)
│   ├── views.py          # Request handlers & prediction workflow
│   ├── utils.py          # Hybrid inference engine, Grad-CAM, risk metadata
│   ├── forms.py          # Patient data input forms
│   └── urls.py           # App-level URL routing
├── ml_models/            # Pre-trained model files
│   ├── cnn_model.h5      # TensorFlow CNN weights
│   ├── xgboost_metadata_model.json
│   └── meta_label_encoders.pkl
├── templates/            # Django HTML templates
├── static/               # CSS, JavaScript, and static assets
├── media/                # User-uploaded images (created at runtime)
└── requirements.txt      # Python dependencies
```

## Credits

Built by **Team Minus One**

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
