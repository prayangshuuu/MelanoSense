# MelanoSense

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white" alt="Django">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/XGBoost-1E8CBE?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">
</p>

## Overview
MelanoSense is a professional medical AI application designed for skin lesion classification and melanoma risk analysis.

## Dataset
The model was trained using the **HAM10000** dataset (The Human Against Machine with 10000 training images).

## Model Architecture
The application utilizes a high-performance **Hybrid Model** combining:
- **CNN (Convolutional Neural Networks)**: Processes lesion images via a TensorFlow/Keras pipeline for visual feature extraction.
- **XGBoost**: Analyzes patient metadata (age, sex, localization) for optimized risk classification.
- **Hybrid Inference**: A weighted ensemble (70% CNN, 30% XGBoost) for final diagnostic confidence.

## Getting Started

### Prerequisites
- Python 3.10+
- Virtual Environment (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/prayangshuuu/MelanoSense.git
   cd MelanoSense
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install django tensorflow xgboost joblib pillow numpy
   ```
4. Run migrations:
   ```bash
   python manage.py migrate
   ```
5. Start the development server:
   ```bash
   python manage.py runserver
   ```

## License
Distributed under the MIT License. See `LICENSE` for more information.
