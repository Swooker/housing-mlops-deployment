# Housing Price Predictor - MLOps Deployment

This repository provides a simple web-based deployment for predicting housing prices using a pre-trained machine learning model. The app is built using [Gradio](https://www.gradio.app/).

## Repository Contents

- `Housing.csv`: Dataset used to train the model.
- `model.pkl`: Pre-trained linear regression model.
- `app.py`: Gradio app that provides a user interface for predictions.

## Getting Started

### 1.Clone Repository
```bash
git clone https://github.com/Swooker/housing-mlops-deployment.git
cd housing-mlops-deployment

### 2. Install Requirements

pip install gradio pandas scikit-learn joblib

### 3. Run the App

python app.py
