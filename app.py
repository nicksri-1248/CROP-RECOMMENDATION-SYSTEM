"""Streamlit Crop Recommendation App

Features:
 - Loads all tuned models (and ensemble) automatically.
 - Displays a dropdown listing: Model Name - accuracy%.
 - Applies saved scaler (if found) and label encoder (for XGBoost encoded labels).
 - Predicts the recommended crop based on user numeric inputs.
"""

import os
import pickle
from pathlib import Path
import numpy as np
import streamlit as st

# -----------------------------
# Paths & loading utilities
# -----------------------------
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent  # one level up (Crop-Recommendation-System)

# Candidate directories where artifacts may exist
MODEL_DIRS = [
    APP_DIR / 'models'  # models copied inside Streamlit folder
    # PROJECT_ROOT / 'Saved'  # tuned models directory
    # PROJECT_ROOT / 'Saved' / 'Best',
    # PROJECT_ROOT / 'Saved' / 'Tuned'
]

def safe_load(path: Path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

def find_file(name_options):
    """Search MODEL_DIRS for the first existing file among name_options (list of filenames)."""
    for d in MODEL_DIRS:
        for name in name_options:
            p = d / name
            if p.exists():
                return p
    return None

# Load scaler (optional)
scaler_path = find_file(['scaler.pkl'])
scaler = safe_load(scaler_path) if scaler_path else None

# Load label encoder (optional - for XGBoost encoded labels)
label_encoder_path = find_file(['model_label_encoder.pkl', 'label_encoder.pkl'])
label_encoder = safe_load(label_encoder_path) if label_encoder_path else None

# Load accuracy dictionaries if available
accuracy_dict_path = find_file(['tuned_accuracy_dict.pkl', 'accuracy_dict.pkl'])
if not accuracy_dict_path:
    # Sometimes stored inside a Tuned subdir
    accuracy_dict_path = find_file(['tuned_accuracy_dict.pkl'])
if accuracy_dict_path:
    accuracies = safe_load(accuracy_dict_path)
else:
    accuracies = {}

# Mapping from file stem to pretty model name
MODEL_NAME_MAP = {
    'model_DT': 'Decision Tree',
    'model_RF': 'Random Forest',
    'model_SVM': 'SVM',
    'model_KNN': 'KNN',
    'model_NB': 'Naive Bayes',
    'model_LR': 'Logistic Regression',
    'model_MLP': 'Neural Network (MLP)',
    'model_GB': 'Gradient Boosting',
    'model_ADA': 'AdaBoost',
    'model_LDA': 'Linear Discriminant Analysis',
    'model_QDA': 'Quadratic Discriminant Analysis',
    'model_XGB': 'XGBoost',
    # 'model_EN_tuned': 'Ensemble (Tuned)',
    'model_EN': 'Ensemble'
}

def discover_models():
    models = {}
    for d in MODEL_DIRS:
        if not d.exists():
            continue
        for fname in os.listdir(d):
            if not fname.endswith('.pkl'):
                continue
            stem = fname[:-4]
            # Skip label encoder artifacts so they do not appear as selectable models
            if 'label_encoder' in stem:
                continue
            if stem.startswith('model_'):
                pretty = MODEL_NAME_MAP.get(stem, stem.replace('model_', '').upper())
                path = d / fname
                if pretty not in models:  # first occurrence wins (prefer earlier dirs)
                    models[pretty] = path
    return models

MODEL_PATHS = discover_models()

@st.cache_resource(show_spinner=False)
def load_models():
    loaded = {}
    for pretty, path in MODEL_PATHS.items():
        obj = safe_load(path)
        if obj is not None:
            loaded[pretty] = obj
    return loaded

MODELS = load_models()

def format_option(name):
    acc = None
    # Try exact match first
    if name in accuracies:
        acc = accuracies[name]
    else:
        # Fallback: attempt fuzzy key match
        for k in accuracies.keys():
            if name.lower().startswith(k.lower()) or k.lower().startswith(name.lower()):
                acc = accuracies[k]
                break
    if acc is not None:
        return f"{name} - {acc:.2f}%"
    return name + " - N/A"

def parse_selected(selected: str):
    # Strip trailing accuracy part
    return selected.split(' - ')[0]

def preprocess(features):
    arr = np.array(features, dtype=float).reshape(1, -1)
    if scaler is not None:
        arr = scaler.transform(arr)
    return arr

def predict(model_name: str, features):
    model = MODELS[model_name]
    X = preprocess(features)
    y_pred = model.predict(X)
    # Decode only for XGBoost numeric outputs using the loaded label encoder
    if model_name == 'XGBoost' and label_encoder is not None and isinstance(y_pred[0], (np.integer, int)):
        try:
            decoded = label_encoder.inverse_transform(y_pred)
            return decoded[0]
        except Exception:
            return y_pred[0]
    return y_pred[0]

def main():
    st.title('Crop Recommendation System')
    st.caption('Select a tuned model and enter soil & climate parameters to get a crop recommendation.')

    st.markdown(
        """**Feature Guide**  
        Enter realistic agronomic values. Ranges shown are approximate typical bounds from publicly available crop datasets (adjust if your local context differs).  
        - **Nitrogen (N)**: 0–140 kg/ha (available nitrogen content in soil)  
        - **Phosphorus (P)**: 0–145 kg/ha (available phosphorus)  
        - **Potassium (K)**: 0–205 kg/ha (available potassium)  
        - **Temperature (°C)**: 0–50 (ambient soil/air temp)  
        - **Humidity (%)**: 10–100 (relative humidity)  
        - **pH**: 3.5–9.5 (soil acidity/alkalinity)  
        - **Rainfall (mm)**: 0–300 (recent / seasonal cumulative)  
        """
    )

    if not MODELS:
        st.error('No models found. Please ensure pickle files are present in Streamlit or Saved directories.')
        return

    # Model selection
    options = sorted([format_option(name) for name in MODELS.keys()])
    selected_display = st.selectbox('Choose Model (name - accuracy):', options)
    selected_model_name = parse_selected(selected_display)

    with st.expander('Model Info', expanded=False):
        st.write(f"Using model: {selected_model_name}")
        if selected_model_name in accuracies:
            st.write(f"Reported accuracy: {accuracies[selected_model_name]:.2f}%")
        else:
            st.write("Accuracy not available in stored dictionary.")

    col1, col2, col3 = st.columns(3)
    with col1:
        Nitrogen = st.number_input('Nitrogen (kg/ha)', min_value=0.0, max_value=140.0, value=60.0, step=1.0, help='Available nitrogen content in soil. Typical 0–140.')
        Phosphorus = st.number_input('Phosphorus (kg/ha)', min_value=0.0, max_value=145.0, value=50.0, step=1.0, help='Available phosphorus content. Typical 0–145.')
        potassium = st.number_input('Potassium (kg/ha)', min_value=0.0, max_value=205.0, value=50.0, step=1.0, help='Available potassium content. Typical 0–205.')
    with col2:
        Temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0, step=0.1, help='Ambient temperature influencing crop growth.')
        Humidity = st.number_input('Humidity (%)', min_value=10.0, max_value=100.0, value=70.0, step=0.1, help='Relative humidity percentage.')
    with col3:
        Ph = st.number_input('Soil pH', min_value=3.5, max_value=9.5, value=6.5, step=0.01, help='Soil pH (acidity/alkalinity). Most crops prefer 5.5–7.5.')
        Rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=300.0, value=100.0, step=0.1, help='Recent cumulative / seasonal rainfall.')

    inputs = [Nitrogen, Phosphorus, potassium, Temperature, Humidity, Ph, Rainfall]

    if st.button('Predict Crop'):
        try:
            result = predict(selected_model_name, inputs)
            st.success(f'Recommended Crop: {result}')
        except Exception as e:
            st.error(f'Prediction failed: {e}')

    st.markdown('---')
    st.caption('Loaded models: ' + ', '.join(sorted(MODELS.keys())))
    if scaler is not None:
        st.caption('Scaler applied: StandardScaler')
    if label_encoder is not None:
        st.caption('Label encoder loaded for decoding numeric predictions.')

if __name__ == '__main__':
    main()
