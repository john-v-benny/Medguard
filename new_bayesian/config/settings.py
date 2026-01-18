CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence required for prediction
MODEL_PATH = 'saved_models/bayesian_disease_model.pkl'
MODEL_INFO_PATH = 'saved_models/model_info.json'

# Streamlit configuration
PAGE_TITLE = "Disease Prediction System"
PAGE_ICON = "🏥"

# Feature descriptions for user interface
FEATURE_DESCRIPTIONS = {
    'Fever': 'Body temperature in Fahrenheit (95°F - 105°F)',
    'Headache': 'Headache intensity on a scale of 0-10',
    'Cough': 'Cough severity on a scale of 0-10', 
    'Fatigue': 'Fatigue level on a scale of 0-10',
    'Body_Pain': 'Body pain intensity on a scale of 0-10'
}

# Disease information
DISEASE_INFO = {
    'Common Cold': 'A viral infection of the upper respiratory tract',
    'Malaria': 'A mosquito-borne infectious disease',
    'Cough': 'A respiratory condition with persistent coughing',
    'Asthma': 'A respiratory condition with breathing difficulties',
    'Normal Fever': 'General fever symptoms',
    'Body Ache': 'General body pain and discomfort',
    'Runny Nose': 'Nasal congestion and discharge',
    'Dengue': 'A mosquito-borne viral infection'
}