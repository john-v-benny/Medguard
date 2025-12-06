import os
import json
import pickle
import base64
import numpy as np
import pandas as pd
import matplotlib

# Set backend to Agg before importing pyplot
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

from io import BytesIO
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
MODEL_PATH = 'saved_models/bayesian_disease_model.pkl'
MODEL_INFO_PATH = 'saved_models/model_info.json'
FEATURES = ["Fever", "Headache", "Cough", "Fatigue", "Body_Pain"]

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    history: List[ChatMessage]
    collected: Dict[str, Any]

class PredictionRequest(BaseModel):
    symptoms: Dict[str, Any]

# --- Services ---
class ServiceContainer:
    model = None
    model_info = None

services = ServiceContainer()

def load_prediction_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            services.model = pickle.load(f)
        with open(MODEL_INFO_PATH, 'r') as f:
            services.model_info = json.load(f)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def create_heatmap_b64(symptoms: Dict[str, float]) -> str:
    try:
        classes = services.model_info['classes']
        class_means = [services.model.theta_[i, :] for i in range(len(classes))]
        means_df = pd.DataFrame(class_means, index=classes, columns=FEATURES)
        
        user_vals = []
        for f in FEATURES:
            val = symptoms.get(f)
            if isinstance(val, dict): val = val.get('value', 0)
            user_vals.append(float(val) if val is not None else 0.0)

        user_input_df = pd.DataFrame([user_vals], columns=FEATURES, index=['Your Symptoms'])
        combined_df = pd.concat([means_df, user_input_df])

        fig = plt.Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        sns.heatmap(combined_df, annot=True, cmap='viridis', ax=ax, fmt='.1f', linewidths=.5, cbar=False)
        ax.set_title('Symptom Comparison', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        fig.tight_layout()
        
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        print(f"Heatmap Error: {e}")
        return ""

def get_gemini_response(chat_history, collected_data):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API Key missing")
    
    genai.configure(api_key=api_key)
    
    # 1. DETERMINISTIC ORDER LOGIC
    # We calculate exactly what is missing based on the fixed FEATURES list order.
    missing_features = [f for f in FEATURES if collected_data.get(f) is None]
    
    # The target is the FIRST missing feature. 
    target_symptom = missing_features[0] if missing_features else None
    
    # We explicitly tell the AI what to focus on.
    system_goal = f"Your IMMEDIATE goal is to ask for the value of: {target_symptom}." if target_symptom else "All data collected. Inform the user you are ready to diagnose."

    SYSTEM_INSTRUCTIONS = f"""
    You are a professional medical intake assistant. 
    CURRENT TASK: {system_goal}
    
    RULES:
    1. If the user provides a value for {target_symptom}, extract it into 'updates'.
    2. If the user denies a symptom (e.g., "no headache"), set its value to 0.
    3. Be brief and professional.
    4. RETURN JSON ONLY.
    
    JSON FORMAT: 
    {{
       "updates": {{ "Fever": 98.6, "Headache": 0, ... }}, 
       "acknowledgment": "Brief text acknowledging input.",
       "next_symptom_to_ask": "{target_symptom}"
    }}
    """
    
    model = genai.GenerativeModel("models/gemini-2.5-flash", system_instruction=SYSTEM_INSTRUCTIONS)
    
    ctx = {
        "collected": {k: v for k, v in collected_data.items() if v is not None},
        "history": [msg.dict() for msg in chat_history[-4:]] # Keep context short
    }
    
    response = model.generate_content(json.dumps(ctx))
    text = response.text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except:
        # Fallback if AI fails to generate JSON
        return {
            "updates": {},
            "acknowledgment": "Could you please repeat that?",
            "next_symptom_to_ask": target_symptom
        }

@app.on_event("startup")
async def startup_event():
    load_prediction_model()

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    return get_gemini_response(request.history, request.collected)

@app.post("/api/predict")
async def predict_endpoint(request: PredictionRequest):
    if not services.model: raise HTTPException(status_code=503, detail="Model loading...")
    
    clean_symptoms = {}
    for k, v in request.symptoms.items():
        if isinstance(v, dict): clean_symptoms[k] = float(v.get('value', 0))
        else: clean_symptoms[k] = float(v if v is not None else 0)

    input_array = np.array([[clean_symptoms[f] for f in FEATURES]])
    
    prediction = services.model.predict(input_array)[0]
    probabilities = services.model.predict_proba(input_array)[0]
    
    classes = services.model_info['classes']
    prob_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
    
    return {
        "prediction": prediction,
        "confidence": float(np.max(probabilities)),
        "all_probabilities": prob_dict,
        "heatmap_b64": create_heatmap_b64(clean_symptoms)
    }