import os
import sys
from typing import Optional, Dict, Any, List
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Local paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from models.predictor import DiseasePredictor
from llm.gemini_agent import make_client, route_user_message

# --- Constants and Configuration ---
CONFIDENCE_THRESHOLD = 0.7
MODEL_PATH = os.path.join(parent_dir, 'saved_models', 'bayesian_disease_model.pkl')
MODEL_INFO_PATH = os.path.join(parent_dir, 'saved_models', 'model_info.json')
FEATURES = ["Fever", "Headache", "Cough", "Fatigue", "Body_Pain"]

# --- Helper Functions ---
def create_explanation_heatmap(symptoms, predictor):
    """Generates a base64 encoded heatmap image comparing user symptoms to model patterns."""
    class_means = [predictor.model.theta_[i, :] for i in range(len(predictor.classes))]
    means_df = pd.DataFrame(class_means, index=predictor.classes, columns=FEATURES)
    user_input_df = pd.DataFrame([symptoms], columns=FEATURES, index=['Your Symptoms'])

    fig, ax = plt.subplots(figsize=(12, 8))
    combined_df = pd.concat([means_df, user_input_df])
    sns.heatmap(combined_df, annot=True, cmap='viridis', ax=ax, fmt='.1f', linewidths=.5)
    ax.set_title('Comparison: Your Symptoms vs. Learned Disease Patterns', fontweight='bold', fontsize=14)
    ax.set_xlabel('Symptoms', fontweight='bold')
    ax.set_ylabel('')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64_img

def initialize_session_state():
    """Sets up the initial session state for the application."""
    if "predictor" not in st.session_state:
        st.session_state.predictor = DiseasePredictor(MODEL_PATH, MODEL_INFO_PATH)
    if "gemini" not in st.session_state:
        st.session_state.gemini = make_client()
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm here to help gather some information about your symptoms. How are you feeling today?"}]
    if "collected" not in st.session_state:
        st.session_state.collected = {k: None for k in FEATURES}
    if "pending_symptom" not in st.session_state:
        st.session_state.pending_symptom = None
    if "final_result" not in st.session_state:
        st.session_state.final_result = None

# --- UI Rendering ---
st.set_page_config(page_title="🤖 AI Medical Assistant", page_icon="🩺", layout="wide")
st.title("🤖 AI-Powered Medical Symptom Collector")
st.caption("This tool uses Gemini for conversational symptom gathering and a Bayesian model for analysis.")

initialize_session_state()

# Sidebar for progress and controls
with st.sidebar:
    st.header("Symptom Progress")
    done_count = sum(1 for v in st.session_state.collected.values() if v is not None)
    st.progress(done_count / len(FEATURES))
    for feature, value in st.session_state.collected.items():
        status = "✅" if value is not None else "⏳"
        display_value = f"{value:.1f}" if isinstance(value, float) else "Not provided"
        st.markdown(f"- {status} **{feature}**: `{display_value}`")
    st.divider()
    if st.button("🔄 Start Over", use_container_width=True):
        for key in ["messages", "collected", "pending_symptom", "final_result"]:
            st.session_state.pop(key, None)
        st.rerun()

# --- Main Application Logic ---
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Check if all data is collected to run the model
if all(v is not None for v in st.session_state.collected.values()) and not st.session_state.final_result:
    with st.spinner("All symptoms collected! Running the Bayesian analysis..."):
        result = st.session_state.predictor.predict(st.session_state.collected, CONFIDENCE_THRESHOLD)
        st.session_state.final_result = result
        st.rerun()

# Display final results if available
if st.session_state.final_result:
    res = st.session_state.final_result
    st.subheader("📊 Bayesian Model Analysis")
    if res.get("success"):
        if res.get("threshold_met"):
            st.success(f"Primary Likelihood: **{res['prediction']}** (Confidence: {res['confidence']:.1%})")
        else:
            st.warning(f"Uncertain Classification: Highest confidence is for **{res['prediction']}** at {res['confidence']:.1%}, which is below the required threshold.")
        
        # Display probability chart
        prob_df = pd.DataFrame(list(res["all_probabilities"].items()), columns=["Disease", "Probability"]).sort_values("Probability", ascending=True)
        fig = px.bar(prob_df, x="Probability", y="Disease", orientation="h", text=prob_df["Probability"].map(lambda p: f"{p:.1%}"), title="Probability Distribution Across Conditions")
        st.plotly_chart(fig, use_container_width=True)

        # Display heatmap explanation
        try:
            heatmap_b64 = create_explanation_heatmap(st.session_state.collected, st.session_state.predictor)
            st.markdown(f'<p style="text-align: center;">The heatmap below shows how your symptoms (bottom row) compare to the typical patterns the model has learned for each disease.</p>', unsafe_allow_html=True)
            st.image(f"data:image/png;base64,{heatmap_b64}")
        except Exception as e:
            st.error(f"Could not generate the explanation heatmap: {e}")
    else:
        st.error("The model could not produce a result with the provided inputs.")
    st.info("Disclaimer: This is an AI demonstration and not a substitute for professional medical advice.")

# Interaction logic: either show widget for pending symptom or general chat input
elif st.session_state.pending_symptom:
    symptom = st.session_state.pending_symptom
    # Render the appropriate input widget
    if symptom == "Fever":
        val = st.number_input("Temperature (°F)", 95.0, 105.0, 98.6, 0.1, key=f"widget_{symptom}")
    else:
        val = st.slider(f"{symptom} Severity (0-10)", 0.0, 10.0, 5.0, 0.5, key=f"widget_{symptom}")

    if st.button(f"Submit {symptom} Value", use_container_width=True):
        # Update state with submitted value
        st.session_state.collected[symptom] = float(val)
        human_readable_val = f"{val}°F" if symptom == "Fever" else f"{val}/10"
        
        # --- FIX: Append the message BEFORE calling the agent ---
        # This ensures the AI has the latest context.
        st.session_state.messages.append({"role": "user", "content": f"My {symptom.lower().replace('_', ' ')} is {human_readable_val}."})
        st.session_state.pending_symptom = None # Clear pending symptom

        # Let the LLM process this new information and decide the next question
        with st.spinner("Thinking..."):
            response = route_user_message(st.session_state.gemini, st.session_state.messages, st.session_state.collected)
        
        # Add AI acknowledgment and next question to chat
        if response.get("acknowledgment"):
            st.session_state.messages.append({"role": "assistant", "content": response["acknowledgment"]})
        
        # Check if there's another question to ask
        if response.get("next_symptom_to_ask"):
            st.session_state.pending_symptom = response["next_symptom_to_ask"]

        st.rerun()

else:
    # General chat input when no specific value is pending
    user_text = st.chat_input("Describe your symptoms...")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        
        # Call Gemini to process the free-text input
        with st.spinner("Thinking..."):
            response = route_user_message(st.session_state.gemini, st.session_state.messages, st.session_state.collected)

        # Apply any updates extracted by the AI
        if updates := response.get("updates"):
            for k, v in updates.items():
                if k in FEATURES and v is not None and st.session_state.collected[k] is None:
                    st.session_state.collected[k] = float(v)

        # Add AI acknowledgment and next question to chat
        if ack := response.get("acknowledgment"):
            st.session_state.messages.append({"role": "assistant", "content": ack})
        
        # Set the next symptom to ask for, which will trigger the widget on the next run
        if symptom_to_ask := response.get("next_symptom_to_ask"):
            st.session_state.pending_symptom = symptom_to_ask
        
        st.rerun()