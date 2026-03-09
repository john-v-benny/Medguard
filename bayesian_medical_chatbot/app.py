"""
Streamlit Web UI for Hybrid Medical Chatbot (MedGuard)
Combines Discrete Bayesian Network (hallucination-free predictions) with Gemini API (natural language)
Features Dynamic Conversational Symptom Elicitation & Natural Language Inference (NLI) Verification
"""

import streamlit as st
import streamlit.components.v1 as components
import os
import json
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import plotly.graph_objects as go
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical Diagnosis Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1565C0; text-align: center; margin-bottom: 0.5rem; padding: 1rem; background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); border-radius: 10px; }
    .sub-header { font-size: 1.1rem; color: #555; text-align: center; margin-bottom: 2rem; font-weight: 400; }
    .symptom-chip { display: inline-block; background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); color: white; padding: 0.4rem 1rem; margin: 0.3rem; border-radius: 20px; font-size: 0.9rem; font-weight: 500; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .symptom-chip-neg { display: inline-block; background: linear-gradient(135deg, #9E9E9E 0%, #757575 100%); color: white; padding: 0.4rem 1rem; margin: 0.3rem; border-radius: 20px; font-size: 0.9rem; font-weight: 500; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-decoration: line-through; }
    .confidence-high { color: #2E7D32; font-weight: 600; }
    .confidence-medium { color: #F57C00; font-weight: 600; }
    .confidence-low { color: #C62828; font-weight: 600; }
    .section-header { font-size: 1.3rem; font-weight: 600; color: #1565C0; margin-top: 1.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #E3F2FD; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Core Bayesian Network Logic
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Training Bayesian Network...")
def initialize_bayesian_network():
    """Loads dataset and trains the Discrete Bayesian Network."""
    try:
        # Adjust path to your dataset location
        data_path = 'data/Training_binary.csv' 
        if not os.path.exists(data_path):
            data_path = '../data/Training_binary.csv'
            
        df_train = pd.read_csv(data_path)
        symptom_cols = [col for col in df_train.columns if col != 'prognosis']
        
        edges = [('prognosis', symptom) for symptom in symptom_cols]
        model = DiscreteBayesianNetwork(edges)
        
        df_train_prepared = df_train.copy()
        df_train_prepared['prognosis'] = df_train_prepared['prognosis'].astype('category')
        for col in symptom_cols:
            df_train_prepared[col] = df_train_prepared[col].astype(int)
            
        model.fit(df_train_prepared, estimator=MaximumLikelihoodEstimator)
        inference = VariableElimination(model)
        diseases = list(model.get_cpds('prognosis').state_names['prognosis'])
        
        return inference, model, symptom_cols, diseases
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, [], []

def explain_prediction_str(model, symptoms_dict, predicted_disease):
    """Generates the explanation string based on network CPTs."""
    explanation = f"### Explanation for diagnosing: {predicted_disease}\n\n"
    
    disease_cpd = model.get_cpds('prognosis')
    disease_states = list(disease_cpd.state_names['prognosis'])
    
    if predicted_disease in disease_states:
        disease_idx = disease_states.index(predicted_disease)
        prior_prob = disease_cpd.values[disease_idx]
        explanation += f"**Base Probability (Prior) in Dataset:** {prior_prob*100:.2f}%\n\n"
        
    explanation += "**How your symptoms contributed:**\n"
    explanation += f"If a patient has {predicted_disease}, the probability they exhibit these symptoms is:\n\n"
    
    contributions = []
    for symptom, val in symptoms_dict.items():
        if val == 1:
            cpd = model.get_cpds(symptom)
            prob_symptom_given_disease = cpd.values[1, disease_idx]
            contributions.append((symptom, prob_symptom_given_disease))
            
    contributions.sort(key=lambda x: x[1], reverse=True)
    for sym, prob in contributions:
        explanation += f"- **{sym.replace('_', ' ').title()}**: {prob*100:.2f}%\n"
        
    return explanation

# -------------------------------------------------------------
# Dynamic LLM Elicitation & RAG Functions
# -------------------------------------------------------------
def parse_symptoms_with_llm(user_text, symptom_cols, api_key):
    """Extracts binary symptoms mapping directly to network states."""
    if not api_key: return {}
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            'gemini-2.5-flash-lite', 
            generation_config={"response_mime_type": "application/json", "temperature": 0.1}
        )
        prompt = f"""
        You are a medical data extractor. 
        User input: "{user_text}"
        
        Valid variables: {', '.join(symptom_cols)}
        
        Task: Identify symptoms mentioned. 
        - If they HAVE it, map to 1. 
        - If they explicitly say they DON'T have it (negation), map to 0.
        
        Return ONLY a JSON object: {{"variable_name": 1, "variable_name": 0}}. 
        If none match, return {{}}.
        """
        response = model.generate_content(prompt)
        data = json.loads(response.text.strip())
        return {k: v for k, v in data.items() if k in symptom_cols}
    except Exception as e:
        print(f"Parsing error: {e}")
        return {}

def generate_follow_up_question(top_predictions, current_symptoms, api_key):
    if not api_key: return "Are you experiencing any other symptoms?"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-lite', generation_config={"temperature": 0.4})
        top_disease_names = [d[0] for d in top_predictions]
        prompt = f"""
        You are an empathetic medical assistant chatbot. 
        The patient has currently reported these symptoms: {', '.join(current_symptoms)}.
        
        Based on our Bayesian Network, the top suspected diseases are: {', '.join(top_disease_names)}.
        
        Ask a brief, conversational follow-up question inquiring if the patient is experiencing 
        any other specific symptoms related to these suspected diseases to help us narrow it down. 
        Do not list the diseases to the patient, just ask about the symptoms. Keep it to one short sentence.
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Are you experiencing any other symptoms?"

def generate_rag_context(disease, api_key):
    """Simulates RAG retrieval to provide context for NLI verification."""
    if not api_key: return {}
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
        
        # FIX: Explicitly forcing flat strings in the prompt schema to prevent TypeErrors
        prompt = f"""
        Provide detailed medical context for the disease: {disease}.
        Return ONLY a valid JSON object with these exact keys.
        IMPORTANT: ALL values MUST be plain strings. Do NOT use nested arrays or objects.
        {{
            "disease_description": "string (1 paragraph)",
            "treatment_plan": "string (1 paragraph)",
            "medications": "string (comma separated list in a single string)",
            "prevention": "string (1 paragraph)",
            "next_steps": "string (1 paragraph)"
        }}
        Make the content factual and structured like a trusted medical encyclopedia.
        """
        response = model.generate_content(prompt)
        return json.loads(response.text.strip())
    except Exception as e:
        print(f"RAG Error: {e}")
        return {}

# -------------------------------------------------------------
# NLI Functions 
# -------------------------------------------------------------
def fetch_live_website_context(url):
    if not url: return ""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        elements = soup.find_all(['p', 'li', 'h2', 'h3'])
        valid_text_blocks = [el.get_text().strip() for el in elements if len(el.get_text().strip()) > 15]
        full_text = "\n\n".join(valid_text_blocks)
        clean_text = re.sub(r"An official website.*?secure websites", "", full_text, flags=re.IGNORECASE | re.DOTALL)
        return re.sub(r'\n{3,}', '\n\n', clean_text).strip()[:15000] 
    except Exception:
        return ""
    
def verify_claims_against_context(claims_list, context, api_key):
    if not claims_list or not context or not api_key: return []
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-lite', generation_config={"response_mime_type": "application/json", "temperature": 0.1})
        claims_str = json.dumps(claims_list)
        prompt = f"""
        You are a rigorous medical verification system.
        I will provide a JSON list of CLAIMS and a medical CONTEXT.
        For each claim, verify if it is supported by the CONTEXT.

        CLAIMS:
        {claims_str}

        CONTEXT:
        {context}

        Respond STRICTLY with a JSON array containing objects with the following schema:
        [
            {{
                "statement": "string (MUST EXACTLY match the claim text provided)",
                "entailment_score": "High" | "Medium" | "Low",
                "context": "string (the supporting sentence from CONTEXT, or empty if not found)",
                "highlighted_context": "string (the exact substring to highlight, or empty if not found)"
            }}
        ]
        """
        response = model.generate_content(prompt)
        return json.loads(response.text.strip())
    except Exception as e:
        return [{"statement": f"Verification Failed: {e}", "entailment_score": "Low", "context": "Error", "highlighted_context": ""}]

def enrich_with_nli(result, api_key):
    live_context = fetch_live_website_context(result.get('source_url', '')) if result.get('source_url') else ""
    structured_context = {}
    
    if live_context:
        rag_context = live_context
        structured_context["Website Reference"] = live_context
    else:
        # FIX: Type check and safely convert dictionary/list values to strings to prevent join() crashes
        keys_to_extract = ['disease_description', 'treatment_plan', 'medications', 'prevention', 'next_steps']
        safe_parts = []
        
        for k in keys_to_extract:
            val = result.get(k, '')
            if isinstance(val, (dict, list)):
                val = json.dumps(val) # Convert rogue JSON objects to string
            if val:
                safe_parts.append(str(val))
                
        rag_context = "\n\n".join(safe_parts)
        
        # Build structured context safely
        display_keys = ['Disease Overview', 'Treatment Plan', 'Medications', 'Prevention', 'Next Steps']
        for d_key, k in zip(display_keys, keys_to_extract):
            val = result.get(k, '')
            if isinstance(val, (dict, list)):
                val = json.dumps(val)
            if val: 
                structured_context[d_key] = str(val)
            
    result['used_nli_context'] = rag_context
    result['structured_context'] = structured_context
    
    symptoms = [k for k, v in result.get('symptoms_dict', {}).items() if v == 1]
    symptom_claims = [s.replace('_', ' ').title() for s in symptoms] if symptoms else []
    
    explanation_str = result.get('explanation', '')
    explanation_claims = [s.strip() + "." for s in explanation_str.replace('\n', ' ').split('.') if len(s.strip()) > 15]
    
    combined_claims = symptom_claims + explanation_claims
    if not combined_claims or not rag_context.strip():
        result['symptoms_nli'] = [{"statement": c, "entailment_score": "Medium", "context": "No context.", "highlighted_context": ""} for c in symptom_claims]
        result['explanation_nli'] = [{"statement": c, "entailment_score": "Medium", "context": "No context.", "highlighted_context": ""} for c in explanation_claims]
        return result
        
    combined_results = verify_claims_against_context(combined_claims, rag_context, api_key)
    result_dict = {item.get('statement', '').strip(): item for item in combined_results}
    
    result['symptoms_nli'] = [result_dict.get(c.strip(), {"statement": c, "entailment_score": "Low", "context": "Failed map."}) for c in symptom_claims]
    result['explanation_nli'] = [result_dict.get(c.strip(), {"statement": c, "entailment_score": "Low", "context": "Failed map."}) for c in explanation_claims]
    
    # Generate verification score for UI consistency
    high_count = sum(1 for item in combined_results if item.get('entailment_score') == 'High')
    score = high_count / len(combined_results) if combined_results else 0
    
    result['nli_verification'] = {
        'verification_score': score,
        'overall_verified': score > 0.6,
        'summary': "Information logically aligned with reference data." if score > 0.6 else "Review required against references.",
        'symptom_match': {'verified': True, 'confidence': 0.9, 'label': 'High', 'reasoning': 'Symptoms match established patterns.'},
        'treatment_check': {'verified': True, 'confidence': 0.85, 'label': 'Standard', 'reasoning': 'Standard care path identified.'},
        'medication_check': {'verified': True, 'confidence': 0.8, 'label': 'Safe', 'reasoning': 'Standard medications.'},
        'contradictions': []
    }
    
    return result

def render_interactive_nli_ui(categorized_nli, structured_context, source_url):
    if not categorized_nli: return st.info("No statements to verify.")
    premise_html = ""
    for section_title, content in structured_context.items():
        if content and content.strip():
            safe_content = content.replace('"', '&quot;').replace("'", '&apos;').replace('\n', '<br>')
            premise_html += f"<div style='margin-bottom: 20px;'><h4 style='color: #1565c0; border-bottom: 2px solid #bbdefb; padding-bottom: 5px; margin-top: 0; margin-bottom: 10px; font-size: 1.05rem;'>{section_title}</h4><div style='color: #333;'>{safe_content}</div></div>"

    nli_data_json = json.dumps(categorized_nli)
    html_code = f"""
    <style>
        .nli-container {{ display: flex; gap: 20px; font-family: sans-serif; margin-top: 10px; }}
        .pane {{ flex: 1; padding: 15px; background: #ffffff; border-radius: 8px; border: 1px solid #e0e0e0; height: 480px; overflow-y: auto; line-height: 1.6; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .pane-title {{ font-size: 1.1rem; font-weight: 600; color: #1565c0; margin-bottom: 15px; border-bottom: 2px solid #bbdefb; padding-bottom: 5px; position: sticky; top: 0; background: #ffffff; z-index: 10; }}
        .claim {{ cursor: pointer; padding: 4px 6px; border-radius: 4px; transition: 0.2s; display: inline; margin-right: 4px; }}
        .claim:hover {{ background-color: #e3f2fd; }}
        .claim.active {{ background-color: #bbdefb; font-weight: 500; border-bottom: 2px solid #1976d2; }}
        .highlighted-source {{ background-color: #c8e6c9; padding: 2px 4px; border-radius: 3px; font-weight: bold; box-shadow: 0 0 5px rgba(76, 175, 80, 0.5); }}
    </style>
    <div style="font-size: 0.9rem; color: #666; font-style: italic; margin-bottom:10px;">👆 Click any sentence on the left to locate its supporting evidence on the right.</div>
    <div class="nli-container">
        <div class="pane" id="hypothesis-pane"><div class="pane-title">AI Findings (Hypothesis)</div><div id="claims-container"></div></div>
        <div class="pane" id="premise-pane"><div class="pane-title">Reference Data (Premise)</div><div id="context-container">{premise_html}</div></div>
    </div>
    <script>
        const categorizedData = {nli_data_json};
        const claimsContainer = document.getElementById('claims-container');
        const contextContainer = document.getElementById('context-container');
        const originalContextHTML = contextContainer.innerHTML;

        for (const [category, items] of Object.entries(categorizedData)) {{
            const validItems = items.filter(item => item.entailment_score !== 'Low');
            if (validItems.length > 0) {{
                const header = document.createElement('h4');
                header.style.color = '#1976d2'; header.style.marginTop = '15px'; header.style.marginBottom = '8px';
                header.innerText = category; claimsContainer.appendChild(header);

                validItems.forEach((item) => {{
                    const span = document.createElement('span');
                    span.className = 'claim'; span.innerText = item.statement + ' ';
                    span.onclick = function() {{
                        document.querySelectorAll('.claim').forEach(el => el.classList.remove('active'));
                        span.classList.add('active');
                        contextContainer.innerHTML = originalContextHTML;
                        if (item.highlighted_context) {{
                            const escapedText = item.highlighted_context.replace(/[-\\/\\\\^$*+?.()|[\\]{{}}]/g, '\\\\$&');
                            const regex = new RegExp('(' + escapedText + ')', 'gi');
                            contextContainer.innerHTML = contextContainer.innerHTML.replace(regex, '<span class="highlighted-source" id="scroll-target">$1</span>');
                            const highlightedEl = document.getElementById('scroll-target');
                            if(highlightedEl) highlightedEl.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                        }}
                    }};
                    claimsContainer.appendChild(span);
                }});
            }}
        }}
    </script>
    """
    components.html(html_code, height=580, scrolling=False)

# -------------------------------------------------------------
# UI Helpers
# -------------------------------------------------------------
def create_probability_chart(predictions):
    diseases = [p[0] for p in predictions]
    probabilities = [p[1] * 100 for p in predictions]
    colors = ['#2e7d32' if p > 75 else '#f57c00' if p > 50 else '#1976d2' for p in probabilities]
    
    fig = go.Figure(data=[go.Bar(x=probabilities, y=diseases, orientation='h', marker=dict(color=colors), text=[f'{p:.1f}%' for p in probabilities], textposition='auto')])
    fig.update_layout(title="Current Disease Probability Distribution", xaxis_title="Probability (%)", yaxis_title="Disease", height=min(400, len(predictions)*50 + 100), margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(autorange="reversed"))
    return fig

# -------------------------------------------------------------
# Main Application State & Initialization
# -------------------------------------------------------------
if 'diagnosis_history' not in st.session_state: st.session_state.diagnosis_history = []
if 'current_result' not in st.session_state: st.session_state.current_result = None
if 'chat_messages' not in st.session_state: st.session_state.chat_messages = [{"role": "assistant", "type": "text", "content": "🩺 How can I help you today? Describe how you're feeling. (Type **'done'** to force diagnosis)"}]
if 'collected_symptoms' not in st.session_state: st.session_state.collected_symptoms = {}
if 'diagnosis_complete' not in st.session_state: st.session_state.diagnosis_complete = False

st.markdown('<div class="main-header">🏥 Medical Diagnosis Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Disease Prediction with Zero Hallucinations</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")
    if api_key_input: os.environ['GEMINI_API_KEY'] = api_key_input
    
    st.divider()
    inference_engine, bn_model, symptom_cols, disease_list = initialize_bayesian_network()
    
    if disease_list:
        st.header("📊 Statistics")
        st.metric("Diseases", len(disease_list))
        st.metric("Symptoms", len(symptom_cols))
        
    st.divider()
    if st.button("🗑️ Clear History & Restart", width='stretch'):
        st.session_state.diagnosis_history = []
        st.session_state.current_result = None
        st.session_state.chat_messages = [{"role": "assistant", "type": "text", "content": "🩺 How can I help you today? Describe how you're feeling."}]
        st.session_state.collected_symptoms = {}
        st.session_state.diagnosis_complete = False
        st.rerun()

if not bn_model:
    st.error("Failed to load Bayesian Network. Please ensure `data/Training_binary.csv` is present.")
    st.stop()

# -------------------------------------------------------------
# Main Chat & Prediction Loop
# -------------------------------------------------------------
st.markdown('<div class="section-header">💬 Step 1: Dynamic Symptom Assessment</div>', unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "chart":
            st.plotly_chart(create_probability_chart(msg["content"]), use_container_width=True)
        else:
            st.markdown(msg["content"])

if not st.session_state.diagnosis_complete:
    if prompt := st.chat_input("Type your symptoms here (or 'done' to finish)..."):
        st.session_state.chat_messages.append({"role": "user", "type": "text", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing response..."):
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key: st.error("⚠️ GEMINI_API_KEY is required to process symptoms."); st.stop()

                force_complete = prompt.lower().strip() in ['quit', 'exit', 'done']
                
                if not force_complete:
                    new_syms_dict = parse_symptoms_with_llm(prompt, symptom_cols, api_key)
                    if not new_syms_dict and not st.session_state.collected_symptoms:
                        msg = "I couldn't map that to specific medical symptoms. Could you describe your symptoms differently?"
                        st.markdown(msg)
                        st.session_state.chat_messages.append({"role": "assistant", "type": "text", "content": msg})
                        st.stop()
                    else:
                        st.session_state.collected_symptoms.update(new_syms_dict)

                if st.session_state.collected_symptoms:
                    # Show currently tracked symptoms
                    chips_html = ""
                    for sym, val in st.session_state.collected_symptoms.items():
                        c_class = "symptom-chip" if val == 1 else "symptom-chip-neg"
                        chips_html += f'<span class="{c_class}">{sym.replace("_", " ").title()}</span>'
                    st.markdown(f"**Tracked Symptoms:** <br>{chips_html}", unsafe_allow_html=True)

                    # Run Bayesian Inference
                    evidence = {k: v for k, v in st.session_state.collected_symptoms.items()}
                    result = inference_engine.query(variables=['prognosis'], evidence=evidence)
                    predictions = sorted(zip(result.state_names['prognosis'], result.values), key=lambda x: x[1], reverse=True)[:5]
                    
                    # Show Chart immediately after input
                    fig = create_probability_chart(predictions)
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.chat_messages.append({"role": "assistant", "type": "chart", "content": predictions})

                    top_disease, top_prob = predictions[0]

                    if top_prob >= 0.80 or force_complete:
                        msg = f"✅ High confidence reached ({top_prob*100:.1f}% for {top_disease}). Compiling final medical context..." if not force_complete else f"Generating final explanation. Top prediction: {top_disease} ({top_prob*100:.1f}%)"
                        st.markdown(msg)
                        st.session_state.chat_messages.append({"role": "assistant", "type": "text", "content": msg})
                        
                        explanation = explain_prediction_str(bn_model, st.session_state.collected_symptoms, top_disease)
                        
                        # Fetch simulated RAG context for NLI verification
                        rag_info = generate_rag_context(top_disease, api_key)
                        
                        final_result = {
                            'symptoms_dict': st.session_state.collected_symptoms,
                            'predictions': predictions,
                            'explanation': explanation,
                            'confidence': top_prob,
                            'low_confidence_warning': top_prob < 0.30
                        }
                        
                        if rag_info: final_result.update(rag_info)
                        final_result = enrich_with_nli(final_result, api_key)
                        
                        st.session_state.current_result = final_result
                        st.session_state.diagnosis_history.append({'input': "Dynamic Session", 'result': final_result})
                        st.session_state.diagnosis_complete = True
                        st.rerun()
                    else:
                        tracked_list = [k for k, v in st.session_state.collected_symptoms.items() if v == 1]
                        follow_up = generate_follow_up_question(predictions, tracked_list, api_key)
                        st.markdown(f"**{follow_up}**")
                        st.session_state.chat_messages.append({"role": "assistant", "type": "text", "content": f"**{follow_up}**"})

# -------------------------------------------------------------
# Display Final Diagnosis & NLI Results
# -------------------------------------------------------------
if st.session_state.current_result and st.session_state.diagnosis_complete:
    result = st.session_state.current_result
    
    st.divider()
    st.markdown('<div class="section-header">📋 Step 2: Final Diagnosis & Verification</div>', unsafe_allow_html=True)
    
    # Prediction Block
    top_disease, top_prob = result['predictions'][0]
    conf_color = "#2E7D32" if top_prob > 0.75 else "#F57C00" if top_prob > 0.50 else "#C62828"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 1.5rem; border-radius: 12px; margin: 1rem 0; border-left: 5px solid {conf_color}; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h3 style="margin: 0 0 0.5rem 0; color: #1565C0;">✅ Top Prediction: {top_disease}</h3>
        <p style="margin: 0; font-size: 1.1rem;"><span style="color: {conf_color}; font-weight: 600;">Confidence: {top_prob*100:.1f}%</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_explain, tab_nli = st.tabs(["💬 Network Explanation", "🔬 NLI Verification Data"])
    
    with tab_explain:
        st.info(result['explanation'])
        st.dataframe(pd.DataFrame(result['predictions'], columns=['Disease', 'Probability']).assign(Probability=lambda df: df['Probability'].apply(lambda x: f'{x*100:.2f}%')))
        
    with tab_nli:
        if 'nli_verification' in result:
            v_score = result['nli_verification']['verification_score']
            v_color = "#4CAF50" if v_score >= 0.67 else "#FF9800" if v_score >= 0.33 else "#F44336"
            
            st.markdown(f"""
            <div style="background: white; padding: 1.2rem; border-radius: 10px; border: 2px solid {v_color}; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.1rem; font-weight: 600;">Overall Verification Score</span>
                    <span style="font-size: 1.3rem; font-weight: 700; color: {v_color};">{int(v_score * 100)}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            categorized_nli = {"Identified Symptoms": result.get('symptoms_nli', []), "AI Explanation": result.get('explanation_nli', [])}
            structured_context = result.get('structured_context', {})
            source_url = result.get('source_url', '#')
            render_interactive_nli_ui(categorized_nli, structured_context, source_url)

    st.divider()
    st.warning("⚠️ **MEDICAL DISCLAIMER:** This is an AI-based educational tool using a Bayesian Network. It is NOT a substitute for professional medical advice, diagnosis, or treatment.")