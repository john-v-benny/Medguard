"""
Streamlit Web UI for Hybrid Medical Chatbot
Combines Bayesian Network (hallucination-free predictions) with Gemini API (natural language)
Features Dynamic Conversational Symptom Elicitation
"""

import streamlit as st
import streamlit.components.v1 as components
import os
import json
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from hybrid_chatbot import HybridMedicalChatbot
import plotly.graph_objects as go
import pandas as pd

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical Diagnosis Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced user experience
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1565C0;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Symptom chips */
    .symptom-chip {
        display: inline-block;
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        padding: 0.4rem 1rem;
        margin: 0.3rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Prediction cards */
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #1976D2;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Confidence colors */
    .confidence-high {
        color: #2E7D32;
        font-weight: 600;
    }
    .confidence-medium {
        color: #F57C00;
        font-weight: 600;
    }
    .confidence-low {
        color: #C62828;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E3F2FD;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state (Added Dynamic Chat States)
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'diagnosis_history' not in st.session_state:
    st.session_state.diagnosis_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": "🩺 How can I help you today? Describe how you're feeling in plain English. (Type **'done'** to force diagnosis early)"}]
if 'collected_symptoms' not in st.session_state:
    st.session_state.collected_symptoms = {}
if 'diagnosis_complete' not in st.session_state:
    st.session_state.diagnosis_complete = False

# Dynamic LLM Elicitation Functions
def parse_symptoms_with_llm(user_text, symptom_cols, api_key):
    """Uses Gemini to extract symptoms from natural language and map them strictly to the database column names."""
    if not api_key: 
        return []
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            'gemini-2.5-flash-lite', 
            generation_config={"response_mime_type": "application/json", "temperature": 0.1}
        )
        prompt = f"""
        You are a medical data parser. 
        The user said: "{user_text}"
        
        Here is the exact list of valid symptom variables in our database:
        {', '.join(symptom_cols)}
        
        Extract the symptoms the user is describing and map them strictly to the exact variable names above.
        Return ONLY a valid JSON list of strings representing the matched variables. 
        If no symptoms match, return an empty list [].
        """
        response = model.generate_content(prompt)
        matched_symptoms = json.loads(response.text.strip())
        return [sym for sym in matched_symptoms if sym in symptom_cols]
    except Exception as e:
        print(f"Parsing error: {e}")
        return []

def generate_follow_up_question(top_predictions, current_symptoms, api_key):
    """Uses Gemini to look at the top predicted diseases and ask the user if they have specific symptoms."""
    if not api_key: 
        return "Are you experiencing any other symptoms?"
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
    except Exception as e:
        return "Are you experiencing any other symptoms?"

# NLI and RAG Functions
def fetch_live_website_context(url):
    """Scrapes a live URL and extracts paragraph and bullet-point text cleanly."""
    if not url:
        return ""
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        elements = soup.find_all(['p', 'li', 'h2', 'h3'])
        
        valid_text_blocks = []
        for el in elements:
            text = el.get_text().strip()
            if len(text) > 15: 
                valid_text_blocks.append(text)
                
        full_text = "\n\n".join(valid_text_blocks)
        boilerplate_pattern = r"An official website of the United States government.*?Share sensitive information only on official, secure websites"
        clean_text = re.sub(boilerplate_pattern, "", full_text, flags=re.IGNORECASE | re.DOTALL)
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()
        
        return clean_text[:15000] 
        
    except Exception as e:
        print(f"Failed to scrape URL {url}: {e}")
        return ""
    
def verify_claims_against_context(claims_list, context, api_key):
    """Verifies a batched list of claims against the MedlinePlus context in a single API call."""
    if not claims_list or not context or not api_key:
        return []
    
    try:
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel(
            'gemini-2.5-flash-lite', 
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        )
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        claims_str = json.dumps(claims_list)
        
        prompt = f"""
        You are a rigorous medical verification system.
        I will provide a JSON list of CLAIMS and a MedlinePlus medical CONTEXT.
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
        
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings
        )
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        return json.loads(response_text.strip())
        
    except Exception as e:
        error_msg = str(e)
        print(f"NLI Verification Error Detail: {error_msg}")
        
        ui_msg = "API Error"
        if "429" in error_msg or "Quota" in error_msg:
            ui_msg = "Rate Limit Reached (Wait a minute before analyzing again)"
        elif "safety" in error_msg.lower() or "blocked" in error_msg.lower():
            ui_msg = "Blocked by API Safety Filters"
            
        return [{
            "statement": f"Verification Failed: {ui_msg}",
            "entailment_score": "Low",
            "context": "Check the terminal console for the exact error trace.",
            "highlighted_context": ""
        }]

def enrich_with_nli(result, api_key):
    """Batches symptoms and explanations into a SINGLE API call to prevent rate limiting and extracts structured context."""
    
    live_context = ""
    source_url = result.get('source_url', '')
    if source_url:
        st.toast(f"Scraping live context from {source_url}...", icon="🌐")
        live_context = fetch_live_website_context(source_url)
    
    structured_context = {}
    if live_context:
        rag_context = live_context
        structured_context["Website Reference"] = live_context
    else:
        rag_context = "\n\n".join(filter(None, [
            result.get('disease_description', ''),
            result.get('treatment_plan', ''),
            result.get('medications', ''),
            result.get('prevention', ''),
            result.get('next_steps', '')
        ]))
        if result.get('disease_description'):
            structured_context["Disease Overview"] = result['disease_description']
        if result.get('treatment_plan'):
            structured_context["Treatment Plan"] = result['treatment_plan']
        if result.get('medications'):
            structured_context["Medications"] = result['medications']
        if result.get('prevention'):
            structured_context["Prevention & Risk Reduction"] = result['prevention']
        if result.get('next_steps'):
            structured_context["Next Steps"] = result['next_steps']
            
    result['used_nli_context'] = rag_context
    result['structured_context'] = structured_context
    
    # Clean Symptom Claims
    symptoms = result.get('symptoms', [])
    symptom_claims = [s.replace('_', ' ').title() for s in symptoms] if symptoms else []
    
    # Clean Explanation Claims
    explanation_str = result.get('rag_explanation', result.get('explanation', ''))
    if explanation_str:
        explanation_str = explanation_str.replace('**', '')
        explanation_claims = [s.strip() + "." for s in explanation_str.replace('\n', ' ').split('.') if len(s.strip()) > 5]
    else:
        explanation_claims = []
    
    if not rag_context.strip():
        result['symptoms_nli'] = [{"statement": c, "entailment_score": "Medium", "context": "No MedlinePlus context available.", "highlighted_context": ""} for c in symptom_claims]
        result['explanation_nli'] = [{"statement": c, "entailment_score": "Medium", "context": "No MedlinePlus context available.", "highlighted_context": ""} for c in explanation_claims]
        return result

    combined_claims = symptom_claims + explanation_claims
    
    if not combined_claims:
        result['symptoms_nli'] = []
        result['explanation_nli'] = []
        return result
        
    combined_results = verify_claims_against_context(combined_claims, rag_context, api_key)
    
    if len(combined_results) == 1 and combined_results[0].get("statement", "").startswith("Verification Failed"):
        result['symptoms_nli'] = combined_results
        result['explanation_nli'] = combined_results
        return result
        
    result_dict = {item.get('statement', '').strip(): item for item in combined_results}
    
    symptoms_nli = []
    for claim in symptom_claims:
        matched = result_dict.get(claim.strip())
        if matched:
            symptoms_nli.append(matched)
        else:
            symptoms_nli.append({"statement": claim, "entailment_score": "Low", "context": "Failed to map API response.", "highlighted_context": ""})
            
    explanation_nli = []
    for claim in explanation_claims:
        matched = result_dict.get(claim.strip())
        if matched:
            explanation_nli.append(matched)
        else:
            explanation_nli.append({"statement": claim, "entailment_score": "Low", "context": "Failed to map API response.", "highlighted_context": ""})
            
    result['symptoms_nli'] = symptoms_nli
    result['explanation_nli'] = explanation_nli
        
    return result

def render_interactive_nli_ui(categorized_nli, structured_context, source_url):
    """Renders a custom HTML/JS split-pane for interactive bidirectional highlighting."""
    if not categorized_nli:
        st.info("No statements to verify.")
        return

    premise_html = ""
    for section_title, content in structured_context.items():
        if content and content.strip():
            safe_content = content.replace('"', '&quot;').replace("'", '&apos;').replace('\n', '<br>')
            premise_html += f"""
            <div style='margin-bottom: 20px;'>
                <h4 style='color: #1565c0; border-bottom: 2px solid #bbdefb; padding-bottom: 5px; margin-top: 0; margin-bottom: 10px; font-size: 1.05rem;'>
                    {section_title}
                </h4>
                <div style='color: #333;'>{safe_content}</div>
            </div>
            """

    if not premise_html:
        premise_html = "<div style='color: #666; font-style: italic;'>No detailed medical context available.</div>"

    nli_data_json = json.dumps(categorized_nli)
    
    html_code = f"""
    <style>
        .nli-container {{
            display: flex;
            gap: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-top: 10px;
        }}
        .pane {{
            flex: 1;
            padding: 15px;
            background: #ffffff;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            height: 480px;
            overflow-y: auto;
            line-height: 1.6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .pane-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #1565c0;
            margin-bottom: 15px;
            border-bottom: 2px solid #bbdefb;
            padding-bottom: 5px;
            position: sticky;
            top: 0;
            background: #ffffff;
            z-index: 10;
        }}
        .claim {{
            cursor: pointer;
            padding: 4px 6px;
            border-radius: 4px;
            transition: background-color 0.2s;
            display: inline;
            margin-right: 4px;
        }}
        .claim:hover {{
            background-color: #e3f2fd;
        }}
        .claim.active {{
            background-color: #bbdefb;
            font-weight: 500;
            border-bottom: 2px solid #1976d2;
        }}
        .highlighted-source {{
            background-color: #c8e6c9;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
            transition: all 0.3s ease;
        }}
        .instruction {{
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 10px;
            font-style: italic;
        }}
    </style>

    <div class="instruction">👆 Click any sentence on the left to locate its supporting evidence on the right.</div>
    
    <div class="nli-container">
        <div class="pane" id="hypothesis-pane">
            <div class="pane-title">AI Findings (Hypothesis)</div>
            <div id="claims-container"></div>
        </div>
        
        <div class="pane" id="premise-pane">
            <div class="pane-title">Reference Data (Premise)</div>
            <div id="context-container">{premise_html}</div>
        </div>
    </div>
    
    <div style="margin-top: 15px; text-align: center;">
        <a href="{source_url}" target="_blank" style="color: #1976D2; text-decoration: none; font-weight: 600; font-size: 0.95rem;">
            🔗 View Source on MedlinePlus
        </a>
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
                header.style.color = '#1976d2';
                header.style.marginTop = '15px';
                header.style.marginBottom = '8px';
                header.style.fontSize = '1.05rem';
                header.innerText = category;
                claimsContainer.appendChild(header);

                validItems.forEach((item) => {{
                    const span = document.createElement('span');
                    span.className = 'claim';
                    span.innerText = item.statement + ' ';
                    
                    span.onclick = function() {{
                        document.querySelectorAll('.claim').forEach(el => el.classList.remove('active'));
                        span.classList.add('active');
                        
                        contextContainer.innerHTML = originalContextHTML;
                        
                        if (item.highlighted_context) {{
                            const highlightText = item.highlighted_context;
                            const escapedText = highlightText.replace(/[-\\/\\\\^$*+?.()|[\\]{{}}]/g, '\\\\$&');
                            const regex = new RegExp('(' + escapedText + ')', 'gi');
                            
                            contextContainer.innerHTML = contextContainer.innerHTML.replace(
                                regex, 
                                '<span class="highlighted-source" id="scroll-target">$1</span>'
                            );
                            
                            const highlightedEl = document.getElementById('scroll-target');
                            if(highlightedEl) {{
                                highlightedEl.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                            }}
                        }}
                    }};
                    claimsContainer.appendChild(span);
                }});
            }}
        }}
    </script>
    """
    
    components.html(html_code, height=580, scrolling=False)

def initialize_chatbot():
    """Initialize the hybrid chatbot."""
    api_key = os.getenv('GEMINI_API_KEY')
    model_path = 'models/disease_bayesian_network.pkl'
    
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found in environment variables!")
        st.info("Please set your API key in the `.env` file or sidebar.")
        return None
    
    try:
        chatbot = HybridMedicalChatbot(model_path, api_key)
        return chatbot
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please train the model first by running `03_bayesian_network_binary.ipynb`")
        return None
    except Exception as e:
        st.error(f"⚠️ Error initializing chatbot: {e}")
        return None

def create_probability_chart(predictions):
    """Create an interactive bar chart for predictions."""
    diseases = [p[0] for p in predictions]
    probabilities = [p[1] * 100 for p in predictions]
    
    colors = []
    for prob in probabilities:
        if prob > 75:
            colors.append('#2e7d32') 
        elif prob > 50:
            colors.append('#f57c00')  
        else:
            colors.append('#1976d2')  
    
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities,
            y=diseases,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{p:.1f}%' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Disease Probability Distribution",
        xaxis_title="Probability (%)",
        yaxis_title="Disease",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def display_symptoms(symptoms):
    """Display symptoms as chips."""
    if symptoms:
        st.markdown("**Identified Symptoms:**")
        chips_html = "".join([
            f'<span class="symptom-chip">{s.replace("_", " ").title()}</span>'
            for s in symptoms
        ])
        st.markdown(chips_html, unsafe_allow_html=True)

def display_predictions(predictions):
    """Display predictions in a formatted table."""
    if predictions:
        df = pd.DataFrame(predictions, columns=['Disease', 'Probability'])
        df['Probability'] = df['Probability'].apply(lambda x: f'{x*100:.2f}%')
        df['Rank'] = range(1, len(df) + 1)
        df = df[['Rank', 'Disease', 'Probability']]
        
        st.dataframe(
            df,
            hide_index=True,
            width='stretch',
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Disease": st.column_config.TextColumn("Disease", width="large"),
                "Probability": st.column_config.TextColumn("Confidence", width="medium"),
            }
        )

# Main UI
st.markdown('<div class="main-header">🏥 Medical Diagnosis Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Disease Prediction with Zero Hallucinations</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key_input = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        help="Leave empty to use .env file",
        placeholder="AIzaSy..."
    )
    
    if api_key_input:
        os.environ['GEMINI_API_KEY'] = api_key_input
    
    st.divider()
    
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot uses:
    - **Bayesian Network** for accurate, hallucination-free predictions
    - **Gemini API** for natural language understanding
    - **131 symptoms** and **41 diseases**
    
    **How it works:**
    1. Describe your symptoms naturally
    2. AI asks dynamic follow-up questions
    3. Bayesian Network evaluates probabilities
    4. Explanations show after high confidence is reached
    """)
    
    st.divider()
    
    st.header("📊 Statistics")
    if st.session_state.chatbot:
        st.metric("Diseases", len(st.session_state.chatbot.diseases))
        st.metric("Symptoms", len(st.session_state.chatbot.symptom_cols))
        st.metric("Diagnoses Made", len(st.session_state.diagnosis_history))
    
    st.divider()
    
    if st.button("🗑️ Clear History & Restart", width='stretch'):
        st.session_state.diagnosis_history = []
        st.session_state.current_result = None
        st.session_state.chat_messages = [{"role": "assistant", "content": "🩺 How can I help you today? Describe how you're feeling in plain English. (Type **'done'** to force diagnosis early)"}]
        st.session_state.collected_symptoms = {}
        st.session_state.diagnosis_complete = False
        st.rerun()

# Initialize chatbot
if st.session_state.chatbot is None:
    with st.spinner("Initializing chatbot..."):
        st.session_state.chatbot = initialize_chatbot()

# Main content
if st.session_state.chatbot:
    st.markdown('<div class="section-header">💬 Step 1: Dynamic Symptom Assessment</div>', unsafe_allow_html=True)
    
    # -------------------------------------------------------------
    # Interactive Chat Loop for Dynamic Symptom Elicitation
    # -------------------------------------------------------------
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.diagnosis_complete:
        if prompt := st.chat_input("Type your symptoms here (or 'done' to finish)..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing response..."):
                    api_key = os.getenv('GEMINI_API_KEY')
                    if not api_key:
                        st.error("⚠️ GEMINI_API_KEY is required to process symptoms.")
                        st.stop()

                    # Handle forcing diagnosis
                    if prompt.lower().strip() in ['quit', 'exit', 'done']:
                        if not st.session_state.collected_symptoms:
                            msg = "No symptoms collected yet. Please describe your symptoms first."
                            st.markdown(msg)
                            st.session_state.chat_messages.append({"role": "assistant", "content": msg})
                            st.rerun()
                        else:
                            force_complete = True
                            new_syms = []
                    else:
                        force_complete = False
                        new_syms = parse_symptoms_with_llm(prompt, st.session_state.chatbot.symptom_cols, api_key)

                    # Ensure the AI recognized something
                    if not force_complete and not new_syms and not st.session_state.collected_symptoms:
                        msg = "I couldn't map that to specific medical symptoms. Could you describe your symptoms differently?"
                        st.markdown(msg)
                        st.session_state.chat_messages.append({"role": "assistant", "content": msg})
                    
                    else:
                        for sym in new_syms:
                            st.session_state.collected_symptoms[sym] = 1

                        tracked_list = list(st.session_state.collected_symptoms.keys())
                        
                        if tracked_list:
                            st.info(f"📋 **Tracked symptoms:** {', '.join(tracked_list).replace('_', ' ').title()}")
                            
                            # Run Bayesian Network Prediction
                            predictions = st.session_state.chatbot.predict_disease(st.session_state.collected_symptoms, top_n=5)
                            
                            if predictions and len(predictions) > 0:
                                top_prob = predictions[0][1]
                                top_disease = predictions[0][0]
                                
                                # Check threshold or force completion
                                if top_prob >= 0.85 or force_complete:
                                    if force_complete:
                                        msg = f"Generating final explanation based on collected symptoms. Top prediction: {top_disease} ({top_prob*100:.1f}%)"
                                    else:
                                        msg = f"✅ High confidence reached ({top_prob*100:.1f}% for {top_disease}). Generating final explanation..."
                                    
                                    st.markdown(msg)
                                    st.session_state.chat_messages.append({"role": "assistant", "content": msg})
                                    st.write("Compiling final report and verifying with NLI...")

                                    # Only provide the explanation here (after full confidence)
                                    explanation = st.session_state.chatbot.explain_results(predictions, tracked_list)
                                    
                                    # Fetch RAG Context if available
                                    rag_info = None
                                    if hasattr(st.session_state.chatbot, 'use_rag') and st.session_state.chatbot.use_rag and hasattr(st.session_state.chatbot, 'rag') and st.session_state.chatbot.rag:
                                        try:
                                            rag_info = st.session_state.chatbot.rag.explain_diagnosis(
                                                disease=top_disease,
                                                confidence=top_prob,
                                                symptoms=tracked_list
                                            )
                                        except Exception as e:
                                            st.warning(f"Could not retrieve detailed medical information: {e}")

                                    # Build Final Output Data
                                    final_result = {
                                        'symptoms': tracked_list,
                                        'predictions': predictions,
                                        'explanation': explanation,
                                        'requires_followup': False,
                                        'followup_questions': [],
                                        'confidence': top_prob,
                                        'low_confidence_warning': top_prob < getattr(st.session_state.chatbot, 'MINIMUM_CONFIDENCE', 0.1)
                                    }
                                    
                                    if rag_info:
                                        final_result.update({
                                            'rag_explanation': rag_info.get('explanation', ''),
                                            'treatment_plan': rag_info.get('treatment_plan', ''),
                                            'medications': rag_info.get('medications', ''),
                                            'next_steps': rag_info.get('next_steps', ''),
                                            'prevention': rag_info.get('prevention', ''),
                                            'source_url': rag_info.get('source_url', ''),
                                            'source_name': rag_info.get('source_name', ''),
                                            'disease_description': rag_info.get('disease_description', ''),
                                            'complications': rag_info.get('complications', '')
                                        })
                                        if 'nli_verification' in rag_info:
                                            final_result['nli_verification'] = rag_info['nli_verification']
                                            
                                    # Enrich result payload with Natural Language Inference Checks
                                    final_result = enrich_with_nli(final_result, api_key)
                                    
                                    st.session_state.current_result = final_result
                                    st.session_state.diagnosis_history.append({
                                        'input': "Dynamic Chat Session (" + ", ".join(tracked_list).replace("_", " ") + ")",
                                        'result': final_result
                                    })
                                    
                                    st.session_state.diagnosis_complete = True
                                    st.rerun()

                                else:
                                    # Confidence is too low, ask follow up question dynamically
                                    follow_up = generate_follow_up_question(predictions, tracked_list, api_key)
                                    pred_info = f"\n\n*Current top suspicion: {top_disease} ({top_prob*100:.1f}%)*"
                                    full_msg = f"{follow_up}{pred_info}"
                                    
                                    st.markdown(full_msg)
                                    st.session_state.chat_messages.append({"role": "assistant", "content": full_msg})
                            else:
                                msg = "I'm having trouble diagnosing with these symptoms. Please provide more details."
                                st.markdown(msg)
                                st.session_state.chat_messages.append({"role": "assistant", "content": msg})
                                
    # -------------------------------------------------------------
    # Display Results ONLY if Diagnosis is Complete
    # -------------------------------------------------------------
    if st.session_state.current_result and st.session_state.diagnosis_complete:
        result = st.session_state.current_result
        
        st.divider()
        st.header("📋 Diagnosis Results")
        
        if result['symptoms']:
            display_symptoms(result['symptoms'])
            st.markdown("")
            
        if result.get('low_confidence_warning'):
            st.error("""
            ⚠️ **CRITICAL: LOW CONFIDENCE PREDICTION**
            
            The predictions below have very low confidence and may not be accurate.
            **MEDICAL CONSULTATION IS REQUIRED** - Do not rely on these predictions alone.
            """)
            st.markdown(result.get('explanation', ''))
            
            if result.get('predictions'):
                st.divider()
                st.subheader("⚠️ Low Confidence Possibilities")
                st.caption("These are statistical possibilities only - NOT a diagnosis")
                for i, (disease, prob) in enumerate(result['predictions'][:3], 1):
                    st.warning(f"**{i}. {disease}** - {prob*100:.1f}% probability (LOW CONFIDENCE)")
            
            # Show Disease Data Tab if Available
            if 'treatment_plan' in result or 'structured_context' in result:
                st.divider()
                st.info("📚 **Additional Medical Information** (for reference only - consult a doctor)")
                tab_disease = st.tabs(["🔬 Disease Data"])[0]
                with tab_disease:
                    categorized_nli = {
                        "Identified Symptoms": result.get('symptoms_nli', []),
                        "AI Explanation": result.get('explanation_nli', [])
                    }
                    structured_context = result.get('structured_context', {})
                    source_url = result.get('source_url', '#')
                    render_interactive_nli_ui(categorized_nli, structured_context, source_url)
            
            st.divider()
            st.error("""
            🚨 **MANDATORY MEDICAL CONSULTATION REQUIRED**
            This is an AI-based educational tool with LOW CONFIDENCE in this prediction. 
            **You MUST consult a qualified healthcare provider** for proper diagnosis and treatment.
            """)
        
        elif result.get('predictions') and len(result['predictions']) > 0:
            st.markdown('<div class="section-header">📋 Step 2: Analysis Results</div>', unsafe_allow_html=True)
            
            top_disease = result['predictions'][0][0]
            top_prob = result['predictions'][0][1]
            
            if top_prob > 0.75:
                confidence_class = "confidence-high"
                confidence_emoji = "✅"
                confidence_text = "High Confidence"
                confidence_color = "#2E7D32"
            elif top_prob > 0.50:
                confidence_class = "confidence-medium"
                confidence_emoji = "⚠️"
                confidence_text = "Medium Confidence"
                confidence_color = "#F57C00"
            else:
                confidence_class = "confidence-low"
                confidence_emoji = "❌"
                confidence_text = "Low Confidence"
                confidence_color = "#C62828"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                padding: 1.5rem;
                border-radius: 12px;
                margin: 1rem 0;
                border-left: 5px solid {confidence_color};
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            ">
                <h3 style="margin: 0 0 0.5rem 0; color: #1565C0;">
                    {confidence_emoji} Top Prediction: {top_disease}
                </h3>
                <p style="margin: 0; font-size: 1.1rem;">
                    <span style="color: {confidence_color}; font-weight: 600;">
                        Confidence: {top_prob*100:.1f}%
                    </span>
                    <span style="color: #666; margin-left: 1rem;">
                        ({confidence_text})
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if 'nli_verification' in result:
                st.markdown("---")
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin: 1rem 0;
                    border-left: 5px solid #4CAF50;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                ">
                    <h3 style="margin: 0; color: #2E7D32;">
                        🔍 Information Verification & Accuracy Check
                    </h3>
                    <p style="margin: 0.5rem 0 0 0; color: #555; font-size: 0.95rem;">
                        AI-powered verification using Natural Language Inference (NLI)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                verification = result['nli_verification']
                verification_score = verification['verification_score']
                score_percentage = int(verification_score * 100)
                
                if verification_score >= 0.67:
                    score_color = "#4CAF50"
                    score_emoji = "✅"
                    score_label = "HIGH CONFIDENCE"
                elif verification_score >= 0.33:
                    score_color = "#FF9800"
                    score_emoji = "⚠️"
                    score_label = "MEDIUM CONFIDENCE"
                else:
                    score_color = "#F44336"
                    score_emoji = "❌"
                    score_label = "LOW CONFIDENCE"
                
                st.markdown(f"""
                <div style="
                    background: white;
                    padding: 1.2rem;
                    border-radius: 10px;
                    margin: 1rem 0;
                    border: 2px solid {score_color};
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <span style="font-size: 1.1rem; font-weight: 600; color: #333;">
                            {score_emoji} Overall Verification Score
                        </span>
                        <span style="font-size: 1.3rem; font-weight: 700; color: {score_color};">
                            {score_percentage}%
                        </span>
                    </div>
                    <div style="background: #f0f0f0; border-radius: 10px; height: 20px; overflow: hidden;">
                        <div style="
                            background: linear-gradient(90deg, {score_color}, {score_color}dd);
                            width: {score_percentage}%;
                            height: 100%;
                            border-radius: 10px;
                            transition: width 0.3s ease;
                        "></div>
                    </div>
                    <div style="margin-top: 0.5rem; text-align: center;">
                        <span style="display: inline-block; background: {score_color}22; color: {score_color}; padding: 0.3rem 1rem; border-radius: 15px; font-weight: 600; font-size: 0.85rem;">
                            {score_label}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if verification['overall_verified']:
                    st.success(f"✅ **{verification['summary']}**")
                else:
                    st.warning(f"⚠️ **{verification['summary']}**")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 📋 Detailed Verification Checks")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    symptom_check = verification['symptom_match']
                    if symptom_check['verified']:
                        card_bg, card_border, status_text, status_color = "#E8F5E9", "#4CAF50", "✅ Verified", "#2E7D32"
                    else:
                        card_bg, card_border, status_text, status_color = "#FFF3E0", "#FF9800", "⚠️ Review Needed", "#E65100"
                    
                    st.markdown(f"""
                    <div style="background: {card_bg}; padding: 1rem; border-radius: 10px; border-left: 4px solid {card_border}; margin-bottom: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🩺</div>
                        <div style="font-weight: 600; color: #333; margin-bottom: 0.5rem;">Symptom Match</div>
                        <div style="font-weight: 700; color: {status_color}; font-size: 0.95rem; margin-bottom: 0.5rem;">{status_text}</div>
                        <div style="background: white; padding: 0.4rem 0.6rem; border-radius: 5px; font-size: 0.85rem; color: #666; margin-bottom: 0.5rem;">Confidence: {int(symptom_check['confidence']*100)}%</div>
                        <div style="background: white; padding: 0.4rem 0.6rem; border-radius: 5px; font-size: 0.8rem; color: #666;"><strong>Status:</strong> {symptom_check['label']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("📖 View Reasoning"):
                        st.write(symptom_check['reasoning'])
                
                with col2:
                    treatment_check = verification['treatment_check']
                    if treatment_check['verified']:
                        card_bg, card_border, status_text, status_color = "#E8F5E9", "#4CAF50", "✅ Appropriate", "#2E7D32"
                    else:
                        card_bg, card_border, status_text, status_color = "#FFF3E0", "#FF9800", "⚠️ Review Needed", "#E65100"
                    
                    st.markdown(f"""
                    <div style="background: {card_bg}; padding: 1rem; border-radius: 10px; border-left: 4px solid {card_border}; margin-bottom: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💊</div>
                        <div style="font-weight: 600; color: #333; margin-bottom: 0.5rem;">Treatment Plan</div>
                        <div style="font-weight: 700; color: {status_color}; font-size: 0.95rem; margin-bottom: 0.5rem;">{status_text}</div>
                        <div style="background: white; padding: 0.4rem 0.6rem; border-radius: 5px; font-size: 0.85rem; color: #666; margin-bottom: 0.5rem;">Confidence: {int(treatment_check['confidence']*100)}%</div>
                        <div style="background: white; padding: 0.4rem 0.6rem; border-radius: 5px; font-size: 0.8rem; color: #666;"><strong>Status:</strong> {treatment_check['label']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("📖 View Reasoning"):
                        st.write(treatment_check['reasoning'])
                
                with col3:
                    medication_check = verification['medication_check']
                    if medication_check['verified']:
                        card_bg, card_border, status_text, status_color = "#E8F5E9", "#4CAF50", "✅ Safe", "#2E7D32"
                    else:
                        card_bg, card_border, status_text, status_color = "#FFF3E0", "#FF9800", "⚠️ Consult Doctor", "#E65100"
                    
                    st.markdown(f"""
                    <div style="background: {card_bg}; padding: 1rem; border-radius: 10px; border-left: 4px solid {card_border}; margin-bottom: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💉</div>
                        <div style="font-weight: 600; color: #333; margin-bottom: 0.5rem;">Medication Safety</div>
                        <div style="font-weight: 700; color: {status_color}; font-size: 0.95rem; margin-bottom: 0.5rem;">{status_text}</div>
                        <div style="background: white; padding: 0.4rem 0.6rem; border-radius: 5px; font-size: 0.85rem; color: #666; margin-bottom: 0.5rem;">Confidence: {int(medication_check['confidence']*100)}%</div>
                        <div style="background: white; padding: 0.4rem 0.6rem; border-radius: 5px; font-size: 0.8rem; color: #666;"><strong>Status:</strong> {medication_check['label']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("📖 View Reasoning"):
                        st.write(medication_check['reasoning'])
                
                if verification.get('contradictions'):
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.error("🚨 **Contradictions Detected in Medical Information**")
                    for i, contradiction in enumerate(verification['contradictions'], 1):
                        st.markdown(f"""
                        <div style="background: #FFEBEE; padding: 1rem; border-radius: 8px; border-left: 4px solid #F44336; margin: 0.5rem 0;">
                            <strong>Contradiction {i}:</strong> {contradiction['type']}<br>
                            <span style="color: #666; font-size: 0.9rem;">{contradiction['details']}</span><br>
                            <span style="color: #999; font-size: 0.85rem;">Confidence: {int(contradiction.get('confidence', 0)*100)}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.info("""
                **ℹ️ About NLI Verification:**
                Natural Language Inference (NLI) is an AI technique that verifies the logical consistency between your symptoms and descriptions, treatments, and suggested medications to ensure accuracy.
                """)
                st.markdown("---")
            
            has_rag = 'treatment_plan' in result or 'structured_context' in result
            st.markdown("### 📑 Detailed Information")
            
            if has_rag:
                tab1, tab_disease = st.tabs(["📊 All Predictions", "🔬 Disease Data"])
                
                with tab1:
                    st.markdown("#### Probability Distribution")
                    fig = create_probability_chart(result['predictions'])
                    st.plotly_chart(fig, width='stretch')
                    st.markdown("#### All Predictions")
                    display_predictions(result['predictions'])
                    
                with tab_disease:
                    categorized_nli = {
                        "Identified Symptoms": result.get('symptoms_nli', []),
                        "AI Explanation": result.get('explanation_nli', [])
                    }
                    structured_context = result.get('structured_context', {})
                    source_url = result.get('source_url', '#')
                    render_interactive_nli_ui(categorized_nli, structured_context, source_url)
            else:
                tab1, tab_explanation = st.tabs(["📊 Predictions", "💬 Explanation"])
                with tab1:
                    st.markdown("#### Probability Distribution")
                    fig = create_probability_chart(result['predictions'])
                    st.plotly_chart(fig, width='stretch')
                    st.markdown("#### All Predictions")
                    display_predictions(result['predictions'])
                with tab_explanation:
                    st.info(result['explanation'])

            st.divider()
            st.warning("""
            ⚠️ **MEDICAL DISCLAIMER**
            This is an AI-based educational tool and NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns.
            """)
        else:
            st.error("⚠️ **Unable to Make Prediction**")
            st.warning("""
            This symptom combination is not well-represented in the training data. 
            The model cannot make a reliable prediction for these symptoms.
            **Please consult a healthcare professional for proper diagnosis.**
            """)
            if result.get('explanation'):
                st.info(result['explanation'])
    
    if st.session_state.diagnosis_history:
        st.divider()
        st.header("📜 Diagnosis History")
        with st.expander(f"View {len(st.session_state.diagnosis_history)} previous diagnoses"):
            for i, item in enumerate(reversed(st.session_state.diagnosis_history), 1):
                st.markdown(f"**#{len(st.session_state.diagnosis_history) - i + 1}:** {item['input'][:100]}...")
                if item['result']['predictions']:
                    top = item['result']['predictions'][0]
                    st.markdown(f"→ **{top[0]}** ({top[1]*100:.1f}%)")
                st.markdown("---")

else:
    st.error("Failed to initialize chatbot. Please check the configuration.")
    st.info("""
    **Setup Instructions:**
    1. Create a `.env` file with your Gemini API key
    2. Train the model by running `03_bayesian_network_binary.ipynb`
    3. Restart this app
    """)

st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Powered by Bayesian Networks + Google Gemini AI | 
    <a href="https://github.com" target="_blank">GitHub</a> | 
    Built with Streamlit
</div>
""", unsafe_allow_html=True)