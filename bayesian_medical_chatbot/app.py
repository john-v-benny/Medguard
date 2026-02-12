"""
Streamlit Web UI for Hybrid Medical Chatbot
Combines Bayesian Network (hallucination-free predictions) with Gemini API (natural language)
"""

import streamlit as st
import os
from dotenv import load_dotenv
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

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'diagnosis_history' not in st.session_state:
    st.session_state.diagnosis_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'followup_answers' not in st.session_state:
    st.session_state.followup_answers = {}
if 'current_symptoms' not in st.session_state:
    st.session_state.current_symptoms = {}
if 'user_input_text' not in st.session_state:
    st.session_state.user_input_text = ""

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
    
    # Color code by confidence
    colors = []
    for prob in probabilities:
        if prob > 75:
            colors.append('#2e7d32')  # Green
        elif prob > 50:
            colors.append('#f57c00')  # Orange
        else:
            colors.append('#1976d2')  # Blue
    
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
            use_container_width=True,
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
    
    # API Key input (optional override)
    api_key_input = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        help="Leave empty to use .env file",
        placeholder="AIzaSy..."
    )
    
    if api_key_input:
        os.environ['GEMINI_API_KEY'] = api_key_input
    
    st.divider()
    
    # About section
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot uses:
    - **Bayesian Network** for accurate, hallucination-free predictions
    - **Gemini API** for natural language understanding
    - **131 symptoms** and **41 diseases**
    
    **How it works:**
    1. Describe your symptoms naturally
    2. AI extracts symptoms from your text
    3. Bayesian Network predicts diseases
    4. Get results with confidence scores
    """)
    
    st.divider()
    
    # Statistics
    st.header("📊 Statistics")
    if st.session_state.chatbot:
        st.metric("Diseases", len(st.session_state.chatbot.diseases))
        st.metric("Symptoms", len(st.session_state.chatbot.symptom_cols))
        st.metric("Diagnoses Made", len(st.session_state.diagnosis_history))
    
    st.divider()
    
    # Clear history
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.diagnosis_history = []
        st.session_state.current_result = None
        st.rerun()

# Initialize chatbot
if st.session_state.chatbot is None:
    with st.spinner("Initializing chatbot..."):
        st.session_state.chatbot = initialize_chatbot()

# Main content
if st.session_state.chatbot:
    # Input section with better organization
    st.markdown('<div class="section-header">💬 Step 1: Describe Your Symptoms</div>', unsafe_allow_html=True)
    
    # Help text
    st.info("📝 **How to use:** Describe your symptoms in plain English. The AI will automatically extract and analyze them.")
    
    # Example symptoms in an expander
    with st.expander("💡 See Example Symptoms"):
        st.markdown("""
        **Example 1:** "I have a high fever, severe headache, and I've been vomiting. I also have chills and muscle pain."
        
        **Example 2:** "I'm experiencing increased thirst, frequent urination, and constant fatigue."
        
        **Example 3:** "I have a persistent cough, runny nose, and sore throat for the past 3 days."
        """)
    
    # Input area
    user_input = st.text_area(
        "Enter your symptoms here:",
        placeholder="Example: I have a high fever, severe headache, and I've been vomiting...",
        height=120,
        key="symptom_input",
        help="Be as specific as possible about your symptoms"
    )
    
    # Diagnose button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        diagnose_button = st.button(
            "🔍 Analyze Symptoms",
            type="primary",
            use_container_width=True,
            disabled=not user_input.strip()
        )
    
    st.divider()
    
    # Process diagnosis
    if diagnose_button and user_input.strip():
        if user_input.strip():
            with st.spinner("Analyzing symptoms..."):
                try:
                    result = st.session_state.chatbot.diagnose(user_input)
                    st.session_state.current_result = result
                    
                    # Store current symptoms for follow-up questions
                    if result.get('symptoms'):
                        st.session_state.current_symptoms = {s: 1 for s in result['symptoms']}
                    
                    # Clear previous follow-up answers
                    st.session_state.followup_answers = {}
                    
                    st.session_state.diagnosis_history.append({
                        'input': user_input,
                        'result': result
                    })
                except Exception as e:
                    st.error(f"Error during diagnosis: {e}")
        else:
            st.warning("Please describe your symptoms first!")
    
    # Display results
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        st.divider()
        st.header("📋 Diagnosis Results")
        
        # Symptoms section
        if result['symptoms']:
            display_symptoms(result['symptoms'])
            st.markdown("")
        else:
            st.warning("⚠️ No symptoms were identified from your description. Please be more specific.")
        
        # SAFE RESPONSE STRATEGY: Always show predictions with appropriate warnings
        # Check if this is a low confidence warning case
        if result.get('low_confidence_warning'):
            # Very low confidence - show predictions with STRONG warnings
            st.error("""
            ⚠️ **CRITICAL: LOW CONFIDENCE PREDICTION**
            
            The predictions below have very low confidence and may not be accurate.
            **MEDICAL CONSULTATION IS REQUIRED** - Do not rely on these predictions alone.
            """)
            
            # Show the explanation with warnings
            st.markdown(result.get('explanation', ''))
            
            # Show predictions in a warning box
            if result.get('predictions'):
                st.divider()
                st.subheader("⚠️ Low Confidence Possibilities")
                st.caption("These are statistical possibilities only - NOT a diagnosis")
                
                # Show top 3 predictions with warning styling
                for i, (disease, prob) in enumerate(result['predictions'][:3], 1):
                    st.warning(f"**{i}. {disease}** - {prob*100:.1f}% probability (LOW CONFIDENCE)")
            
            # Show RAG information if available
            if 'treatment_plan' in result:
                st.divider()
                st.info("📚 **Additional Medical Information** (for reference only - consult a doctor)")
                
                # Create tabs for RAG information
                tab1, tab2, tab3, tab4 = st.tabs([
                    "💊 Treatment Info",
                    "💉 Medications",
                    "🔔 Next Steps",
                    "🛡️ Prevention"
                ])
                
                with tab1:
                    st.markdown("### Treatment Information")
                    st.caption("⚠️ This information is for educational purposes. Consult a healthcare professional.")
                    if 'treatment_plan' in result:
                        st.markdown(result['treatment_plan'])
                    else:
                        st.info("Treatment information not available.")
                
                with tab2:
                    st.markdown("### Medication Information")
                    st.caption("⚠️ NEVER self-medicate. All medications must be prescribed by a doctor.")
                    if 'medications' in result:
                        st.markdown(result['medications'])
                    else:
                        st.info("Medication information not available.")
                
                with tab3:
                    st.markdown("### Recommended Next Steps")
                    if 'next_steps' in result:
                        st.markdown(result['next_steps'])
                    else:
                        st.info("Please consult a healthcare professional for guidance.")
                
                with tab4:
                    st.markdown("### Prevention & Risk Reduction")
                    if 'prevention' in result:
                        st.markdown(result['prevention'])
                    else:
                        st.info("Prevention information not available.")
                
                # Show source
                if 'source_url' in result:
                    st.caption(f"**Source:** [{result.get('source_name', 'MedlinePlus')}]({result['source_url']})")
            
            # Show follow-up questions as optional (if any)
            if result.get('followup_questions'):
                st.divider()
                st.markdown("### 🔍 Optional: Answer Questions to Refine Prediction")
                st.caption("*Note: Even with more information, medical consultation is still required*")
                
                # Display questions
                for i, q in enumerate(result['followup_questions']):
                    symptom = q['symptom']
                    question = q['question']
                    
                    st.markdown(f"**{i+1}. {question}**")
                    
                    col1, col2, col3 = st.columns([1, 1, 4])
                    
                    with col1:
                        if st.button("✅ Yes", key=f"yes_{symptom}_low"):
                            st.session_state.followup_answers[symptom] = 1
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ No", key=f"no_{symptom}_low"):
                            st.session_state.followup_answers[symptom] = 0
                            st.rerun()
                    
                    with col3:
                        if symptom in st.session_state.followup_answers:
                            answer = "Yes" if st.session_state.followup_answers[symptom] == 1 else "No"
                            st.markdown(f"*Answer: {answer}*")
                
                # Update button (this code path shouldn't normally execute now)
                if st.session_state.followup_answers:
                    st.divider()
                    st.info("This is a safe response with low confidence. Medical consultation is required regardless of additional information.")

            
            # Show follow-up questions as optional
            if result.get('followup_questions'):
                st.divider()
                st.markdown("### 🔍 Optional: Answer Questions to Refine Prediction")
                st.caption("*Note: Even with more information, medical consultation is still required*")
                
                # Display questions
                for i, q in enumerate(result['followup_questions']):
                    symptom = q['symptom']
                    question = q['question']
                    
                    st.markdown(f"**{i+1}. {question}**")
                    
                    col1, col2, col3 = st.columns([1, 1, 4])
                    
                    with col1:
                        if st.button("✅ Yes", key=f"yes_{symptom}_low"):
                            st.session_state.followup_answers[symptom] = 1
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ No", key=f"no_{symptom}_low"):
                            st.session_state.followup_answers[symptom] = 0
                            st.rerun()
                    
                    with col3:
                        if symptom in st.session_state.followup_answers:
                            answer = "Yes" if st.session_state.followup_answers[symptom] == 1 else "No"
                            st.markdown(f"*Answer: {answer}*")
                
                # Update button
                if st.session_state.followup_answers:
                    st.divider()
                    if st.button("🔄 Update Prediction with Answers", type="primary", use_container_width=True):
                        with st.spinner("Updating prediction..."):
                            try:
                                # Combine original symptoms with follow-up answers
                                all_symptoms = st.session_state.current_symptoms.copy()
                                all_symptoms.update(st.session_state.followup_answers)
                                
                                # Make new prediction
                                predictions = st.session_state.chatbot.predict_disease(all_symptoms, top_n=5)
                                
                                if predictions and len(predictions) > 0:
                                    # Get explanation
                                    symptom_list = [s for s, v in all_symptoms.items() if v == 1]
                                    explanation = st.session_state.chatbot.explain_results(predictions, symptom_list)
                                    
                                    # Check new confidence
                                    new_confidence = predictions[0][1]
                                    
                                    # Create updated result
                                    updated_result = {
                                        'symptoms': symptom_list,
                                        'predictions': predictions,
                                        'explanation': explanation,
                                        'requires_followup': new_confidence < st.session_state.chatbot.FOLLOWUP_THRESHOLD,
                                        'followup_questions': [],
                                        'confidence': new_confidence,
                                        'low_confidence_warning': new_confidence < st.session_state.chatbot.MINIMUM_CONFIDENCE
                                    }
                                    st.success(f"✅ Prediction updated! Confidence: {new_confidence*100:.1f}%")
                                    
                                    st.session_state.current_result = updated_result
                                    st.session_state.followup_answers = {}
                                    st.rerun()
                                else:
                                    st.warning("""
                                    ⚠️ **Unable to Generate Prediction**
                                    
                                    The symptom combination is unusual. Please consult a healthcare professional for proper diagnosis.
                                    
                                    **Recommended Actions:**
                                    - 🏥 Schedule an appointment with a doctor
                                    - 📝 Bring your symptom list
                                    - 🔍 Get proper medical tests
                                    """)
                            
                            except Exception as e:
                                st.error(f"Error updating prediction: {str(e)}")
            
            # Medical disclaimer
            st.divider()
            st.error("""
            🚨 **MANDATORY MEDICAL CONSULTATION REQUIRED**
            
            This is an AI-based educational tool with LOW CONFIDENCE in this prediction. 
            **You MUST consult a qualified healthcare provider** for proper diagnosis and treatment.
            Do NOT make any medical decisions based on this information alone.
            """)
        
        # Normal predictions with good confidence
        elif result.get('predictions') and len(result['predictions']) > 0:
            # Display identified symptoms
            display_symptoms(result['symptoms'])
            
            # Results section header
            st.markdown('<div class="section-header">📋 Step 2: Analysis Results</div>', unsafe_allow_html=True)
            
            # Top prediction card with better styling
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
            
            # Enhanced top prediction card
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
            
            # Enhanced NLI Verification Section
            if 'nli_verification' in result:
                st.markdown("---")
                
                # Prominent header with icon
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
                
                # Overall verification score with progress bar
                verification_score = verification['verification_score']
                score_percentage = int(verification_score * 100)
                
                # Color based on score
                if verification_score >= 0.67:
                    score_color = "#4CAF50"  # Green
                    score_emoji = "✅"
                    score_label = "HIGH CONFIDENCE"
                elif verification_score >= 0.33:
                    score_color = "#FF9800"  # Orange
                    score_emoji = "⚠️"
                    score_label = "MEDIUM CONFIDENCE"
                else:
                    score_color = "#F44336"  # Red
                    score_emoji = "❌"
                    score_label = "LOW CONFIDENCE"
                
                # Display overall score
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
                    <div style="
                        background: #f0f0f0;
                        border-radius: 10px;
                        height: 20px;
                        overflow: hidden;
                    ">
                        <div style="
                            background: linear-gradient(90deg, {score_color}, {score_color}dd);
                            width: {score_percentage}%;
                            height: 100%;
                            border-radius: 10px;
                            transition: width 0.3s ease;
                        "></div>
                    </div>
                    <div style="margin-top: 0.5rem; text-align: center;">
                        <span style="
                            display: inline-block;
                            background: {score_color}22;
                            color: {score_color};
                            padding: 0.3rem 1rem;
                            border-radius: 15px;
                            font-weight: 600;
                            font-size: 0.85rem;
                        ">
                            {score_label}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Summary message
                if verification['overall_verified']:
                    st.success(f"✅ **{verification['summary']}**")
                else:
                    st.warning(f"⚠️ **{verification['summary']}**")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Detailed verification checks with enhanced cards
                st.markdown("#### 📋 Detailed Verification Checks")
                
                # Detailed checks in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    symptom_check = verification['symptom_match']
                    
                    # Determine card styling
                    if symptom_check['verified']:
                        card_bg = "#E8F5E9"
                        card_border = "#4CAF50"
                        status_text = "✅ Verified"
                        status_color = "#2E7D32"
                    else:
                        card_bg = "#FFF3E0"
                        card_border = "#FF9800"
                        status_text = "⚠️ Review Needed"
                        status_color = "#E65100"
                    
                    st.markdown(f"""
                    <div style="
                        background: {card_bg};
                        padding: 1rem;
                        border-radius: 10px;
                        border-left: 4px solid {card_border};
                        margin-bottom: 1rem;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    ">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🩺</div>
                        <div style="font-weight: 600; color: #333; margin-bottom: 0.5rem;">Symptom Match</div>
                        <div style="
                            font-weight: 700;
                            color: {status_color};
                            font-size: 0.95rem;
                            margin-bottom: 0.5rem;
                        ">
                            {status_text}
                        </div>
                        <div style="
                            background: white;
                            padding: 0.4rem 0.6rem;
                            border-radius: 5px;
                            font-size: 0.85rem;
                            color: #666;
                            margin-bottom: 0.5rem;
                        ">
                            Confidence: {int(symptom_check['confidence']*100)}%
                        </div>
                        <div style="
                            background: white;
                            padding: 0.4rem 0.6rem;
                            border-radius: 5px;
                            font-size: 0.8rem;
                            color: #666;
                        ">
                            <strong>Status:</strong> {symptom_check['label']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📖 View Reasoning"):
                        st.write(symptom_check['reasoning'])
                
                with col2:
                    treatment_check = verification['treatment_check']
                    
                    # Determine card styling
                    if treatment_check['verified']:
                        card_bg = "#E8F5E9"
                        card_border = "#4CAF50"
                        status_text = "✅ Appropriate"
                        status_color = "#2E7D32"
                    else:
                        card_bg = "#FFF3E0"
                        card_border = "#FF9800"
                        status_text = "⚠️ Review Needed"
                        status_color = "#E65100"
                    
                    st.markdown(f"""
                    <div style="
                        background: {card_bg};
                        padding: 1rem;
                        border-radius: 10px;
                        border-left: 4px solid {card_border};
                        margin-bottom: 1rem;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    ">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💊</div>
                        <div style="font-weight: 600; color: #333; margin-bottom: 0.5rem;">Treatment Plan</div>
                        <div style="
                            font-weight: 700;
                            color: {status_color};
                            font-size: 0.95rem;
                            margin-bottom: 0.5rem;
                        ">
                            {status_text}
                        </div>
                        <div style="
                            background: white;
                            padding: 0.4rem 0.6rem;
                            border-radius: 5px;
                            font-size: 0.85rem;
                            color: #666;
                            margin-bottom: 0.5rem;
                        ">
                            Confidence: {int(treatment_check['confidence']*100)}%
                        </div>
                        <div style="
                            background: white;
                            padding: 0.4rem 0.6rem;
                            border-radius: 5px;
                            font-size: 0.8rem;
                            color: #666;
                        ">
                            <strong>Status:</strong> {treatment_check['label']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📖 View Reasoning"):
                        st.write(treatment_check['reasoning'])
                
                with col3:
                    medication_check = verification['medication_check']
                    
                    # Determine card styling
                    if medication_check['verified']:
                        card_bg = "#E8F5E9"
                        card_border = "#4CAF50"
                        status_text = "✅ Safe"
                        status_color = "#2E7D32"
                    else:
                        card_bg = "#FFF3E0"
                        card_border = "#FF9800"
                        status_text = "⚠️ Consult Doctor"
                        status_color = "#E65100"
                    
                    st.markdown(f"""
                    <div style="
                        background: {card_bg};
                        padding: 1rem;
                        border-radius: 10px;
                        border-left: 4px solid {card_border};
                        margin-bottom: 1rem;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    ">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💉</div>
                        <div style="font-weight: 600; color: #333; margin-bottom: 0.5rem;">Medication Safety</div>
                        <div style="
                            font-weight: 700;
                            color: {status_color};
                            font-size: 0.95rem;
                            margin-bottom: 0.5rem;
                        ">
                            {status_text}
                        </div>
                        <div style="
                            background: white;
                            padding: 0.4rem 0.6rem;
                            border-radius: 5px;
                            font-size: 0.85rem;
                            color: #666;
                            margin-bottom: 0.5rem;
                        ">
                            Confidence: {int(medication_check['confidence']*100)}%
                        </div>
                        <div style="
                            background: white;
                            padding: 0.4rem 0.6rem;
                            border-radius: 5px;
                            font-size: 0.8rem;
                            color: #666;
                        ">
                            <strong>Status:</strong> {medication_check['label']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📖 View Reasoning"):
                        st.write(medication_check['reasoning'])
                
                # Show contradictions if any
                if verification.get('contradictions'):
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.error("🚨 **Contradictions Detected in Medical Information**")
                    for i, contradiction in enumerate(verification['contradictions'], 1):
                        st.markdown(f"""
                        <div style="
                            background: #FFEBEE;
                            padding: 1rem;
                            border-radius: 8px;
                            border-left: 4px solid #F44336;
                            margin: 0.5rem 0;
                        ">
                            <strong>Contradiction {i}:</strong> {contradiction['type']}<br>
                            <span style="color: #666; font-size: 0.9rem;">{contradiction['details']}</span><br>
                            <span style="color: #999; font-size: 0.85rem;">Confidence: {int(contradiction.get('confidence', 0)*100)}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Information box about NLI
                st.info("""
                **ℹ️ About NLI Verification:**
                
                Natural Language Inference (NLI) is an AI technique that verifies the logical consistency between:
                - Your symptoms and the predicted disease description
                - The disease and recommended treatments
                - The condition and suggested medications
                
                This helps ensure the information provided is accurate and consistent with medical knowledge.
                """)
                
                st.markdown("---")
            
            # Tabs for different views
            # Check if RAG information is available
            has_rag = 'treatment_plan' in result
            
            st.markdown("### 📑 Detailed Information")
            
            if has_rag:
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 All Predictions", 
                    "📖 Explanation", 
                    "💊 Treatment",
                    "💉 Medications",
                    "🔔 Next Steps",
                    "🛡️ Prevention"
                ])
            else:
                tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📋 Details", "💬 Explanation"])
            
            with tab1:
                st.markdown("#### Probability Distribution")
                fig = create_probability_chart(result['predictions'])
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### All Predictions")
                display_predictions(result['predictions'])
            
            with tab2:
                st.subheader("AI Explanation")
                
                # Show RAG explanation if available, otherwise Gemini explanation
                if has_rag and 'rag_explanation' in result:
                    st.markdown(result['rag_explanation'])
                    
                    # Show disease description
                    if 'disease_description' in result:
                        st.divider()
                        st.markdown("**Disease Overview:**")
                        st.info(result['disease_description'])
                    
                    # Show source citation
                    if 'source_url' in result:
                        st.divider()
                        st.markdown(f"**Source:** [{result.get('source_name', 'MedlinePlus')}]({result['source_url']})")
                else:
                    st.info(result['explanation'])
            
            # RAG-specific tabs
            if has_rag:
                with tab3:
                    st.subheader("💊 Treatment Plan")
                    if 'treatment_plan' in result:
                        st.markdown(result['treatment_plan'])
                    else:
                        st.info("Treatment information not available. Please consult a healthcare professional.")
                
                with tab4:
                    st.subheader("💉 Medication Information")
                    if 'medications' in result:
                        st.markdown(result['medications'])
                        
                        # Show complications if available
                        if 'complications' in result and result['complications']:
                            st.divider()
                            st.warning("**Potential Complications:**")
                            st.markdown(result['complications'])
                    else:
                        st.info("Medication information not available. Please consult a healthcare professional.")
                
                with tab5:
                    st.subheader("🔔 Next Steps")
                    if 'next_steps' in result:
                        st.markdown(result['next_steps'])
                    else:
                        st.info("Please consult a healthcare professional for guidance on next steps.")
                
                with tab6:
                    st.subheader("🛡️ Prevention & Risk Reduction")
                    if 'prevention' in result:
                        st.markdown(result['prevention'])
                    else:
                        st.info("Prevention information not available.")
                    
                    # Show source
                    if 'source_url' in result:
                        st.divider()
                        st.caption(f"**Source:** [{result.get('source_name', 'MedlinePlus')}]({result['source_url']})")
            
            # Follow-up questions section (for medium confidence predictions)
            if result.get('requires_followup') and result.get('followup_questions'):
                st.divider()
                st.markdown("### 🔍 Improve Prediction Accuracy")
                
                # Show info based on confidence level
                if top_prob < 0.50:
                    st.warning(f"""
                    ⚠️ **Medium-Low Confidence ({top_prob*100:.1f}%)**
                    
                    The current prediction has medium-low confidence. Answering the questions below 
                    can help improve the accuracy of the diagnosis.
                    """)
                else:
                    st.info(f"""
                    📊 **Confidence: {top_prob*100:.1f}%**
                    
                    You can optionally answer these questions to further refine the prediction.
                    """)
                
                # Display follow-up questions
                for i, q in enumerate(result['followup_questions']):
                    symptom = q['symptom']
                    question = q['question']
                    
                    st.markdown(f"**{i+1}. {question}**")
                    
                    col1, col2, col3 = st.columns([1, 1, 4])
                    
                    with col1:
                        if st.button("✅ Yes", key=f"yes_{symptom}_normal"):
                            st.session_state.followup_answers[symptom] = 1
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ No", key=f"no_{symptom}_normal"):
                            st.session_state.followup_answers[symptom] = 0
                            st.rerun()
                    
                    with col3:
                        if symptom in st.session_state.followup_answers:
                            answer = "Yes" if st.session_state.followup_answers[symptom] == 1 else "No"
                            st.markdown(f"*Answer: {answer}*")
                    
                    st.markdown("")
                
                # Update prediction button
                if st.session_state.followup_answers:
                    st.divider()
                    if st.button("🔄 Update Prediction with Answers", type="primary", use_container_width=True, key="update_normal_conf"):
                        with st.spinner("Updating prediction..."):
                            try:
                                # Combine original symptoms with follow-up answers
                                all_symptoms = st.session_state.current_symptoms.copy()
                                all_symptoms.update(st.session_state.followup_answers)
                                
                                # Make new prediction
                                predictions = st.session_state.chatbot.predict_disease(all_symptoms, top_n=5)
                                
                                if predictions and len(predictions) > 0:
                                    # Get explanation
                                    symptom_list = [s for s, v in all_symptoms.items() if v == 1]
                                    explanation = st.session_state.chatbot.explain_results(predictions, symptom_list)
                                    
                                    # Check new confidence
                                    new_confidence = predictions[0][1]
                                    top_disease = predictions[0][0]
                                    
                                    # Get RAG information if available
                                    rag_info = None
                                    if st.session_state.chatbot.use_rag and st.session_state.chatbot.rag:
                                        try:
                                            rag_info = st.session_state.chatbot.rag.explain_diagnosis(
                                                disease=top_disease,
                                                confidence=new_confidence,
                                                symptoms=symptom_list
                                            )
                                        except Exception as e:
                                            st.warning(f"Could not retrieve detailed medical information: {e}")
                                    
                                    # Create updated result
                                    updated_result = {
                                        'symptoms': symptom_list,
                                        'predictions': predictions,
                                        'explanation': explanation,
                                        'requires_followup': new_confidence < st.session_state.chatbot.FOLLOWUP_THRESHOLD,
                                        'followup_questions': [],
                                        'confidence': new_confidence,
                                        'low_confidence_warning': new_confidence < st.session_state.chatbot.MINIMUM_CONFIDENCE
                                    }
                                    
                                    # Add RAG information if available
                                    if rag_info:
                                        updated_result.update({
                                            'rag_explanation': rag_info['explanation'],
                                            'treatment_plan': rag_info['treatment_plan'],
                                            'medications': rag_info['medications'],
                                            'next_steps': rag_info['next_steps'],
                                            'prevention': rag_info['prevention'],
                                            'source_url': rag_info['source_url'],
                                            'source_name': rag_info['source_name'],
                                            'disease_description': rag_info['disease_description'],
                                            'complications': rag_info['complications']
                                        })
                                        
                                        # Add NLI verification if available
                                        if 'nli_verification' in rag_info:
                                            updated_result['nli_verification'] = rag_info['nli_verification']
                                    
                                    st.success(f"✅ Prediction updated! New confidence: {new_confidence*100:.1f}%")
                                    
                                    st.session_state.current_result = updated_result
                                    st.session_state.current_symptoms = all_symptoms
                                    st.session_state.followup_answers = {}
                                    st.rerun()
                                else:
                                    st.warning("""
                                    ⚠️ **Unable to Generate Prediction**
                                    
                                    The symptom combination is unusual. Please consult a healthcare professional for proper diagnosis.
                                    """)
                            
                            except Exception as e:
                                st.error(f"Error updating prediction: {str(e)}")
            
            # Medical disclaimer
            st.divider()
            st.warning("""
            ⚠️ **MEDICAL DISCLAIMER**
            
            This is an AI-based educational tool and NOT a substitute for professional medical advice, 
            diagnosis, or treatment. Always consult a qualified healthcare provider for any medical 
            concerns or before making health-related decisions.
            """)
        
        # Check if we need more information (old flow - kept for backward compatibility)
        elif result.get('requires_followup') and not result.get('predictions'):
            # This case should rarely happen now with safe response strategy
            st.warning("⚠️ **Need More Information for Reliable Prediction**")
            
            st.info(result.get('explanation', 'I need more information to make a reliable prediction.'))
            
            # Show follow-up questions prominently
            if result.get('followup_questions'):
                st.markdown("### 🔍 Please Answer These Questions")
                st.markdown("*Answering these questions will help me make a more accurate diagnosis.*")
                
                # Display questions with yes/no buttons
                for i, q in enumerate(result['followup_questions']):
                    symptom = q['symptom']
                    question = q['question']
                    
                    st.markdown(f"**{i+1}. {question}**")
                    
                    col1, col2, col3 = st.columns([1, 1, 4])
                    
                    with col1:
                        if st.button("✅ Yes", key=f"yes_{symptom}"):
                            st.session_state.followup_answers[symptom] = 1
                            st.session_state.current_symptoms[symptom] = 1
                    
                    with col2:
                        if st.button("❌ No", key=f"no_{symptom}"):
                            st.session_state.followup_answers[symptom] = 0
                            st.session_state.current_symptoms[symptom] = 0
                    
                    # Show current answer if exists
                    if symptom in st.session_state.followup_answers:
                        answer = "Yes" if st.session_state.followup_answers[symptom] == 1 else "No"
                        with col3:
                            st.markdown(f"*Answer: {answer}*")
                    
                    st.markdown("")
                
                # Update prediction button
                if len(st.session_state.followup_answers) > 0:
                    st.markdown("")
                    if st.button("🔄 Update Prediction with Answers", type="primary", use_container_width=True, key="update_low_conf"):
                        with st.spinner("Updating prediction..."):
                            try:
                                # Combine original symptoms with follow-up answers
                                all_symptoms = st.session_state.current_symptoms.copy()
                                
                                # Make new prediction
                                predictions = st.session_state.chatbot.predict_disease(all_symptoms, top_n=5)
                                
                                if predictions and len(predictions) > 0:
                                    # Get explanation
                                    symptom_list = [s for s, v in all_symptoms.items() if v == 1]
                                    explanation = st.session_state.chatbot.explain_results(predictions, symptom_list)
                                    
                                    # Check new confidence
                                    new_confidence = predictions[0][1]
                                    top_disease = predictions[0][0]
                                    
                                    # Get RAG information if available
                                    rag_info = None
                                    if st.session_state.chatbot.use_rag and st.session_state.chatbot.rag:
                                        try:
                                            rag_info = st.session_state.chatbot.rag.explain_diagnosis(
                                                disease=top_disease,
                                                confidence=new_confidence,
                                                symptoms=symptom_list
                                            )
                                        except Exception as e:
                                            st.warning(f"Could not retrieve detailed medical information: {e}")
                                    
                                    # Determine if we still need follow-up OR show safe response
                                    if new_confidence < st.session_state.chatbot.MINIMUM_CONFIDENCE:
                                        # SAFE RESPONSE: Show predictions with strong warnings
                                        safe_explanation = f"""⚠️ **Low Confidence Prediction - Medical Consultation Required**

Based on your symptoms, the model has identified possible conditions, but the confidence is still very low ({new_confidence*100:.1f}%). 

**Top Possibilities (Low Confidence):**
"""
                                        for i, (disease, prob) in enumerate(predictions[:3], 1):
                                            safe_explanation += f"\n{i}. {disease} ({prob*100:.1f}% probability)"
                                        
                                        safe_explanation += """

**⚠️ IMPORTANT WARNINGS:**
- These predictions have LOW confidence and may not be accurate
- Your symptoms could indicate multiple conditions
- Professional medical evaluation is ESSENTIAL
- Do NOT self-diagnose or self-medicate

**What You Should Do:**
1. 🏥 **Consult a healthcare professional** as soon as possible
2. 📝 **Bring this symptom list** to your doctor
3. 🔍 **Get proper medical tests** for accurate diagnosis
4. ⚠️ **Seek emergency care** if symptoms worsen

The model cannot provide a reliable diagnosis with the current information. Please seek professional medical help."""
                                        
                                        updated_result = {
                                            'symptoms': symptom_list,
                                            'predictions': predictions,  # Show with warnings
                                            'explanation': safe_explanation,
                                            'requires_followup': False,  # No more questions
                                            'followup_questions': [],
                                            'confidence': new_confidence,
                                            'low_confidence_warning': True  # Show safe response UI
                                        }
                                        
                                        # Add RAG info if available
                                        if rag_info:
                                            updated_result.update({
                                                'rag_explanation': rag_info['explanation'],
                                                'treatment_plan': rag_info['treatment_plan'],
                                                'medications': rag_info['medications'],
                                                'next_steps': rag_info['next_steps'],
                                                'prevention': rag_info['prevention'],
                                                'source_url': rag_info['source_url'],
                                                'source_name': rag_info['source_name'],
                                                'disease_description': rag_info['disease_description'],
                                                'complications': rag_info['complications']
                                            })
                                        
                                        st.warning(f"⚠️ Confidence is {new_confidence*100:.1f}% - showing safe response with medical consultation advice")
                                    else:
                                        # Good enough to show predictions normally
                                        updated_result = {
                                            'symptoms': symptom_list,
                                            'predictions': predictions,
                                            'explanation': explanation,
                                            'requires_followup': new_confidence < st.session_state.chatbot.FOLLOWUP_THRESHOLD,
                                            'followup_questions': [],
                                            'confidence': new_confidence
                                        }
                                        
                                        # Add RAG info if available
                                        if rag_info:
                                            updated_result.update({
                                                'rag_explanation': rag_info['explanation'],
                                                'treatment_plan': rag_info['treatment_plan'],
                                                'medications': rag_info['medications'],
                                                'next_steps': rag_info['next_steps'],
                                                'prevention': rag_info['prevention'],
                                                'source_url': rag_info['source_url'],
                                                'source_name': rag_info['source_name'],
                                                'disease_description': rag_info['disease_description'],
                                                'complications': rag_info['complications']
                                            })
                                        
                                        st.success(f"✅ Prediction updated! Confidence: {new_confidence*100:.1f}%")
                                    
                                    st.session_state.current_result = updated_result
                                    st.session_state.followup_answers = {}  # Clear answers
                                    st.rerun()
                                else:
                                    # Safe fallback - show message but don't block user
                                    st.warning("""
                                    ⚠️ **Unable to Generate Prediction**
                                    
                                    The symptom combination is unusual. Please consult a healthcare professional for proper diagnosis.
                                    
                                    **Recommended Actions:**
                                    - 🏥 Schedule an appointment with a doctor
                                    - 📝 Bring your symptom list
                                    - 🔍 Get proper medical tests
                                    """)
                            
                            except Exception as e:
                                st.error(f"Error updating prediction: {str(e)}")
            
            # Medical disclaimer
            st.divider()
            st.warning("""
            ⚠️ **MEDICAL DISCLAIMER**
            
            This is an AI-based educational tool and NOT a substitute for professional medical advice, 
            diagnosis, or treatment. Always consult a qualified healthcare provider for any medical 
            concerns or before making health-related decisions.
            """)
            # Display identified symptoms
            display_symptoms(result['symptoms'])
            
            # Results section header
            st.markdown('<div class="section-header">📋 Step 2: Analysis Results</div>', unsafe_allow_html=True)
            
            # Check if predictions exist before accessing
            if result['predictions'] and len(result['predictions']) > 0:
                # Top prediction card with better styling
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
                
                # Enhanced top prediction card
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
            else:
                # No predictions available
                st.warning("""
                ⚠️ **Unable to Make Prediction**
                
                The symptom combination you've provided is not well-represented in our training data. 
                This could mean:
                - Multiple conditions may be present
                - Rare condition not in our database
                - Symptom description may need clarification
                
                **Recommended Action:** Please consult a healthcare professional for proper diagnosis.
                """)
            
            # NLI Verification Section
            if 'nli_verification' in result:
                st.markdown("---")
                st.markdown("### ✅ Information Verification")
                
                verification = result['nli_verification']
                
                # Overall status
                if verification['overall_verified']:
                    st.success(f"✅ {verification['summary']}")
                else:
                    st.warning(f"⚠️ {verification['summary']}")
                
                # Detailed checks in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    symptom_check = verification['symptom_match']
                    if symptom_check['verified']:
                        st.metric(
                            "Symptom Match",
                            "✅ Verified",
                            f"{symptom_check['confidence']*100:.0f}% confidence"
                        )
                    else:
                        st.metric(
                            "Symptom Match",
                            "⚠️ Review",
                            f"{symptom_check['confidence']*100:.0f}% confidence"
                        )
                    with st.expander("Details"):
                        st.caption(f"**Status:** {symptom_check['label']}")
                        st.caption(f"**Reasoning:** {symptom_check['reasoning']}")
                
                with col2:
                    treatment_check = verification['treatment_check']
                    if treatment_check['verified']:
                        st.metric(
                            "Treatment",
                            "✅ Appropriate",
                            f"{treatment_check['confidence']*100:.0f}% confidence"
                        )
                    else:
                        st.metric(
                            "Treatment",
                            "⚠️ Review",
                            f"{treatment_check['confidence']*100:.0f}% confidence"
                        )
                    with st.expander("Details"):
                        st.caption(f"**Status:** {treatment_check['label']}")
                        st.caption(f"**Reasoning:** {treatment_check['reasoning']}")
                
                with col3:
                    medication_check = verification['medication_check']
                    if medication_check['verified']:
                        st.metric(
                            "Medications",
                            "✅ Safe",
                            f"{medication_check['confidence']*100:.0f}% confidence"
                        )
                    else:
                        st.metric(
                            "Medications",
                            "⚠️ Consult Doctor",
                            f"{medication_check['confidence']*100:.0f}% confidence"
                        )
                    with st.expander("Details"):
                        st.caption(f"**Status:** {medication_check['label']}")
                        st.caption(f"**Reasoning:** {medication_check['reasoning']}")
                
                # Show contradictions if any
                if verification['contradictions']:
                    st.error("⚠️ **Contradictions Detected**")
                    for contradiction in verification['contradictions']:
                        st.warning(f"- **{contradiction['type']}**: {contradiction['details']}")
                
                st.markdown("---")
            
            # Tabs for different views
            # Check if RAG information is available
            has_rag = 'treatment_plan' in result
            
            st.markdown("### 📑 Detailed Information")
            
            if has_rag:
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 All Predictions", 
                    "📖 Explanation", 
                    "💊 Treatment",
                    "💉 Medications",
                    "🔔 Next Steps",
                    "🛡️ Prevention"
                ])
            else:
                tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📋 Details", "💬 Explanation"])
            
            with tab1:
                st.markdown("#### Probability Distribution")
                fig = create_probability_chart(result['predictions'])
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### All Predictions")
                display_predictions(result['predictions'])
            
            with tab2:
                st.subheader("AI Explanation")
                
                # Show RAG explanation if available, otherwise Gemini explanation
                if has_rag and 'rag_explanation' in result:
                    st.markdown(result['rag_explanation'])
                    
                    # Show disease description
                    if 'disease_description' in result:
                        st.divider()
                        st.markdown("**Disease Overview:**")
                        st.info(result['disease_description'])
                    
                    # Show source citation
                    if 'source_url' in result:
                        st.divider()
                        st.markdown(f"**Source:** [{result.get('source_name', 'MedlinePlus')}]({result['source_url']})")
                else:
                    st.info(result['explanation'])
            
            # RAG-specific tabs
            if has_rag:
                with tab3:
                    st.subheader("💊 Treatment Plan")
                    if 'treatment_plan' in result:
                        st.markdown(result['treatment_plan'])
                    else:
                        st.info("Treatment information not available. Please consult a healthcare professional.")
                
                with tab4:
                    st.subheader("💉 Medication Information")
                    if 'medications' in result:
                        st.markdown(result['medications'])
                    else:
                        st.info("Medication information not available. Please consult a healthcare professional.")
                    
                    st.error("""
                    **⚠️ IMPORTANT MEDICATION WARNING**
                    
                    - Never self-medicate
                    - All medications must be prescribed by a licensed healthcare professional
                    - This information is for educational purposes only
                    """)
                
                with tab5:
                    st.subheader("🔔 Recommended Next Steps")
                    if 'next_steps' in result:
                        st.markdown(result['next_steps'])
                    else:
                        st.info("Please consult a healthcare professional for guidance on next steps.")
                
                with tab6:
                    st.subheader("🛡️ Prevention & Risk Reduction")
                    if 'prevention' in result:
                        st.markdown(result['prevention'])
                    else:
                        st.info("Prevention information not available.")
                    
                    # Show complications if available
                    if 'complications' in result and result['complications']:
                        st.divider()
                        st.markdown("**Potential Complications:**")
                        st.warning(result['complications'])
            else:
                # Fallback tabs without RAG
                with tab3:
                    st.markdown("### AI Explanation")
                    st.info(result['explanation'])

            
            # Follow-up Questions Section (for medium confidence)
            if result.get('followup_questions') and len(result['followup_questions']) > 0:
                st.divider()
                st.header("🔍 Follow-up Questions (Optional)")
                
                confidence = result.get('confidence', 0)
                st.info(f"""
                **Current confidence: {confidence*100:.1f}%**
                
                To improve prediction accuracy, you may answer the following questions about additional symptoms:
                """)
                
                # Display questions with yes/no buttons
                for i, q in enumerate(result['followup_questions']):
                    symptom = q['symptom']
                    question = q['question']
                    
                    st.markdown(f"**{i+1}. {question}**")
                    
                    col1, col2, col3 = st.columns([1, 1, 4])
                    
                    with col1:
                        if st.button("✅ Yes", key=f"yes_{symptom}"):
                            st.session_state.followup_answers[symptom] = 1
                            st.session_state.current_symptoms[symptom] = 1
                    
                    with col2:
                        if st.button("❌ No", key=f"no_{symptom}"):
                            st.session_state.followup_answers[symptom] = 0
                            st.session_state.current_symptoms[symptom] = 0
                    
                    # Show current answer if exists
                    if symptom in st.session_state.followup_answers:
                        answer = "Yes" if st.session_state.followup_answers[symptom] == 1 else "No"
                        with col3:
                            st.markdown(f"*Answer: {answer}*")
                    
                    st.markdown("")
                
                # Update prediction button
                if len(st.session_state.followup_answers) > 0:
                    st.markdown("")
                    if st.button("🔄 Update Prediction with Answers", type="primary", use_container_width=True, key="update_med_conf"):
                        with st.spinner("Updating prediction..."):
                            try:
                                # Combine original symptoms with follow-up answers
                                all_symptoms = st.session_state.current_symptoms.copy()
                                
                                # Make new prediction
                                predictions = st.session_state.chatbot.predict_disease(all_symptoms, top_n=5)
                                
                                if predictions and len(predictions) > 0:
                                    # Get explanation
                                    symptom_list = [s for s, v in all_symptoms.items() if v == 1]
                                    explanation = st.session_state.chatbot.explain_results(predictions, symptom_list)
                                    
                                    # Update result
                                    updated_result = {
                                        'symptoms': symptom_list,
                                        'predictions': predictions,
                                        'explanation': explanation,
                                        'requires_followup': predictions[0][1] < st.session_state.chatbot.FOLLOWUP_THRESHOLD,
                                        'followup_questions': [],  # Clear follow-up questions
                                        'confidence': predictions[0][1]
                                    }
                                    
                                    st.session_state.current_result = updated_result
                                    st.session_state.followup_answers = {}  # Clear answers
                                    
                                    st.success(f"✅ Prediction updated! New confidence: {predictions[0][1]*100:.1f}%")
                                    st.rerun()
                                else:
                                    st.error("Unable to make prediction with these symptoms.")
                            
                            except Exception as e:
                                st.error(f"Error updating prediction: {str(e)}")
            
            # Medical disclaimer
            st.divider()
            st.warning("""
            ⚠️ **MEDICAL DISCLAIMER**
            
            This is an AI-based educational tool and NOT a substitute for professional medical advice, 
            diagnosis, or treatment. Always consult a qualified healthcare provider for any medical 
            concerns or before making health-related decisions.
            """)
        else:
            # No predictions available
            st.error("⚠️ **Unable to Make Prediction**")
            st.warning("""
            This symptom combination is not well-represented in the training data. 
            The model cannot make a reliable prediction for these symptoms.
            
            **Please consult a healthcare professional for proper diagnosis.**
            """)
            
            # Still show the explanation if available
            if result.get('explanation'):
                st.info(result['explanation'])
    
    
    # History section
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
    # Chatbot initialization failed
    st.error("Failed to initialize chatbot. Please check the configuration.")
    st.info("""
    **Setup Instructions:**
    1. Create a `.env` file with your Gemini API key
    2. Train the model by running `03_bayesian_network_binary.ipynb`
    3. Restart this app
    
    See `HYBRID_CHATBOT_SETUP.md` for detailed instructions.
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Powered by Bayesian Networks + Google Gemini AI | 
    <a href="https://github.com" target="_blank">GitHub</a> | 
    Built with Streamlit
</div>
""", unsafe_allow_html=True)
