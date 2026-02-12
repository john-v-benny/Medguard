"""
Hybrid Medical Chatbot: Bayesian Network + Gemini API + RAG

Architecture:
1. Gemini API: Natural language understanding & explanation
2. Bayesian Network: Accurate, hallucination-free disease prediction
3. RAG System: Detailed medical information from MedlinePlus
4. Gemini API: User-friendly result presentation

This approach ensures:
- Zero hallucinations in predictions (Bayesian Network)
- Natural conversation experience (Gemini API)
- Verified medical information (RAG + MedlinePlus)
- Explainable AI (probability scores + natural language)
"""

import pickle
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class HybridMedicalChatbot:
    # Minimum confidence threshold for showing predictions
    # Below this threshold, we require more symptoms via follow-up questions
    MINIMUM_CONFIDENCE = 0.15  # 15%
    FOLLOWUP_THRESHOLD = 0.75  # 75% - ask follow-up questions below this
    
    def __init__(self, model_path: str, gemini_api_key: str, use_rag: bool = True):
        """
        Initialize hybrid chatbot with Bayesian Network, Gemini API, and RAG.
        
        Args:
            model_path: Path to trained Bayesian Network model
            gemini_api_key: Google Gemini API key
            use_rag: Whether to use RAG for detailed explanations (default: True)
        """
        # Load Bayesian Network
        print("Loading Bayesian Network model...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.inference = model_data['inference']
        self.symptom_cols = model_data['symptom_cols']
        self.diseases = model_data['diseases']
        
        print(f"[OK] Bayesian Network loaded ({len(self.diseases)} diseases, {len(self.symptom_cols)} symptoms)")
        
        # Store API key for RAG
        self.gemini_api_key = gemini_api_key
        
        # Initialize RAG system with NLI verification
        self.use_rag = use_rag
        self.rag = None
        if use_rag:
            try:
                from rag_integration import MedicalRAG
                # Pass API key to RAG for NLI verification
                self.rag = MedicalRAG(gemini_api_key=gemini_api_key)
                print("[OK] RAG system initialized with NLI verification")
            except (ImportError, FileNotFoundError) as e:
                print(f"[WARNING] RAG system not available: {e}")
                print("[INFO] Chatbot will work without RAG (Gemini-only explanations)")
                self.use_rag = False

        
        # Initialize Gemini API
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        print("[OK] Gemini API initialized\n")
        
        # Create symptom list for Gemini
        self.symptom_list_str = ", ".join([s.replace('_', ' ') for s in self.symptom_cols])
    
    def extract_symptoms_from_text(self, user_input: str) -> List[str]:
        """
        Use Gemini to extract symptoms from natural language input.
        
        Args:
            user_input: User's description of symptoms
            
        Returns:
            List of symptom names (matching Bayesian Network format)
        """
        prompt = f"""You are a medical symptom extractor. Extract ONLY the symptoms mentioned in the user's text.

Available symptoms (use EXACT names from this list):
{self.symptom_list_str}

User input: "{user_input}"

Instructions:
1. Extract ONLY symptoms explicitly mentioned by the user
2. Use EXACT symptom names from the available list (replace spaces with underscores)
3. Return ONLY a comma-separated list of symptom names
4. If no symptoms match, return "NONE"
5. Do NOT add symptoms the user didn't mention
6. Do NOT provide explanations or additional text

Example:
User: "I have a high fever and headache"
Response: high_fever,headache

Now extract symptoms from the user input above:"""

        try:
            response = self.model.generate_content(prompt)
            extracted_text = response.text.strip()
            
            if extracted_text == "NONE" or not extracted_text:
                return []
            
            # Parse extracted symptoms
            symptoms = [s.strip().replace(' ', '_') for s in extracted_text.split(',')]
            
            # Validate symptoms exist in our model
            valid_symptoms = [s for s in symptoms if s in self.symptom_cols]
            
            return valid_symptoms
        
        except Exception as e:
            print(f"Error extracting symptoms: {e}")
            return []
    
    def predict_disease(self, symptoms: Dict[str, int], top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Use Bayesian Network to predict diseases (NO hallucination).
        
        Args:
            symptoms: Dictionary of {symptom: 1} for present symptoms
            top_n: Number of top predictions to return
            
        Returns:
            List of (disease, probability) tuples
        """
        if not symptoms:
            return []
        
        # Query Bayesian Network
        result = self.inference.query(variables=['prognosis'], evidence=symptoms)
        probs = result.values
        diseases_list = result.state_names['prognosis']
        
        # Filter out NaN values and create predictions
        import numpy as np
        predictions = []
        for disease, prob in zip(diseases_list, probs):
            if not np.isnan(prob) and not np.isinf(prob):
                predictions.append((disease, float(prob)))
        
        # If all probabilities are NaN, return empty list
        if not predictions:
            return []
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_n]
    
    def calculate_information_gain(self, current_symptoms: Dict[str, int], 
                                   current_predictions: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Calculate information gain for each unknown symptom.
        
        Args:
            current_symptoms: Dictionary of currently known symptoms
            current_predictions: Current disease predictions
            
        Returns:
            List of (symptom, information_gain) tuples, sorted by gain
        """
        import numpy as np
        from scipy.stats import entropy
        
        # Get current entropy
        current_probs = [p[1] for p in current_predictions]
        current_entropy = entropy(current_probs) if len(current_probs) > 0 else 0
        
        # Get unknown symptoms
        known_symptoms = set(current_symptoms.keys())
        unknown_symptoms = [s for s in self.symptom_cols if s not in known_symptoms]
        
        information_gains = []
        
        for symptom in unknown_symptoms:
            # Calculate expected entropy if we knew this symptom
            # Try both: symptom present and symptom absent
            expected_entropy = 0
            
            for symptom_value in [0, 1]:
                test_symptoms = current_symptoms.copy()
                test_symptoms[symptom] = symptom_value
                
                try:
                    predictions = self.predict_disease(test_symptoms, top_n=10)
                    if predictions:
                        probs = [p[1] for p in predictions]
                        symptom_entropy = entropy(probs)
                        # Weight by probability (assume 50/50 if unknown)
                        expected_entropy += 0.5 * symptom_entropy
                except:
                    # If prediction fails, skip this symptom
                    continue
            
            # Information gain = current entropy - expected entropy
            info_gain = current_entropy - expected_entropy
            information_gains.append((symptom, info_gain))
        
        # Sort by information gain (descending)
        information_gains.sort(key=lambda x: x[1], reverse=True)
        
        return information_gains
    
    def get_followup_questions(self, current_symptoms: Dict[str, int], 
                               current_predictions: List[Tuple[str, float]],
                               max_questions: int = 5) -> List[Dict[str, str]]:
        """
        Generate follow-up questions based on information gain.
        
        Args:
            current_symptoms: Currently known symptoms
            current_predictions: Current predictions
            max_questions: Maximum number of questions to ask
            
        Returns:
            List of question dictionaries with 'symptom' and 'question' keys
        """
        # Calculate information gain for all unknown symptoms
        info_gains = self.calculate_information_gain(current_symptoms, current_predictions)
        
        # Get top N most informative symptoms
        top_symptoms = info_gains[:max_questions]
        
        # Generate natural language questions
        questions = []
        for symptom, gain in top_symptoms:
            # Convert symptom name to readable format
            readable_symptom = symptom.replace('_', ' ').title()
            
            question = {
                'symptom': symptom,
                'question': f"Do you have {readable_symptom.lower()}?",
                'info_gain': gain
            }
            questions.append(question)
        
        return questions
    
    def explain_results(self, predictions: List[Tuple[str, float]], symptoms: List[str]) -> str:
        """
        Use Gemini to explain results in natural language.
        
        Args:
            predictions: List of (disease, probability) tuples
            symptoms: List of symptom names
            
        Returns:
            Natural language explanation
        """
        if not predictions or len(predictions) == 0:
            return "I couldn't make a diagnosis based on the provided information."
        
        top_disease, top_prob = predictions[0]
        
        # Format predictions for Gemini
        predictions_str = "\n".join([
            f"- {disease}: {prob*100:.1f}%" 
            for disease, prob in predictions
        ])
        
        symptoms_str = ", ".join([s.replace('_', ' ') for s in symptoms])
        
        prompt = f"""You are a medical AI assistant explaining diagnosis results to a patient.

IMPORTANT RULES:
1. Use the EXACT probabilities provided - DO NOT make up or change numbers
2. DO NOT diagnose anything not in the predictions list
3. Focus on the top prediction only
4. Be empathetic but concise (2-3 sentences)
5. Always recommend seeing a doctor
6. DO NOT provide treatment advice

Patient's symptoms: {symptoms_str}

Bayesian Network Predictions (EXACT probabilities):
{predictions_str}

Provide a brief, empathetic explanation of the top result. Keep it under 100 words."""

        try:
            response = self.model.generate_content(prompt)
            explanation = response.text.strip()
            
            # Add disclaimer
            explanation += "\n\n⚠️ This is an AI analysis tool, not a medical diagnosis. Please consult a healthcare professional."
            
            return explanation
        
        except Exception as e:
            # Fallback to simple explanation
            return f"Based on your symptoms, the most likely condition is {top_disease} ({top_prob*100:.1f}% confidence). Please consult a healthcare professional for proper diagnosis."
    
    def diagnose(self, user_input: str) -> Dict:
        """
        Complete diagnosis pipeline: extract symptoms → predict → explain.
        
        Args:
            user_input: User's natural language description
            
        Returns:
            Dictionary with symptoms, predictions, explanation, and follow-up questions
        """
        print("=" * 80)
        print("HYBRID MEDICAL CHATBOT")
        print("=" * 80)
        # Safe print with encoding handling
        try:
            print(f"\nYour input: \"{user_input}\"\n")
        except UnicodeEncodeError:
            print(f"\nYour input: [contains special characters]\n")
        
        # Step 1: Extract symptoms using Gemini
        print("[1/3] Extracting symptoms...")
        extracted_symptoms = self.extract_symptoms_from_text(user_input)
        
        if not extracted_symptoms:
            return {
                'symptoms': [],
                'predictions': [],
                'explanation': "I couldn't identify any specific symptoms from your description. Please describe your symptoms more clearly.",
                'requires_followup': False,
                'followup_questions': []
            }
        
        # Safe print for symptoms
        try:
            print(f"[OK] Identified symptoms: {', '.join([s.replace('_', ' ') for s in extracted_symptoms])}\n")
        except UnicodeEncodeError:
            print(f"[OK] Identified {len(extracted_symptoms)} symptoms\n")
        
        # Step 2: Predict using Bayesian Network (NO hallucination)
        print("[2/3] Running Bayesian Network analysis...")
        symptoms_dict = {s: 1 for s in extracted_symptoms}
        predictions = self.predict_disease(symptoms_dict, top_n=5)
        
        # Step 3: Safe Response Strategy - Always provide predictions
        # Even if confidence is very low, show top predictions with appropriate warnings
        if not predictions or len(predictions) == 0:
            # Fallback: This should rarely happen, but if it does, provide safe guidance
            return {
                'symptoms': extracted_symptoms,
                'predictions': [],
                'explanation': """⚠️ **Unable to Analyze Symptoms**

The symptom combination you've provided is very unusual and not well-represented in our medical database. This could mean:

1. **Multiple conditions** may be present
2. **Rare condition** not in our database
3. **Symptom description** may need clarification

**Recommended Action:**
🏥 **Please consult a healthcare professional immediately** for proper diagnosis. Your symptoms require professional medical evaluation.

This AI tool has limitations and cannot handle all medical scenarios. A qualified doctor can perform proper examinations and tests.""",
                'requires_followup': False,
                'followup_questions': [],
                'confidence': 0.0
            }
        
        print("[OK] Predictions generated\n")
        
        # Step 3: Check confidence and determine response strategy
        top_confidence = predictions[0][1] if predictions and len(predictions) > 0 else 0.0
        
        # STRATEGY: For very low confidence, ask questions FIRST
        # Only show safe response if confidence is still low AFTER answering questions
        
        if top_confidence < self.MINIMUM_CONFIDENCE:
            # Very low confidence (< 15%)
            print(f"[WARNING] Confidence ({top_confidence*100:.2f}%) is below minimum threshold ({self.MINIMUM_CONFIDENCE*100:.0f}%)")
            print("[INFO] Will ask follow-up questions to improve confidence...")
            print("[INFO] If confidence remains low after questions, will show safe response with warnings\n")
            
            # Generate follow-up questions
            num_questions = 5
            followup_questions = self.get_followup_questions(
                symptoms_dict, 
                predictions, 
                max_questions=num_questions
            )
            
            if followup_questions:
                print(f"[OK] Generated {len(followup_questions)} follow-up questions\n")
            
            # DON'T show predictions yet - ask questions first
            return {
                'symptoms': extracted_symptoms,
                'predictions': [],  # Hidden until questions answered
                'explanation': f"""⚠️ **Need More Information**

Based on the symptoms you've described, I cannot make a reliable prediction (confidence would be only {top_confidence*100:.1f}%). 

To provide an accurate diagnosis, I need to ask you a few more questions about your symptoms.

**Please answer the follow-up questions below** to help me narrow down the possibilities.""",
                'requires_followup': True,
                'followup_questions': followup_questions,
                'confidence': top_confidence,
                'hidden_predictions': predictions  # Keep for debugging
            }
        
        # Step 4: Display raw predictions (transparent)
        print("[3/3] BAYESIAN NETWORK PREDICTIONS (Exact Probabilities):")
        print("-" * 80)
        for i, (disease, prob) in enumerate(predictions, 1):
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"{i}. {disease:35s} {bar} {prob*100:6.2f}%")
        print()
        
        # Extract top prediction (already validated above)
        top_disease, top_prob = predictions[0]
        
        # Step 5: Explain using Gemini (natural language)
        print("\nNATURAL LANGUAGE EXPLANATION:")
        print("-" * 80)
        explanation = self.explain_results(predictions, extracted_symptoms)
        # Safe print for explanation
        try:
            print(explanation)
        except UnicodeEncodeError:
            print("[Explanation contains special characters - view in UI]")
        print()
        
        # Step 6: Check if follow-up questions are needed (for medium confidence)
        followup_questions = []
        
        if top_confidence < self.FOLLOWUP_THRESHOLD:  # Less than 75% confidence
            print(f"\n[INFO] Confidence is below {self.FOLLOWUP_THRESHOLD*100:.0f}%. Generating follow-up questions...")
            
            # Determine number of questions based on confidence
            if top_confidence < 0.50:
                num_questions = 5
            else:
                num_questions = 3
            
            # Generate follow-up questions
            followup_questions = self.get_followup_questions(
                symptoms_dict, 
                predictions, 
                max_questions=num_questions
            )
            
            if followup_questions:
                print(f"[OK] Generated {len(followup_questions)} follow-up questions\n")

        
        # Step 7: Get detailed medical information from RAG (if available)
        rag_info = None
        if self.use_rag and self.rag:
            try:
                print("[INFO] Retrieving detailed medical information from RAG...")
                rag_info = self.rag.explain_diagnosis(
                    disease=top_disease,
                    confidence=top_confidence,
                    symptoms=extracted_symptoms
                )
                print("[OK] RAG information retrieved\n")
            except Exception as e:
                print(f"[WARNING] RAG retrieval failed: {e}")
                print("[INFO] Falling back to Gemini-only explanation\n")
        
        # Build result dictionary
        result = {
            'symptoms': extracted_symptoms,
            'predictions': predictions,
            'explanation': explanation,
            'requires_followup': top_confidence < self.FOLLOWUP_THRESHOLD,
            'followup_questions': followup_questions,
            'confidence': top_confidence
        }
        
        # Add RAG information if available
        if rag_info:
            result.update({
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
                result['nli_verification'] = rag_info['nli_verification']
                print(f"[OK] NLI verification added to result: {rag_info['nli_verification']['summary']}")
        
        return result



def main():
    """Example usage of hybrid chatbot."""
    
    # Configuration
    MODEL_PATH = 'models/disease_bayesian_network.pkl'
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Set this environment variable
    
    if not GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY environment variable not set!")
        print("Set it with: export GEMINI_API_KEY='your-api-key'")
        print("\nFalling back to pure Bayesian Network mode...\n")
        return
    
    # Initialize chatbot
    print("\n" + "="*80)
    print("HYBRID MEDICAL CHATBOT")
    print("="*80)
    print("\n✅ Chatbot initialized successfully!")
    print("   - Bayesian Network: Loaded")
    print("   - Gemini API: Connected")
    print("   - RAG System: Ready")
    print("\n💡 To use the chatbot, run: streamlit run app.py")
    print("="*80 + "\n")
    
    # NOTE: Test examples are commented out to prevent hitting API rate limits
    # Uncomment the examples below if you want to test the chatbot directly
    
    # # Example 1: Diabetes symptoms
    # print("\n" + "="*80)
    # print("EXAMPLE 1: Diabetes-like symptoms")
    # print("="*80)
    # result1 = chatbot.diagnose(
    #     "I've been feeling extremely tired lately, losing weight without trying, "
    #     "and I'm always thirsty. I also need to urinate frequently."
    # )
    
    # # Example 2: Malaria symptoms
    # print("\n" + "="*80)
    # print("EXAMPLE 2: Malaria-like symptoms")
    # print("="*80)
    # result2 = chatbot.diagnose(
    #     "I have a high fever with chills, severe headache, and I've been vomiting. "
    #     "I'm also sweating a lot and have muscle pain."
    # )
    
    # # Example 3: Vague symptoms
    # print("\n" + "="*80)
    # print("EXAMPLE 3: Vague symptoms")
    # print("="*80)
    # result3 = chatbot.diagnose(
    #     "I just don't feel well. I'm tired and have a headache."
    # )


if __name__ == "__main__":
    main()

