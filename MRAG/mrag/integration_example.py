"""
Example integration: Bayesian Model + MRAG Reasoning System

This demonstrates how to combine the Bayesian disease predictor 
with the MRAG explainable AI system.
"""

import requests
import json

def get_diagnostic_explanation(symptoms: list[str], model_predictions: dict[str, float]):
    """
    Get explainable AI diagnostic reasoning for a disease prediction.
    
    Args:
        symptoms: List of symptom descriptions
        model_predictions: Dictionary of {disease: probability} from Bayesian model
    
    Returns:
        Dictionary with explanation and metadata
    """
    
    # Call MRAG backend API
    response = requests.post(
        "http://localhost:8000/mrag",
        json={
            "symptoms": symptoms,
            "bayesian_output": model_predictions
        }
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": f"API request failed with status {response.status_code}",
            "details": response.text
        }


# Example 1: After Bayesian model makes a prediction
print("=" * 80)
print("INTEGRATION EXAMPLE: Bayesian Model → MRAG Explanation")
print("=" * 80)

# Simulated output from Bayesian disease predictor
patient_symptoms = [
    "Fever (101.5°F)",
    "Headache (7/10 severity)", 
    "Cough (5/10 severity)",
    "Fatigue (8/10 severity)",
    "Body pain (6/10 severity)"
]

# Simulated Bayesian model predictions
bayesian_predictions = {
    "Influenza": 0.68,
    "COVID-19": 0.22,
    "Common Cold": 0.10
}

print("\n📊 STEP 1: Bayesian Model Prediction")
print("-" * 40)
print(f"Patient Symptoms: {', '.join(patient_symptoms)}")
print("\nDisease Probabilities:")
for disease, prob in sorted(bayesian_predictions.items(), key=lambda x: x[1], reverse=True):
    print(f"  {'█' * int(prob * 50)} {disease}: {prob:.1%}")

print("\n🧠 STEP 2: MRAG Diagnostic Reasoning")
print("-" * 40)

# Get explanation from MRAG system
result = get_diagnostic_explanation(patient_symptoms, bayesian_predictions)

if "error" not in result:
    print(f"\n✅ Diagnosis: {result['top_disease']}")
    print(f"📈 Confidence: {result['confidence']:.1%}")
    print(f"\n📝 Explanation:\n")
    print(result['explanation'])
    
    print("\n" + "=" * 80)
    print("COMPLETE DIAGNOSTIC REPORT")
    print("=" * 80)
    print(json.dumps(result, indent=2))
else:
    print(f"\n❌ Error: {result['error']}")
    print("\nMake sure the MRAG backend is running:")
    print("  cd MRAG/mrag")
    print("  uvicorn backend.app:app --reload --port 8000")


# Example 2: Streamlit Integration Pattern
print("\n" + "=" * 80)
print("STREAMLIT INTEGRATION PATTERN")
print("=" * 80)

streamlit_code = '''
import streamlit as st
import requests

# After collecting symptoms and running Bayesian model
if st.button("Get Diagnosis Explanation"):
    with st.spinner("Analyzing diagnostic reasoning..."):
        response = requests.post(
            "http://localhost:8000/mrag",
            json={
                "symptoms": collected_symptoms,
                "bayesian_output": model_predictions
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            st.success(f"Diagnosis: {result['top_disease']}")
            st.metric("Confidence", f"{result['confidence']:.1%}")
            
            # Show all predictions
            st.subheader("Differential Diagnoses")
            for disease, prob in result['all_predictions'].items():
                st.progress(prob)
                st.write(f"{disease}: {prob:.1%}")
            
            # Display reasoning
            st.subheader("Diagnostic Reasoning")
            st.markdown(result['explanation'])
'''

print(streamlit_code)

print("\n" + "=" * 80)
print("KEY BENEFITS")
print("=" * 80)
print("""
✓ Explainable AI: Understand WHY the model made its prediction
✓ Transparency: Clear reasoning for each diagnosis
✓ Clinical Insights: Symptom-disease correlations explained
✓ Differential Analysis: Why alternatives were ruled out
✓ Confidence Metrics: Understand certainty levels
✓ Educational Value: Learn about disease presentations
""")
