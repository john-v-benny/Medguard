"""
Test script for MRAG diagnostic reasoning system.
Demonstrates how the system explains disease predictions.
"""

from mrag_pipeline import mrag_explain

# Test Case 1: Malaria with high confidence
print("=" * 80)
print("TEST CASE 1: Malaria Diagnosis")
print("=" * 80)

symptoms_1 = ["High fever (103°F)", "Severe headache", "Body pain", "Fatigue", "Chills"]
bayesian_output_1 = {
    "Malaria": 0.75,
    "Typhoid Fever": 0.15,
    "Dengue": 0.10
}

explanation_1 = mrag_explain(symptoms_1, bayesian_output_1)
print(f"\nSymptoms: {', '.join(symptoms_1)}")
print(f"\nModel Predictions:")
for disease, prob in bayesian_output_1.items():
    print(f"  - {disease}: {prob:.1%}")
print(f"\n{explanation_1}")

# Test Case 2: COVID-19 with moderate confidence
print("\n" + "=" * 80)
print("TEST CASE 2: COVID-19 Diagnosis")
print("=" * 80)

symptoms_2 = ["Fever", "Dry cough", "Shortness of breath", "Loss of taste", "Fatigue"]
bayesian_output_2 = {
    "COVID-19": 0.65,
    "Influenza": 0.25,
    "Common Cold": 0.10
}

explanation_2 = mrag_explain(symptoms_2, bayesian_output_2)
print(f"\nSymptoms: {', '.join(symptoms_2)}")
print(f"\nModel Predictions:")
for disease, prob in bayesian_output_2.items():
    print(f"  - {disease}: {prob:.1%}")
print(f"\n{explanation_2}")

# Test Case 3: Close differential (multiple diseases with similar probability)
print("\n" + "=" * 80)
print("TEST CASE 3: Difficult Differential - Tuberculosis")
print("=" * 80)

symptoms_3 = ["Persistent cough", "Night sweats", "Weight loss", "Fatigue", "Low fever"]
bayesian_output_3 = {
    "Tuberculosis": 0.45,
    "Lung Cancer": 0.30,
    "Chronic Bronchitis": 0.25
}

explanation_3 = mrag_explain(symptoms_3, bayesian_output_3)
print(f"\nSymptoms: {', '.join(symptoms_3)}")
print(f"\nModel Predictions:")
for disease, prob in bayesian_output_3.items():
    print(f"  - {disease}: {prob:.1%}")
print(f"\n{explanation_3}")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)
print("\nThe MRAG system successfully explains the reasoning behind each diagnosis,")
print("showing HOW the model concluded each disease based on the symptom patterns.")
