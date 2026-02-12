"""
Natural Language Inference Verifier for Medical RAG
Verifies accuracy and consistency of RAG-generated medical information
"""

import json
import google.genai as genai
from typing import List, Dict

class MedicalNLIVerifier:
    """
    Verifies medical information using Natural Language Inference (NLI)
    
    Uses Gemini API to determine logical relationships between:
    - Patient symptoms and disease descriptions
    - Diseases and treatment plans
    - Conditions and medication recommendations
    
    Returns verification results with confidence scores.
    """
    
    def __init__(self, gemini_api_key: str):
        """
        Initialize NLI verifier with Gemini API
        
        Args:
            gemini_api_key: Google Gemini API key
        """
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # NLI prompt template
        self.nli_template = """You are a medical Natural Language Inference expert.

Task: Determine if the hypothesis logically follows from the premise.

Premise: "{premise}"
Hypothesis: "{hypothesis}"

Determine the relationship:
- ENTAILMENT: Hypothesis logically follows from premise (symptoms match, treatment is appropriate, etc.)
- CONTRADICTION: Hypothesis contradicts premise (conflicting information)
- NEUTRAL: No clear logical relationship

Respond with valid JSON only (no markdown, no code blocks):
{{
    "label": "ENTAILMENT",
    "confidence": 0.95,
    "reasoning": "Brief explanation why"
}}
"""
    
    def _nli_check(self, premise: str, hypothesis: str) -> dict:
        """
        Perform NLI check using Gemini API
        
        Args:
            premise: The base statement (e.g., patient symptoms)
            hypothesis: The statement to verify (e.g., disease description)
        
        Returns:
            dict with label, confidence, and reasoning
        """
        try:
            prompt = self.nli_template.format(
                premise=premise,
                hypothesis=hypothesis
            )
            
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean up response - remove markdown code blocks if present
            if "```json" in text:
                parts = text.split("```json")
                if len(parts) > 1:
                    inner_parts = parts[1].split("```")
                    if len(inner_parts) > 0:
                        text = inner_parts[0].strip()
            elif "```" in text:
                parts = text.split("```")
                if len(parts) > 2:
                    text = parts[1].strip()
            
            # Parse JSON
            result = json.loads(text)
            
            # Validate result
            if 'label' not in result:
                result['label'] = 'NEUTRAL'
            if 'confidence' not in result:
                result['confidence'] = 0.5
            if 'reasoning' not in result:
                result['reasoning'] = 'No reasoning provided'
            
            # Ensure label is uppercase
            result['label'] = result['label'].upper()
            
            print(f"[NLI] {result['label']} ({result['confidence']:.2f}): {result['reasoning'][:50]}...")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse NLI response: {e}")
            print(f"[DEBUG] Response text: {text[:200]}")
            return {
                "label": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": f"JSON parsing error: {str(e)}"
            }
        except Exception as e:
            print(f"[ERROR] NLI check failed: {e}")
            return {
                "label": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }
    
    def verify_symptom_disease_match(
        self, 
        symptoms: List[str], 
        disease: str,
        disease_description: str
    ) -> dict:
        """
        Verify if patient symptoms match the predicted disease
        
        Args:
            symptoms: List of patient symptoms
            disease: Predicted disease name
            disease_description: Disease description from RAG
        
        Returns:
            Verification result with label, confidence, reasoning, and verified flag
        """
        print(f"\n[NLI] Checking symptom-disease match for {disease}...")
        
        # Create premise from symptoms
        symptom_text = ", ".join(symptoms)
        premise = f"Patient presents with the following symptoms: {symptom_text}"
        
        # Create hypothesis from disease description
        # Extract first 200 chars of description for focused check
        desc_short = disease_description[:200] if len(disease_description) > 200 else disease_description
        hypothesis = f"{disease} is characterized by: {desc_short}"
        
        result = self._nli_check(premise, hypothesis)
        
        return {
            "check_type": "symptom_disease_match",
            "label": result["label"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
            "verified": result["label"] == "ENTAILMENT" and result["confidence"] > 0.7
        }
    
    def verify_treatment_appropriateness(
        self,
        disease: str,
        treatment_plan: str
    ) -> dict:
        """
        Verify if treatment plan is appropriate for the disease
        
        Args:
            disease: Disease name
            treatment_plan: Recommended treatment from RAG
        
        Returns:
            Verification result
        """
        print(f"[NLI] Checking treatment appropriateness for {disease}...")
        
        premise = f"Patient has been diagnosed with {disease}"
        
        # Extract first 200 chars of treatment plan
        treatment_short = treatment_plan[:200] if len(treatment_plan) > 200 else treatment_plan
        hypothesis = f"Appropriate treatment includes: {treatment_short}"
        
        result = self._nli_check(premise, hypothesis)
        
        return {
            "check_type": "treatment_appropriateness",
            "label": result["label"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
            "verified": result["label"] == "ENTAILMENT" and result["confidence"] > 0.7
        }
    
    def verify_medication_safety(
        self,
        disease: str,
        symptoms: List[str],
        medications: str
    ) -> dict:
        """
        Verify if medications are safe and appropriate
        
        Args:
            disease: Disease name
            symptoms: Patient symptoms
            medications: Recommended medications from RAG
        
        Returns:
            Verification result
        """
        print(f"[NLI] Checking medication safety for {disease}...")
        
        symptom_text = ", ".join(symptoms[:5])  # Limit to first 5 symptoms
        premise = f"Patient has {disease} with symptoms including: {symptom_text}"
        
        # Extract first 200 chars of medications
        meds_short = medications[:200] if len(medications) > 200 else medications
        hypothesis = f"Safe and appropriate medications include: {meds_short}"
        
        result = self._nli_check(premise, hypothesis)
        
        return {
            "check_type": "medication_safety",
            "label": result["label"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
            "verified": result["label"] == "ENTAILMENT" and result["confidence"] > 0.7
        }
    
    def detect_contradictions(
        self,
        disease_info: dict
    ) -> List[dict]:
        """
        Detect contradictions in RAG-retrieved information
        
        Args:
            disease_info: Complete disease information from RAG
        
        Returns:
            List of detected contradictions
        """
        print("[NLI] Checking for contradictions...")
        
        contradictions = []
        
        # Check if treatment contradicts disease description
        if 'description' in disease_info and 'treatment' in disease_info:
            desc = disease_info['description'][:200]
            treatment = disease_info['treatment'][:200]
            
            result = self._nli_check(
                premise=f"Disease description: {desc}",
                hypothesis=f"Treatment approach: {treatment}"
            )
            
            if result['label'] == 'CONTRADICTION':
                contradictions.append({
                    "type": "treatment_disease_mismatch",
                    "severity": "high",
                    "details": result['reasoning'],
                    "confidence": result['confidence']
                })
        
        # Check if medications contradict disease type
        if 'description' in disease_info and 'medications' in disease_info:
            desc = disease_info['description'][:200]
            meds = disease_info['medications'][:200]
            
            result = self._nli_check(
                premise=f"Disease description: {desc}",
                hypothesis=f"Medications: {meds}"
            )
            
            if result['label'] == 'CONTRADICTION':
                contradictions.append({
                    "type": "medication_disease_mismatch",
                    "severity": "high",
                    "details": result['reasoning'],
                    "confidence": result['confidence']
                })
        
        if contradictions:
            print(f"[WARNING] Found {len(contradictions)} contradiction(s)")
        else:
            print("[OK] No contradictions detected")
        
        return contradictions
    
    def verify_all(
        self,
        symptoms: List[str],
        disease: str,
        disease_info: dict
    ) -> dict:
        """
        Run all verification checks
        
        Args:
            symptoms: Patient symptoms
            disease: Predicted disease
            disease_info: RAG-retrieved information
        
        Returns:
            Complete verification report
        """
        print(f"\n{'='*60}")
        print(f"[NLI] Running comprehensive verification for {disease}")
        print(f"{'='*60}")
        
        # Run all checks
        symptom_check = self.verify_symptom_disease_match(
            symptoms, 
            disease, 
            disease_info.get('description', '')
        )
        
        treatment_check = self.verify_treatment_appropriateness(
            disease,
            disease_info.get('treatment', '')
        )
        
        medication_check = self.verify_medication_safety(
            disease,
            symptoms,
            disease_info.get('medications', '')
        )
        
        contradictions = self.detect_contradictions(disease_info)
        
        # Calculate overall verification score
        checks = [symptom_check, treatment_check, medication_check]
        verified_count = sum(1 for c in checks if c['verified'])
        overall_score = verified_count / len(checks)
        
        # Generate summary
        summary = self._generate_summary(
            symptom_check, treatment_check, medication_check, contradictions
        )
        
        print(f"\n[NLI] Verification Score: {overall_score:.1%}")
        print(f"[NLI] Summary: {summary}")
        print(f"{'='*60}\n")
        
        return {
            "symptom_match": symptom_check,
            "treatment_check": treatment_check,
            "medication_check": medication_check,
            "contradictions": contradictions,
            "overall_verified": overall_score >= 0.67,  # At least 2/3 checks pass
            "verification_score": overall_score,
            "summary": summary
        }
    
    def _generate_summary(
        self, 
        symptom_check: dict, 
        treatment_check: dict, 
        medication_check: dict, 
        contradictions: List[dict]
    ) -> str:
        """
        Generate human-readable summary of verification results
        
        Args:
            symptom_check: Symptom verification result
            treatment_check: Treatment verification result
            medication_check: Medication verification result
            contradictions: List of contradictions
        
        Returns:
            Summary string
        """
        if contradictions:
            return "⚠️ Contradictions detected in medical information"
        
        verified_count = sum(1 for c in [symptom_check, treatment_check, medication_check] if c['verified'])
        
        if verified_count == 3:
            return "✅ All information verified and consistent"
        elif verified_count == 2:
            return "⚠️ Most information verified, some uncertainty"
        elif verified_count == 1:
            return "⚠️ Limited verification, consult healthcare provider"
        else:
            return "❌ Information needs manual review by medical professional"
