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
        desc_short = disease_description[:2000] if len(disease_description) > 2000 else disease_description
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
        treatment_short = treatment_plan[:2000] if len(treatment_plan) > 2000 else treatment_plan
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
        meds_short = medications[:2000] if len(medications) > 2000 else medications
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
            desc = disease_info['description'][:2000]
            treatment = disease_info['treatment'][:2000]
            
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
            desc = disease_info['description'][:2000]
            meds = disease_info['medications'][:2000]
            
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

    def verify_and_correct_response(self, response_text: str, context_text: str) -> Dict:
        """
        Verifies a generated response against the retrieved context by:
        1. Breaking response into atomic claims
        2. Verifying each claim against context (entailment check)
        3. Regenerating if contradictions or lack of proof are found
        
        Args:
            response_text: The generated answer to verify
            context_text: The source context (RAG retrieval)
            
        Returns:
            Dict containing verified/corrected response and metadata
        """
        print(f"\n[NLI] Verifying response against context...")
        
        # 1. Extract atomic claims
        claims = self._extract_atomic_claims(response_text)
        if not claims:
            print("[NLI] No claims extracted.")
            return {"verified_response": response_text, "was_corrected": False, "failed_claims": []}
            
        # 2. Verify each claim
        verified_claims = []
        failed_claims = []
        
        for claim in claims:
            # We use the context as the premise and the claim as the hypothesis
            result = self._nli_check(premise=context_text, hypothesis=claim)
            
            # Strict check: Must be ENTAILMENT with high confidence
            if result['label'] == 'ENTAILMENT' and result['confidence'] > 0.7:
                verified_claims.append(claim)
            else:
                print(f"[NLI] Claim failed: '{claim}' ({result['label']})")
                failed_claims.append({
                    'claim': claim,
                    'reason': result['label'],
                    'reasoning': result['reasoning']
                })
        
        # 3. Regenerate if needed
        if failed_claims:
            print(f"[NLI] Found {len(failed_claims)} unsupported claims. Regenerating...")
            corrected_response = self._regenerate_response(response_text, context_text, failed_claims)
            return {
                "verified_response": corrected_response,
                "was_corrected": True,
                "failed_claims": failed_claims,
                "original_response": response_text
            }
        
        print("[NLI] Response fully verified.")
        return {
            "verified_response": response_text,
            "was_corrected": False,
            "failed_claims": []
        }

    def _extract_atomic_claims(self, text: str) -> List[str]:
        """Extract atomic claims from text using Gemini."""
        prompt = f"""You are a precise text analyzer.
Task: Break the following text into a list of atomic, independent factual claims.
Rules:
1. Each claim must be a single, standalone fact.
2. Ignore conversational filler (e.g., "Here is the info", "I hope this helps").
3. Preserve exact medical details (dosages, names, conditions).
4. Output valid JSON list of strings ONLY.

Text: "{text}"

JSON Output:"""
        
        try:
            response = self.model.generate_content(prompt)
            text_resp = response.text.strip()
            
            # Clean markdown
            if "```json" in text_resp:
                text_resp = text_resp.split("```json")[1].split("```")[0].strip()
            elif "```" in text_resp:
                text_resp = text_resp.split("```")[1].strip()
                
            return json.loads(text_resp)
        except Exception as e:
            print(f"[ERROR] Claim extraction failed: {e}")
            # Fallback: split by sentences
            return [s.strip() for s in text.split('.') if len(s.strip()) > 10]

    def _regenerate_response(self, original_response: str, context: str, failed_claims: List[Dict]) -> str:
        """Regenerate response excluding unsupported claims."""
        
        failures_text = "\n".join([f"- Claim: '{f['claim']}' (Issue: {f['reason']})" for f in failed_claims])
        
        prompt = f"""You are a medical AI assistant acting as a safety filter.

Task: Rewrite the Original Response to be strictly accurate based on the provided Context.

Context (True Source):
{context}

Original Response:
{original_response}

Verification Issues (False/Unproven Claims):
{failures_text}

Instructions:
1. Rewrite the response to convey the correct information from the Context.
2. REMOVE or CORRECT any claims flagged as False/Unproven.
3. Do NOT include information not present in the Context.
4. Maintain a helpful, professional tone.

Corrected Response:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[ERROR] Regeneration failed: {e}")
            return "Unable to verify and regenerate response. Please consult a healthcare professional."

    def verify_statement_with_citation(self, statement: str, context: str) -> dict:
        """
        Verify a statement against context and extract supporting quote.
        """
        prompt = f"""You are a medical fact-checker.
Task: Verify if the Statement is supported by the Context.
If supported (Entailment), extract the EXACT text segment from the Context that supports it.

Context: "{context}"
Statement: "{statement}"

Respond with valid JSON only:
{{
    "is_supported": true,
    "confidence": 0.95,
    "supporting_quote": "exact text substring from context",
    "reasoning": "brief explanation"
}}
If not supported, set is_supported to false and supporting_quote to null.
"""
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].strip()
            
            return json.loads(text)
        except Exception as e:
            print(f"[ERROR] Citation check failed: {e}")
            return {"is_supported": False, "confidence": 0.0, "supporting_quote": None}

    def verify_claims_against_context(self, claims: List[str], context: str) -> Dict:
        """
        Verify a list of claims against context and return evidence.
        """
        verified_items = []
        evidence = {}
        citation_idx = 1
        
        for claim in claims:
            res = self.verify_statement_with_citation(claim, context)
            
            # Check if supported and high confidence
            is_supported = res.get("is_supported", False) and res.get("confidence", 0) > 0.7
            quote = res.get("supporting_quote")
            
            # If quote is missing or empty, mark as not supported for citation purposes
            if is_supported and not quote:
                is_supported = False
            
            item = {
                "claim": claim,
                "is_supported": is_supported,
                "citation_id": None
            }
            
            if is_supported:
                item["citation_id"] = citation_idx
                evidence[citation_idx] = {
                    "quote": quote,
                    "reasoning": res.get("reasoning", "")
                }
                citation_idx += 1
            
            verified_items.append(item)
            
        return {"items": verified_items, "evidence": evidence}
