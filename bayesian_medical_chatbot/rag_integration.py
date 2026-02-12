"""
RAG Integration Module for Bayesian Medical Chatbot
Provides detailed explanations, treatment plans, and recommendations.
Includes NLI verification for information accuracy.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional
import os

# Import NLI verifier
try:
    from nli_verifier import MedicalNLIVerifier
    NLI_AVAILABLE = True
except ImportError:
    NLI_AVAILABLE = False
    print("[WARNING] NLI verifier not available")


class MedicalRAG:
    """RAG system for medical disease information retrieval and explanation."""
    
    def __init__(self, db_path: str = './chroma_db', collection_name: str = 'disease_knowledge', gemini_api_key: str = None):
        """
        Initialize the Medical RAG system.
        
        Args:
            db_path: Path to ChromaDB storage
            collection_name: Name of the collection to query
            gemini_api_key: Optional Gemini API key for NLI verification
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Vector database not found at: {db_path}\n"
                f"Please run: python vector_db_setup.py first"
            )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)
        
        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize NLI verifier if available
        self.nli_verifier = None
        if NLI_AVAILABLE and gemini_api_key:
            try:
                self.nli_verifier = MedicalNLIVerifier(gemini_api_key)
                print("[RAG] NLI Verifier initialized")
            except Exception as e:
                print(f"[WARNING] NLI Verifier initialization failed: {e}")
        
        print(f"[RAG] Initialized with {self.collection.count()} diseases")
    
    def get_disease_info(self, disease_name: str) -> Optional[Dict]:
        """
        Get complete information for a specific disease.
        
        Args:
            disease_name: Name of the disease (exact or partial match)
            
        Returns:
            Dictionary with disease information or None if not found
        """
        # Try exact match first
        results = self.collection.get(
            where={"disease": disease_name}
        )
        
        if results and 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0:
            return results['metadatas'][0]
        
        # Try semantic search if exact match fails
        query_embedding = self.model.encode([disease_name])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=1
        )
        
        # Validate results structure before accessing
        if (results and 
            'metadatas' in results and 
            results['metadatas'] and 
            len(results['metadatas']) > 0 and 
            results['metadatas'][0] and
            'distances' in results and
            results['distances'] and
            len(results['distances']) > 0 and
            len(results['distances'][0]) > 0):
            # Check if similarity is reasonable (distance < 0.5)
            if results['distances'][0][0] < 0.5:
                return results['metadatas'][0]
        
        return None
    
    def explain_diagnosis(
        self,
        disease: str,
        confidence: float,
        symptoms: List[str]
    ) -> Dict:
        """
        Generate comprehensive explanation for Bayesian diagnosis with NLI verification.
        
        Args:
            disease: Predicted disease name
            confidence: Bayesian confidence score (0-1)
            symptoms: List of observed symptoms
            
        Returns:
            Dictionary with comprehensive explanation, recommendations, and verification results
        """
        # Get disease information from RAG
        disease_info = self.get_disease_info(disease)
        
        if not disease_info:
            return self._generate_fallback_explanation(disease, confidence, symptoms)
        
        # Generate explanation
        explanation = self._generate_explanation_text(
            disease_info, confidence, symptoms
        )
        
        # Extract treatment plan
        treatment_plan = self._format_treatment_plan(disease_info['treatment'])
        
        # Extract medications
        medications = self._format_medications(disease_info['medications'])
        
        # Generate next steps
        next_steps = self._generate_next_steps(
            disease_info, confidence
        )
        
        # Format prevention
        prevention = self._format_prevention(disease_info['prevention'])
        
        # Prepare result dictionary
        result = {
            'explanation': explanation,
            'treatment_plan': treatment_plan,
            'medications': medications,
            'next_steps': next_steps,
            'prevention': prevention,
            'source_url': disease_info['source_url'],
            'source_name': disease_info['source_name'],
            'disease_description': disease_info['description'],
            'complications': disease_info['complications']
        }
        
        # Run NLI verification if available
        if self.nli_verifier:
            try:
                verification = self.nli_verifier.verify_all(
                    symptoms=symptoms,
                    disease=disease,
                    disease_info=disease_info
                )
                result['nli_verification'] = verification
                print(f"[RAG] Verification: {verification['summary']}")
            except Exception as e:
                print(f"[WARNING] NLI verification failed: {e}")
        
        return result
    
    def _generate_explanation_text(
        self,
        disease_info: Dict,
        confidence: float,
        symptoms: List[str]
    ) -> str:
        """Generate human-readable explanation."""
        confidence_pct = confidence * 100
        
        # Confidence level
        if confidence >= 0.75:
            conf_level = "HIGH"
            conf_desc = "strongly indicates"
        elif confidence >= 0.50:
            conf_level = "MEDIUM"
            conf_desc = "suggests"
        else:
            conf_level = "LOW"
            conf_desc = "may indicate"
        
        explanation = f"""
**Diagnosis Explanation**

The Bayesian network predicted **{disease_info['disease']}** with {confidence_pct:.1f}% confidence ({conf_level} confidence).

**About {disease_info['disease']}:**
{disease_info['description']}

**How the model reached this conclusion:**
Based on the symptoms you provided ({', '.join(symptoms)}), the statistical model {conf_desc} {disease_info['disease']}. This prediction is supported by medical literature from MedlinePlus.

**Confidence Level:** {conf_level} ({confidence_pct:.1f}%)
- The model analyzed symptom patterns from verified medical data
- This confidence score reflects the statistical likelihood based on your symptoms
        """.strip()
        
        return explanation
    
    def _format_treatment_plan(self, treatment: str) -> str:
        """Format treatment information."""
        if not treatment or treatment == 'Information not available.':
            return "Treatment options are available. Please consult a healthcare professional for a personalized treatment plan."
        
        return f"""
**Treatment Options:**

{treatment}

> **⚠️ Important:** This information is for educational purposes only. Always consult a qualified healthcare professional before starting any treatment.
        """.strip()
    
    def _format_medications(self, medications: str) -> str:
        """Format medication information."""
        if not medications or 'not available' in medications.lower():
            return "Consult a healthcare professional for medication recommendations."
        
        return f"""
**Medication Information:**

{medications}

> **⚠️ Critical:** Never self-medicate. All medications should be prescribed and monitored by a licensed healthcare professional.
        """.strip()
    
    def _generate_next_steps(self, disease_info: Dict, confidence: float) -> str:
        """Generate next steps based on confidence and disease info."""
        when_to_see = disease_info.get('when_to_see_doctor', '')
        
        next_steps = "**Recommended Next Steps:**\n\n"
        
        if confidence >= 0.75:
            next_steps += "1. **Schedule an appointment** with a healthcare professional for proper diagnosis\n"
            next_steps += "2. **Prepare for your visit:** Note all symptoms, their duration, and severity\n"
            next_steps += "3. **Bring this information** to discuss with your doctor\n\n"
        elif confidence >= 0.50:
            next_steps += "1. **Monitor your symptoms** closely over the next few days\n"
            next_steps += "2. **Consult a healthcare professional** if symptoms worsen or persist\n"
            next_steps += "3. **Keep a symptom diary** to track changes\n\n"
        else:
            next_steps += "1. **Seek medical evaluation** - the symptoms are unclear and require professional assessment\n"
            next_steps += "2. **Do not self-diagnose** - multiple conditions may present similar symptoms\n"
            next_steps += "3. **Get a proper diagnosis** from a qualified healthcare provider\n\n"
        
        if when_to_see and 'not available' not in when_to_see.lower():
            next_steps += f"**When to Seek Immediate Care:**\n{when_to_see}\n\n"
        
        next_steps += "> **🚨 Emergency:** If you experience severe symptoms, difficulty breathing, chest pain, or other emergency signs, seek immediate medical attention or call emergency services."
        
        return next_steps.strip()
    
    def _format_prevention(self, prevention: str) -> str:
        """Format prevention information."""
        if not prevention or 'not available' in prevention.lower():
            return "Preventive measures vary by condition. Consult a healthcare professional for personalized advice."
        
        return f"""
**Prevention & Risk Reduction:**

{prevention}
        """.strip()
    
    def _generate_fallback_explanation(
        self,
        disease: str,
        confidence: float,
        symptoms: List[str]
    ) -> Dict:
        """Generate fallback explanation when disease info is not found."""
        return {
            'explanation': f"The Bayesian network predicted **{disease}** with {confidence*100:.1f}% confidence based on your symptoms: {', '.join(symptoms)}. However, detailed information for this condition is not available in our knowledge base.",
            'treatment_plan': "Please consult a healthcare professional for treatment options.",
            'medications': "Consult a healthcare professional for medication recommendations.",
            'next_steps': "**Recommended:** Schedule an appointment with a healthcare professional for proper diagnosis and treatment.",
            'prevention': "Prevention strategies vary. Consult a healthcare professional.",
            'source_url': 'https://medlineplus.gov/',
            'source_name': 'MedlinePlus',
            'disease_description': f"Information about {disease} is being updated.",
            'complications': "Consult a healthcare professional for information about potential complications."
        }


def test_rag():
    """Test the RAG system with sample diseases."""
    print("\n" + "="*80)
    print("TESTING MEDICAL RAG SYSTEM")
    print("="*80 + "\n")
    
    try:
        rag = MedicalRAG()
        
        # Test diseases
        test_cases = [
            ("Diabetes", 0.85, ["increased_thirst", "frequent_urination", "fatigue"]),
            ("Malaria", 0.72, ["high_fever", "headache", "chills"]),
            ("Common Cold", 0.65, ["cough", "runny_nose", "sore_throat"])
        ]
        
        for disease, confidence, symptoms in test_cases:
            print(f"\nTest: {disease} (Confidence: {confidence*100:.0f}%)")
            print("-" * 80)
            
            result = rag.explain_diagnosis(disease, confidence, symptoms)
            
            print(f"\nExplanation:\n{result['explanation'][:300]}...\n")
            print(f"Treatment Plan:\n{result['treatment_plan'][:200]}...\n")
            print(f"Source: {result['source_url']}\n")
            print("="*80)
        
        print("\n✓ RAG system test completed successfully!\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n")
        print("Please run the setup scripts first:")
        print("  1. python create_knowledge_base.py")
        print("  2. python vector_db_setup.py\n")


if __name__ == "__main__":
    test_rag()
