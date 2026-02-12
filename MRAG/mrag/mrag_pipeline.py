import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

BASE_DIR = os.path.dirname(__file__)
VECTOR_DIR = os.path.join(BASE_DIR, "vector_db", "faiss_index")

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS DB
db = FAISS.load_local(
    VECTOR_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

# STRONGER MODEL (still CPU friendly)
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=-1,
    max_new_tokens=400
)

def mrag_explain(symptoms: list[str], bayesian_output: dict[str, float]) -> str:
    if not bayesian_output:
        return "No disease prediction available."

    # Get top predicted disease and its probability
    top_disease = max(bayesian_output, key=bayesian_output.get)
    confidence = bayesian_output[top_disease]
    
    # Get other diseases for differential analysis
    other_diseases = sorted(
        [(k, v) for k, v in bayesian_output.items() if k != top_disease],
        key=lambda x: x[1],
        reverse=True
    )[:2]  # Top 2 alternative diagnoses

    # Retrieve relevant medical knowledge
    query = f"{top_disease} symptoms clinical presentation diagnosis"
    docs = db.similarity_search(query, k=3)

    # Build context from retrieved documents
    context = ""
    for d in docs:
        clean = d.page_content.replace("\n", " ")
        context += clean[:250] + " "

    # Format alternative diseases
    alternatives_text = ""
    if other_diseases:
        alternatives_text = " vs ".join([f"{d[0]} ({d[1]:.1%})" for d in other_diseases])

    prompt = f"""
You are a medical AI explaining diagnostic reasoning.

TASK: Explain WHY the model predicted {top_disease} based on the symptoms.

PREDICTION:
- Diagnosed: {top_disease}
- Confidence: {confidence:.1%}
- Alternatives considered: {alternatives_text if alternatives_text else "None"}

PATIENT SYMPTOMS:
{", ".join(symptoms)}

MEDICAL KNOWLEDGE BASE:
{context}

INSTRUCTIONS:
1. Focus on the REASONING - why these symptoms point to {top_disease}
2. Explain the symptom-disease correlation
3. Address why alternatives were less likely
4. Use clear, simple language
5. Be concise and avoid repetition

OUTPUT FORMAT:

Diagnostic Reasoning:
[Explain how the combination of symptoms led to this diagnosis. Which symptoms are most indicative of {top_disease}?]

Clinical Correlation:
[Explain how each reported symptom specifically relates to the pathophysiology of {top_disease}]

Differential Analysis:
[Explain why {top_disease} was chosen over other possibilities. What made this diagnosis more likely than alternatives?]

Confidence Assessment:
[Explain what the {confidence:.1%} confidence means and what factors contributed to this level of certainty]

Reference: MedQuAD Database
"""

    result = llm(prompt)[0]["generated_text"]
    return result
