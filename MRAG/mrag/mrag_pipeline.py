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

    top_disease = max(bayesian_output, key=bayesian_output.get)

    query = f"{top_disease} symptoms causes diagnosis"
    docs = db.similarity_search(query, k=2)

    # HARD LIMIT context (prevents repetition)
    context = ""
    for d in docs:
        clean = d.page_content.replace("\n", " ")
        context += clean[:200] + " "

    prompt = f"""
You are a medical assistant.

IMPORTANT RULES:
- Do NOT repeat sentences
- Do NOT repeat disease name unnecessarily
- Do NOT copy text verbatim
- Write original explanation in simple language

Disease: {top_disease}
Symptoms reported: {", ".join(symptoms)}

Medical reference (for understanding only):
{context}

Write the answer in this EXACT format:

Overview:
Explain what the disease is in one clear paragraph.

Symptom Explanation:
Explain how each symptom relates to the disease.

Differential Notes:
Explain why this disease fits better than others.

Sources:
MedQuAD
"""

    result = llm(prompt)[0]["generated_text"]
    return result
