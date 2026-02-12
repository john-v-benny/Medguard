from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from mrag_pipeline import mrag_explain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class MRAGRequest(BaseModel):
    symptoms: List[str]
    bayesian_output: Dict[str, float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/mrag")
def mrag(req: MRAGRequest):
    print("=" * 60)
    print("🔍 DIAGNOSTIC EXPLANATION REQUEST")
    print("=" * 60)
    print(f"Symptoms: {req.symptoms}")
    print(f"Bayesian Model Output: {req.bayesian_output}")
    
    # Get top prediction
    top_disease = None
    confidence = 0
    if req.bayesian_output:
        top_disease = max(req.bayesian_output, key=req.bayesian_output.get)
        confidence = req.bayesian_output[top_disease]
        print(f"Top Prediction: {top_disease} ({confidence:.2%} confidence)")
    
    # Generate diagnostic reasoning explanation
    explanation = mrag_explain(
        symptoms=req.symptoms,
        bayesian_output=req.bayesian_output
    )
    
    print("✅ Explanation generated successfully")
    print("=" * 60)
    
    return {
        "explanation": explanation,
        "top_disease": top_disease,
        "confidence": confidence,
        "all_predictions": req.bayesian_output
    }
