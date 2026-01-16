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
    print("Symptoms received:", req.symptoms)
    print("Bayesian predictions:", req.bayesian_output)
    explanation = mrag_explain(
        symptoms=req.symptoms,
        bayesian_output=req.bayesian_output
    )
    return {"explanation": explanation}
