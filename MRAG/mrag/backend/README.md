# MRAG Backend API

## Overview
FastAPI server that provides explainable AI diagnostic reasoning. Explains **WHY** the Bayesian model predicted a specific disease based on patient symptoms.

## Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

### `POST /mrag`
Generate diagnostic reasoning explanation.

**Request Body:**
```json
{
  "symptoms": ["fever", "headache", "fatigue", "body pain"],
  "bayesian_output": {
    "Malaria": 0.75,
    "Typhoid Fever": 0.15,
    "Dengue": 0.10
  }
}
```

**Response:**
```json
{
  "explanation": "Diagnostic Reasoning:\nThe combination of symptoms...",
  "top_disease": "Malaria",
  "confidence": 0.75,
  "all_predictions": {
    "Malaria": 0.75,
    "Typhoid Fever": 0.15,
    "Dengue": 0.10
  }
}
```

## Installation

### PowerShell
```powershell
cd MRAG/mrag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows CMD
```cmd
cd MRAG\mrag
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

The server will start at: `http://localhost:8000`

## Testing

### Using curl
```bash
curl -X POST http://localhost:8000/mrag \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["fever", "fatigue", "headache"],
    "bayesian_output": {
      "Malaria": 0.80,
      "Flu": 0.15,
      "Typhoid": 0.05
    }
  }'
```

### Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/mrag",
    json={
        "symptoms": ["fever", "headache", "cough"],
        "bayesian_output": {
            "COVID-19": 0.65,
            "Flu": 0.25,
            "Common Cold": 0.10
        }
    }
)

result = response.json()
print(result["explanation"])
print(f"Diagnosed: {result['top_disease']} ({result['confidence']:.1%})")
```

## Response Structure

### `explanation` (string)
Detailed multi-section explanation:
- **Diagnostic Reasoning**: Why this disease was predicted
- **Clinical Correlation**: How each symptom relates to the disease
- **Differential Analysis**: Why alternatives were ruled out
- **Confidence Assessment**: What the confidence score means

### `top_disease` (string)
The highest probability disease from Bayesian output.

### `confidence` (float)
Probability score for the top disease (0.0 to 1.0).

### `all_predictions` (dict)
Complete dictionary of all disease predictions with probabilities.

## Console Output
The server logs diagnostic details:
```
============================================================
🔍 DIAGNOSTIC EXPLANATION REQUEST
============================================================
Symptoms: ['fever', 'headache', 'fatigue']
Bayesian Model Output: {'Malaria': 0.75, 'Typhoid': 0.15, 'Dengue': 0.1}
Top Prediction: Malaria (75.00% confidence)
✅ Explanation generated successfully
============================================================
```

## Integration Example

### With Streamlit Frontend
```python
import streamlit as st
import requests

# After Bayesian model prediction
response = requests.post(
    "http://localhost:8000/mrag",
    json={
        "symptoms": collected_symptoms,
        "bayesian_output": model_predictions
    }
)

explanation_data = response.json()

st.header(f"Diagnosis: {explanation_data['top_disease']}")
st.metric("Confidence", f"{explanation_data['confidence']:.1%}")
st.markdown(explanation_data['explanation'])
```

## Error Handling

### No Disease Output
```json
{
  "explanation": "No disease prediction available.",
  "top_disease": null,
  "confidence": 0,
  "all_predictions": {}
}
```

### Invalid Request
Returns FastAPI validation error with 422 status code.

## CORS Configuration
Currently allows all origins (`*`). For production, restrict to specific domains:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

## Dependencies
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `pydantic`: Request/response validation
- `mrag_pipeline`: Core reasoning engine
- `langchain-huggingface`: Embeddings
- `transformers`: LLM inference
- `faiss-cpu`: Vector search

## Performance Notes
- First request may be slow (model loading)
- Subsequent requests: ~2-5 seconds
- Consider GPU deployment for production
- Can cache explanations for common symptom patterns

