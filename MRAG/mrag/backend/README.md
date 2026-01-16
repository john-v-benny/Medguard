# Backend (FastAPI)

This is a minimal FastAPI server used by the frontend. It exposes `POST /mrag` that accepts JSON:

{
  "symptoms": ["fever", "fatigue"],
  "bayesian_output": {"Typhoid Fever": 0.62}
}

and returns a JSON with an `explanation` field.

Run locally (recommended inside a virtualenv):

PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Windows CMD
```cmd
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Test with curl:

```bash
curl -s -X POST http://localhost:8000/mrag -H "Content-Type: application/json" -d '{"symptoms":["fever"],"bayesian_output":{"Malaria":0.8}}'
```

The server reads `data/medquad_chunks.json` and returns a small excerpt for matching diseases.
