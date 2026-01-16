from typing import Dict, Any, List, Tuple, Optional
import os
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv, find_dotenv

ALLOWED_FEATURES = ["Fever", "Headache", "Cough", "Fatigue", "Body_Pain"]

SYSTEM_INSTRUCTIONS = """
You are a highly empathetic and efficient clinical intake assistant. Your primary goal is to collect data for 5 specific medical symptoms.

**Your Task:**
Analyze the user's message and the conversation history to fill in the 5 required features. Your response MUST be a single, clean JSON object.

**Required Features:**
1.  **Fever**: A numeric temperature in Fahrenheit (e.g., 98.6, 101.5).
2.  **Headache**: A severity rating from 0 to 10.
3.  **Cough**: A severity rating from 0 to 10.
4.  **Fatigue**: A severity rating from 0 to 10.
5.  **Body_Pain**: A severity rating from 0 to 10.

**Critical Rules for Your Response:**
1.  **Acknowledge and Empathize First**: Always start by acknowledging what the user has shared in a warm and empathetic tone.
2.  **Extract All Possible Values**:
    - If the user provides a number for a symptom (e.g., "my headache is an 8", "temp is 101.2"), extract it.
    - **Handle Negations**: If a user explicitly denies a symptom (e.g., "no headache", "I am not coughing"), you MUST set its value in `updates` to `0`.
3.  **Identify the Next Question**:
    - If the user mentions a symptom WITHOUT a value (e.g., "I have a cough"), you must prioritize asking for its value.
    - If multiple symptoms are mentioned, pick the first one mentioned that is still missing a value.
    - If no symptoms are mentioned, or all mentioned symptoms have values, pick the next feature from the `missing` list.
4.  **Formulate a Clear Question**: Your question should be specific to the symptom you are asking about.

**JSON Output Structure (Strictly follow this format):**
{
  "updates": {
    "Fever": <number or null>,
    "Headache": <0-10 or null>,
    "Cough": <0-10 or null>,
    "Fatigue": <0-10 or null>,
    "Body_Pain": <0-10 or null>
  },
  "acknowledgment": "<A genuinely empathetic sentence acknowledging the user's input.>",
  "next_symptom_to_ask": "<The name of the symptom to ask about next, or null if all are collected.>",
  "question_for_next_symptom": "<The full, natural question to ask the user about the next symptom.>"
}

**Example Scenarios:**
- **User**: "I feel terrible. I have a fever and no headache at all."
- **Your JSON**:
  {
    "updates": {"Fever": null, "Headache": 1},
    "acknowledgment": "I'm so sorry to hear you're feeling terrible with a fever. I've also noted that you don't have a headache.",
    "next_symptom_to_ask": "Fever",
    "question_for_next_symptom": "Could you please tell me your current temperature in Fahrenheit?"
  }

- **User**: "My cough is a 7/10 and I'm really tired."
- **Your JSON**:
  {
    "updates": {"Cough": 7, "Fatigue": null},
    "acknowledgment": "Thank you for sharing that. A cough at that level sounds really uncomfortable, and I understand you're feeling tired.",
    "next_symptom_to_ask": "Fatigue",
    "question_for_next_symptom": "On a scale of 0 to 10, how would you rate your fatigue level?"
  }

- **User**: "99.8" (in response to a question about fever)
- **Your JSON**:
  {
    "updates": {"Fever": 99.8},
    "acknowledgment": "Thank you, I've recorded your temperature as 99.8°F.",
    "next_symptom_to_ask": "Headache", // Assuming Headache is the next missing feature
    "question_for_next_symptom": "Now, could you please rate your headache on a scale of 0 to 10?"
  }

Provide ONLY the JSON object in your response.
"""

def _configure() -> None:
    load_dotenv(find_dotenv(), override=False)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            api_key = None
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found (.env or Streamlit secrets).")
    genai.configure(api_key=api_key)

def make_client(model_name: str = "models/gemini-2.5-flash"):
    _configure()
    
    # --- THIS IS THE FIX ---
    # Adjust safety settings to be more permissive for this medical context.
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    return genai.GenerativeModel(
        model_name,
        system_instruction=SYSTEM_INSTRUCTIONS,
        safety_settings=safety_settings  # Pass the settings here
    )

def _safe_json(txt: str) -> Dict[str, Any]:
    s = txt.strip() 
    if s.startswith("```") and s.endswith("```"):
        s = s.strip("`").strip()
        if "\n" in s:
            first, rest = s.split("\n", 1)
            if first.strip().lower() == "json":
                s = rest
    return json.loads(s)

def route_user_message(
    client,
    chat_history: List[Dict[str, str]],
    collected: Dict[str, Optional[float]]
) -> Dict[str, Any]:
    """
    Routes user message to Gemini and returns a structured dictionary.
    """
    last_msgs = chat_history[-6:]
    ctx = {
        "collected_so_far": {k: v for k, v in collected.items() if v is not None},
        "missing": [k for k, v in collected.items() if v is None],
        "chat_history": last_msgs,
    }
    prompt = json.dumps(ctx, indent=2)

    try:
        resp = client.generate_content(prompt)
        data = _safe_json(resp.text)

        # Basic validation
        if not all(k in data for k in ["updates", "acknowledgment", "next_symptom_to_ask", "question_for_next_symptom"]):
            raise ValueError("LLM response missing required keys")

        return data
    except (Exception, json.JSONDecodeError) as e:
        print(f"Error processing LLM response: {e}")
        # Return a safe fallback response
        missing = [k for k, v in collected.items() if v is None]
        next_symptom = missing[0] if missing else None
        return {
            "updates": {},
            "acknowledgment": "I'm sorry, I had a little trouble understanding that. Let's try again.",
            "next_symptom_to_ask": next_symptom,
            "question_for_next_symptom": f"Let's focus on {next_symptom}. Could you provide a value for it?" if next_symptom else ""
        }