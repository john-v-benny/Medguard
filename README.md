# 🤖 AI-Powered Medical Symptom Analyzer

A comprehensive machine learning application that combines **Gaussian Naive Bayes** for disease prediction with **Google Gemini AI** for conversational symptom collection.

## 📋 Project Overview

This project implements an intelligent medical symptom analysis system with two main components:

1. **Bayesian Disease Predictor**: Trained Gaussian Naive Bayes model that predicts diseases based on 5 key symptoms
2. **Conversational AI Interface**: Gemini-powered chatbot that naturally collects symptom data from users
3. **Interactive Dashboard**: Streamlit-based web application that ties everything together

## 🏗️ Project Structure

```
new_bayesian/
├── bayesian_5000.ipynb           # Main ML model training & analysis
├── streamlit_app/
│   ├── app.py                    # Main Streamlit application
│   ├── models/
│   │   └── predictor.py          # Disease prediction logic
│   └── llm/
│       └── gemini_agent.py       # Gemini AI integration
├── saved_models/
│   ├── bayesian_disease_model.pkl   # Trained model
│   └── model_info.json              # Model metadata
├── data/
│   └── dataset1.csv              # Training dataset
├── .gitignore
├── README.md
└── requirements.txt
```

## 🎯 Key Features

### 1. **Bayesian Model Training** (`bayesian_5000.ipynb`)
- Loads and preprocesses medical symptom data
- Trains Gaussian Naive Bayes classifier
- Evaluates model performance with:
  - Confusion matrix
  - Classification report
  - Per-class accuracy metrics
- Visualizes decision boundaries and feature importance
- Generates sample predictions with confidence scores

### 2. **Conversational Symptom Collection** (`gemini_agent.py`)
- Natural language processing via Google Gemini
- Intelligent symptom extraction from user input
- Handles negations (e.g., "no headache" → value 0)
- Tracks missing symptoms and asks follow-up questions
- Returns structured JSON with predictions and acknowledgments

### 3. **Interactive Web Application** (`app.py`)
- Real-time symptom progress tracking
- Chat-based symptom collection interface
- Automatic disease prediction when all symptoms collected
- Confidence score visualization
- Probability distribution charts
- Comparison heatmaps vs. learned disease patterns

## 📊 Symptoms Tracked

| Symptom | Range | Unit |
|---------|-------|------|
| Fever | 95.0 - 105.0 | °F |
| Headache | 0 - 10 | Severity Scale |
| Cough | 0 - 10 | Severity Scale |
| Fatigue | 0 - 10 | Severity Scale |
| Body Pain | 0 - 10 | Severity Scale |

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone or download the project**
```bash
cd new_bayesian
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash

pip install google-generativeai python-dotenv seaborn matplotlib scikit-learn fastapi uvicorn pandas numpy 

```

4. **Set up environment variables**
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

5. **Running the Application**


```bash
cd backend
uvicorn main:app --reload  

```
   
Open a new terminal

```

cd frontend
cd my-app
npm install
npm i clsx lucide-react axios
npm run dev

```

## 📈 Model Performance

The trained Bayesian model achieves:
- **Training Accuracy**: ~95%+ (varies by dataset)
- **Test Accuracy**: Validated on hold-out test set
- **Confidence Threshold**: 0.7 (70%) for high-confidence predictions

### Evaluation Metrics
- Precision, Recall, F1-Score per disease class
- Confusion matrix showing misclassification patterns
- Per-class accuracy analysis

## 🔄 How It Works

### Data Flow

```
User Input 
    ↓
Gemini AI (Natural Language Processing)
    ↓
Extract Symptoms (JSON)
    ↓
Accumulate Symptoms (Session State)
    ↓
All 5 Symptoms Collected?
    ├─ No → Ask Next Symptom
    └─ Yes → Run Bayesian Model
         ↓
    Disease Prediction + Confidence Score
         ↓
    Display Results & Visualizations
```

### Key Functions

**`predict_with_uncertainty(model, symptoms, threshold)`**
- Input: Dictionary of symptom values
- Output: Disease probabilities and uncertainty assessment
- Compares max probability against threshold

**`route_user_message(client, chat_history, collected)`**
- Processes user input through Gemini
- Extracts symptom values
- Returns structured response with next question

**`DiseasePredictor.predict(symptoms, threshold)`**
- Main prediction function
- Loads trained model
- Returns prediction result with confidence metrics

## 📝 Example Usage

### Via Next.js and FastAPI
1. Start the app
- cd frontend 
- npm run dev
- cd backend 
- uvicorn main:app --reload
2. Chat with the AI to describe your symptoms
3. Model automatically predicts disease when ready
4. View confidence scores and probability distributions

### Via Python Code
```python
from models.predictor import DiseasePredictor

# Load model
predictor = DiseasePredictor('saved_models/bayesian_disease_model.pkl', 
                              'saved_models/model_info.json')

# Make prediction
symptoms = {
    "Fever": 101.2,
    "Headache": 8.0,
    "Cough": 3.0,
    "Fatigue": 7.0,
    "Body_Pain": 5.0
}

result = predictor.predict(symptoms, confidence_threshold=0.7)
print(result)
```

## 🛠️ Configuration

### Confidence Threshold
Adjust in `streamlit_app/app.py`:
```python
CONFIDENCE_THRESHOLD = 0.7  # Range: 0.0 - 1.0
```

### Model Selection
Change Gemini model in `gemini_agent.py`:
```python
def make_client(model_name: str = "models/gemini-2.5-flash"):
```

## 📦 Dependencies

See `requirements.txt` for complete list:
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning (Naive Bayes)
- `google-generativeai`: Gemini API
- `matplotlib`, `seaborn`: Visualization
- `plotly`: Interactive charts
- `python-dotenv`: Environment management

## ⚠️ Important Notes

1. **Medical Disclaimer**: This tool is for educational purposes only. Not intended for medical diagnosis.
2. **API Key**: Keep your Gemini API key secure. Never commit `.env` to version control.
3. **Data Privacy**: User symptoms are processed through Gemini API. Review privacy policy before use.
4. **Model Retraining**: Retrain model with new data in `bayesian_5000.ipynb` as needed.

## 🔍 Troubleshooting

### Gemini API Errors
- Verify API key is correct and active
- Check `.env` 
- Ensure sufficient API quota

### Model Loading Issues
- Confirm `bayesian_disease_model.pkl` exists in `saved_models/`
- Retrain model if file is corrupted: Run `bayesian_5000.ipynb`


## 📚 References

- [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes)
- [Google Generative AI](https://ai.google.dev/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Bayesian Methods](https://en.wikipedia.org/wiki/Bayes%27_theorem)

## 📄 License

This project is provided as-is for educational purposes.

## 👨‍💻 Author

Created as a demonstration of ML + LLM integration for medical symptom analysis.

## 🤝 Contributing

Improvements and bug reports are welcome! Please ensure:
- Code follows PEP 8 style guide
- All features are documented
- Sensitive information is not committed

---

**Last Updated**: October 31, 2025
