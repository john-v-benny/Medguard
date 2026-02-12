# Bayesian Network Medical Chatbot

## Overview

An intelligent medical chatbot that uses Bayesian networks for probabilistic disease prediction based on symptoms. The chatbot interactively asks questions until it reaches sufficient confidence in its diagnosis.

## Features

- **Probabilistic Disease Prediction**: Uses Bayesian inference for explainable predictions
- **Interactive Symptom Collection**: Asks follow-up questions when confidence is low
- **Confidence Scoring**: Displays probability scores for all predictions
- **No Hallucinations**: Only uses trained probability distributions from verified data
- **41 Diseases Coverage**: Trained on comprehensive medical dataset

## Dataset

- **Source**: Kaggle Disease Prediction Dataset
- **Diseases**: 41 different diseases
- **Symptoms**: 132 unique symptoms
- **Training Samples**: 4,920 cases
- **Test Samples**: 42 cases

## Installation

1. **Create Virtual Environment**:
```bash
python -m venv venv
```

2. **Activate Virtual Environment**:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebooks in order:

1. **01_data_exploration.ipynb**: Explore and understand the dataset
2. **02_bayesian_network_training.ipynb**: Train the Bayesian network model
3. **03_chatbot_implementation.ipynb**: Interactive chatbot implementation
4. **04_demo_and_testing.ipynb**: Test and demonstrate the chatbot

## Project Structure

```
bayesian_medical_chatbot/
├── data/                    # Medical datasets
├── notebooks/               # Jupyter notebooks
├── models/                  # Trained Bayesian network models
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Medical Disclaimer

⚠️ **IMPORTANT**: This chatbot is for educational and demonstration purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

## Architecture

The system uses a Naive Bayes structure where:
- **Disease** is the root node (41 possible values)
- **Symptoms** are child nodes (132 binary features)
- **Inference** uses Variable Elimination algorithm
- **Questioning Strategy** uses information gain to select next symptom to ask

## License

Educational use only. Dataset from Kaggle with appropriate attribution.
