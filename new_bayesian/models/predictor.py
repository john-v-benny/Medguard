import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class DiseasePredictor:
    def __init__(self, model_path: str, model_info_path: str):
        self.model_path = model_path
        self.model_info_path = model_info_path
        self.model = None
        self.model_info = None
        self.feature_names = None
        self.classes = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and its information"""
        try:
            # Load the model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load model information
            with open(self.model_info_path, 'r') as f:
                self.model_info = json.load(f)
            
            self.feature_names = self.model_info['features']
            self.classes = self.model_info['classes']
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def validate_input(self, symptoms: Dict[str, float]) -> bool:
        """Validate user input"""
        required_features = set(self.feature_names)
        provided_features = set(symptoms.keys())
        
        if required_features != provided_features:
            missing = required_features - provided_features
            extra = provided_features - required_features
            if missing:
                print(f"Missing features: {missing}")
            if extra:
                print(f"Extra features: {extra}")
            return False
        
        # Validate ranges
        ranges = self.model_info['feature_ranges']
        for feature, value in symptoms.items():
            if feature in ranges:
                min_val = ranges[feature]['min']
                max_val = ranges[feature]['max']
                if not (min_val <= value <= max_val):
                    print(f"{feature} value {value} is outside valid range [{min_val}, {max_val}]")
                    return False
        
        return True
    
    def predict(self, symptoms: Dict[str, float], confidence_threshold: float = 0.6) -> Dict:
        """Make prediction with confidence check"""
        if not self.validate_input(symptoms):
            return {
                'success': False,
                'error': 'Invalid input provided',
                'prediction': None,
                'confidence': 0,
                'all_probabilities': {}
            }
        
        try:
            # Prepare input array
            input_array = np.array([[symptoms[feature] for feature in self.feature_names]])
            
            # Make prediction
            prediction = self.model.predict(input_array)[0]
            probabilities = self.model.predict_proba(input_array)[0]
            
            # Get confidence (highest probability)
            max_confidence = np.max(probabilities)
            
            # Create probability dictionary
            prob_dict = {self.classes[i]: float(probabilities[i]) for i in range(len(self.classes))}
            
            # Check confidence threshold
            if max_confidence < confidence_threshold:
                return {
                    'success': True,
                    'prediction': None,
                    'confidence': float(max_confidence),
                    'all_probabilities': prob_dict,
                    'message': 'Low confidence prediction. Please consult a medical professional.',
                    'threshold_met': False
                }
            
            return {
                'success': True,
                'prediction': prediction,
                'confidence': float(max_confidence),
                'all_probabilities': prob_dict,
                'threshold_met': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction error: {str(e)}',
                'prediction': None,
                'confidence': 0,
                'all_probabilities': {}
            }
    
    def get_top_predictions(self, symptoms: Dict[str, float], top_n: int = 3) -> List[Tuple[str, float]]:
        """Get top N predictions"""
        result = self.predict(symptoms, confidence_threshold=0.0)
        if result['success']:
            probabilities = result['all_probabilities']
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            return sorted_probs[:top_n]
        return []