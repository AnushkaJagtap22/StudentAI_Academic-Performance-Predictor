"""
Prediction Core Module for Kaggle Student Performance Prediction System.
Handles model loading, input validation, inference, risk level assignment, and feature attribution.
"""

import os
import joblib
import numpy as np
import pandas as pd
from preprocessing import KaggleStudentPreprocessor, CATEGORY_MAPS, FEATURE_COLS

RISK_THRESHOLDS = {
    'LOW': 80.0,
    'MODERATE': 60.0
}

PERFORMANCE_CATEGORIES = [
    (85.0, "Excellent", "Outstanding academic consistency. Top exam tier expected!"),
    (70.0, "Good", "Solid academic foundation with minor areas for optimization."),
    (55.0, "Average", "Moderate performance. Key academic drivers need targeted focus."),
    (0.0, "Needs Improvement", "Academic risk zone. Immediate structured intervention required.")
]

class StudentPredictor:
    """
    Inference manager for student performance prediction using Kaggle trained models.
    """
    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
            
        self.model_path = os.path.join(models_dir, "best_model.pkl")
        self.preprocessor_path = os.path.join(models_dir, "scaler.pkl")
        
        self.model = None
        self.preprocessor = None
        self.load_artifacts()

    def load_artifacts(self):
        """Load trained model and preprocessor."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Model artifacts not found in {os.path.dirname(self.model_path)}. "
                "Please run train_model.py first."
            )
            
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

    def validate_and_sanitize(self, input_dict: dict) -> dict:
        """Sanitize and encode raw user inputs."""
        clean = {}
        for col in self.preprocessor.feature_cols:
            val = input_dict.get(col, None)
            if val is None:
                # Default fallbacks
                if col in CATEGORY_MAPS:
                    clean[col] = list(CATEGORY_MAPS[col].keys())[1] # Pick default middle
                else:
                    clean[col] = 0.0
            else:
                clean[col] = val
        return clean

    def classify_performance(self, score: float):
        """Categorize predicted exam score into performance level and interpretation."""
        for threshold, category, interpretation in PERFORMANCE_CATEGORIES:
            if score >= threshold:
                return category, interpretation
        return "Needs Improvement", "Academic risk zone. Immediate structured intervention required."

    def determine_risk(self, score: float) -> str:
        """Classify academic risk level based on configurable project thresholds."""
        if score >= RISK_THRESHOLDS['LOW']:
            return "LOW RISK"
        elif score >= RISK_THRESHOLDS['MODERATE']:
            return "MODERATE RISK"
        else:
            return "HIGH RISK"

    def get_feature_importances(self) -> dict:
        """Extract global feature importances from trained model."""
        feature_cols = self.preprocessor.feature_cols
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            total = np.sum(importances) if np.sum(importances) > 0 else 1.0
            norm_imp = importances / total
            return dict(zip(feature_cols, [round(float(val), 4) for val in norm_imp]))
        elif hasattr(self.model, 'coef_'):
            coefs = np.abs(self.model.coef_)
            total = np.sum(coefs) if np.sum(coefs) > 0 else 1.0
            norm_coef = coefs / total
            return dict(zip(feature_cols, [round(float(val), 4) for val in norm_coef]))
        else:
            eq = 1.0 / len(feature_cols)
            return {col: eq for col in feature_cols}

    def predict(self, input_dict: dict) -> dict:
        """
        Execute prediction pipeline for Kaggle student profile.
        """
        clean_inputs = self.validate_and_sanitize(input_dict)
        scaled_features = self.preprocessor.transform_single(clean_inputs)
        
        raw_pred = self.model.predict(scaled_features)[0]
        predicted_marks = round(float(np.clip(raw_pred, 0.0, 100.0)), 1)
        
        category, interpretation = self.classify_performance(predicted_marks)
        risk = self.determine_risk(predicted_marks)
        importances = self.get_feature_importances()
        
        return {
            'inputs': clean_inputs,
            'predicted_marks': predicted_marks,
            'performance_category': category,
            'risk_level': risk,
            'interpretation': interpretation,
            'feature_importances': importances
        }
