"""
Data Preprocessing Pipeline for Kaggle Student Performance Factors Dataset.
Handles categorical encoding, missing value imputation, domain boundary clipping, feature scaling, and train-test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Column Schema Definitions
NUMERICAL_COLS = [
    'Hours_Studied',
    'Attendance',
    'Sleep_Hours',
    'Previous_Scores',
    'Tutoring_Sessions',
    'Physical_Activity'
]

CATEGORICAL_COLS = [
    'Parental_Involvement',
    'Access_to_Resources',
    'Extracurricular_Activities',
    'Motivation_Level',
    'Internet_Access',
    'Family_Income',
    'Teacher_Quality',
    'School_Type',
    'Peer_Influence',
    'Learning_Disabilities',
    'Parental_Education_Level',
    'Distance_from_Home',
    'Gender'
]

FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS
TARGET_COL = 'Exam_Score'

# Ordinal & Binary Categorical Encoding Maps
CATEGORY_MAPS = {
    'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
    'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
    'Extracurricular_Activities': {'No': 0, 'Yes': 1},
    'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Internet_Access': {'No': 0, 'Yes': 1},
    'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
    'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
    'School_Type': {'Public': 0, 'Private': 1},
    'Peer_Influence': {'Negative': 0, 'Neutral': 1, 'Positive': 2},
    'Learning_Disabilities': {'No': 0, 'Yes': 1},
    'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
    'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2},
    'Gender': {'Female': 0, 'Male': 1}
}

# Reverse Mapping for UI display
REVERSE_MAPS = {col: {v: k for k, v in m.items()} for col, m in CATEGORY_MAPS.items()}

# Default Fill Values for Missing Data
IMPUTE_DEFAULTS = {
    'Teacher_Quality': 'Medium',
    'Parental_Education_Level': 'College',
    'Distance_from_Home': 'Moderate'
}

# Valid Domain Bounds
NUMERICAL_BOUNDS = {
    'Hours_Studied': (1.0, 50.0),
    'Attendance': (50.0, 100.0),
    'Sleep_Hours': (3.0, 12.0),
    'Previous_Scores': (40.0, 100.0),
    'Tutoring_Sessions': (0.0, 10.0),
    'Physical_Activity': (0.0, 10.0)
}

class KaggleStudentPreprocessor:
    """
    Preprocessor for Kaggle Student Performance Factors dataset.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = FEATURE_COLS
        self.numerical_cols = NUMERICAL_COLS
        self.categorical_cols = CATEGORICAL_COLS
        self.target_col = TARGET_COL
        self.is_fitted = False

    def clean_and_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean missing values, map categoricals to integer representations, and clip numerical bounds.
        """
        data = df.copy()
        
        # 1. Fill missing values
        for col, default_val in IMPUTE_DEFAULTS.items():
            if col in data.columns:
                data[col] = data[col].fillna(default_val)
                
        # 2. Encode categorical columns
        for col, mapping in CATEGORY_MAPS.items():
            if col in data.columns:
                # If values are already numeric, leave them, else map
                if data[col].dtype == object or isinstance(data[col].iloc[0], str):
                    data[col] = data[col].map(mapping).fillna(1).astype(int)
                    
        # 3. Clip numerical bounds
        for col, (min_val, max_val) in NUMERICAL_BOUNDS.items():
            if col in data.columns:
                data[col] = np.clip(data[col], min_val, max_val)
                
        if self.target_col in data.columns:
            data[self.target_col] = np.clip(data[self.target_col], 0.0, 100.0)
            
        return data

    def fit_transform_train(self, df: pd.DataFrame, test_size=0.2, random_state=42):
        """
        Fit preprocessor on train set only and return scaled feature matrices.
        """
        processed_df = self.clean_and_encode(df)
        
        X = processed_df[self.feature_cols]
        y = processed_df[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Fit scaler ONLY on numerical columns in training split to avoid data leakage
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[self.numerical_cols] = self.scaler.fit_transform(X_train[self.numerical_cols])
        X_test_scaled[self.numerical_cols] = self.scaler.transform(X_test[self.numerical_cols])
        
        self.is_fitted = True
        
        return X_train_scaled.values, X_test_scaled.values, y_train.values, y_test.values, X_train, X_test

    def transform_single(self, input_dict: dict) -> np.ndarray:
        """
        Transform a single student profile dictionary into scaled numpy array matching model feature order.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted yet.")
            
        row_df = pd.DataFrame([input_dict])
        processed_row = self.clean_and_encode(row_df)
        
        # Scale numerical columns
        processed_row[self.numerical_cols] = self.scaler.transform(processed_row[self.numerical_cols])
        
        return processed_row[self.feature_cols].values

    def save(self, filepath: str):
        """Save preprocessor state."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load preprocessor state."""
        return joblib.load(filepath)
