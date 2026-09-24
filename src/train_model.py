"""
Model Training and Automated Benchmark Module for Kaggle Student Performance Factors Dataset.
Compares Linear Regression, Decision Tree, Random Forest, and Gradient Boosting Regressors.
Automatically selects and serializes the top-performing model.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import KaggleStudentPreprocessor

def evaluate_model(model, X_test, y_test):
    """Compute MAE, MSE, RMSE, and R2 score for a given model."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    return {
        'MAE': round(float(mae), 4),
        'MSE': round(float(mse), 4),
        'RMSE': round(float(rmse), 4),
        'R2': round(float(r2), 4)
    }

def train_and_evaluate_models(data_path: str, models_dir: str):
    """
    Train multiple ML algorithms on Kaggle dataset, generate benchmark leaderboard,
    select the best model based on R2 score & RMSE, and save artifacts.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Preprocess
    preprocessor = KaggleStudentPreprocessor()
    X_train_scaled, X_test_scaled, y_train, y_test, X_train_df, X_test_df = preprocessor.fit_transform_train(
        df, test_size=0.2, random_state=42
    )
    
    # Candidate ML models
    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
    }
    
    results = {}
    trained_model_objs = {}
    
    print("\n--- Training ML Models on Kaggle Dataset (6,607 Records) ---")
    for name, model in candidate_models.items():
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_test_scaled, y_test)
        results[name] = metrics
        trained_model_objs[name] = model
        print(f"{name:20s} | MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | R²: {metrics['R2']:.4f}")
        
    leaderboard_df = pd.DataFrame(results).T.sort_values(by=['R2', 'RMSE'], ascending=[False, True])
    
    best_model_name = leaderboard_df.index[0]
    best_model = trained_model_objs[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n==========================================")
    print(f"AUTOMATICALLY SELECTED BEST MODEL: {best_model_name}")
    print(f"Metrics -> R²: {best_metrics['R2']} | RMSE: {best_metrics['RMSE']} | MAE: {best_metrics['MAE']}")
    print(f"==========================================\n")
    
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    preprocessor_path = os.path.join(models_dir, 'scaler.pkl')
    metrics_path = os.path.join(models_dir, 'model_metrics.json')
    
    joblib.dump(best_model, best_model_path)
    preprocessor.save(preprocessor_path)
    
    metrics_payload = {
        'best_model_name': best_model_name,
        'leaderboard': results,
        'feature_names': preprocessor.feature_cols
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_payload, f, indent=4)
        
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved preprocessor to: {preprocessor_path}")
    print(f"Saved metrics summary to: {metrics_path}")
    
    return best_model_name, results, leaderboard_df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_file = os.path.join(project_root, "data", "StudentPerformanceFactors.csv")
    models_directory = os.path.join(project_root, "models")
    
    train_and_evaluate_models(data_file, models_directory)
