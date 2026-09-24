import os
import nbformat as nbf

def create_model_training_notebook(output_path):
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Title Cell
    cells.append(nbf.v4.new_markdown_cell(
        "# Kaggle Student Performance Factors Dataset\n"
        "## Machine Learning Benchmark & Exploratory Data Analysis (EDA)\n\n"
        "This notebook documents the end-to-end ML workflow on the real Kaggle **Student Performance Factors** dataset (6,607 records, 20 attributes).\n"
        "- Dataset Inspection & Null Handling\n"
        "- Categorical Encoding & Numerical Scaling\n"
        "- Exploratory Data Analysis & Feature Correlations\n"
        "- Benchmark Training: Linear Regression, Decision Tree, Random Forest, Gradient Boosting\n"
        "- Model Evaluation Leaderboard (MAE, MSE, RMSE, R²)\n"
        "- Model Selection & Artifact Serialization\n"
    ))
    
    # Code Imports
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LinearRegression\n"
        "from sklearn.tree import DecisionTreeRegressor\n"
        "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n"
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
        "import joblib\n\n"
        "%matplotlib inline\n"
        "plt.style.use('seaborn-v0_8-whitegrid')\n"
    ))
    
    # Section 1
    cells.append(nbf.v4.new_markdown_cell("### 1. Load and Inspect Kaggle Dataset"))
    
    cells.append(nbf.v4.new_code_cell(
        "data_path = os.path.join('..', 'data', 'StudentPerformanceFactors.csv')\n"
        "df = pd.read_csv(data_path)\n"
        "print(f'Dataset Shape: {df.shape}')\n"
        "display(df.head())\n"
        "print('\\nMissing Values Count:')\n"
        "print(df.isnull().sum()[df.isnull().sum() > 0])"
    ))
    
    # Section 2
    cells.append(nbf.v4.new_markdown_cell("### 2. Exploratory Data Analysis (EDA)"))
    
    cells.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(10, 6))\n"
        "sns.histplot(df['Exam_Score'], kde=True, color='#2563eb', bins=30)\n"
        "plt.title('Distribution of Target Exam Score (Kaggle Dataset)', fontsize=14, fontweight='bold')\n"
        "plt.xlabel('Exam Score')\n"
        "plt.show()"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
        "sns.scatterplot(data=df, x='Hours_Studied', y='Exam_Score', ax=axes[0], alpha=0.5, color='#1e3a8a')\n"
        "axes[0].set_title('Hours Studied vs Exam Score')\n\n"
        "sns.scatterplot(data=df, x='Attendance', y='Exam_Score', ax=axes[1], alpha=0.5, color='#059669')\n"
        "axes[1].set_title('Attendance % vs Exam Score')\n\n"
        "sns.scatterplot(data=df, x='Previous_Scores', y='Exam_Score', ax=axes[2], alpha=0.5, color='#d97706')\n"
        "axes[2].set_title('Previous Scores vs Exam Score')\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # Section 3
    cells.append(nbf.v4.new_markdown_cell("### 3. Preprocessing & Encoding Pipeline"))
    
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "sys.path.append(os.path.join('..', 'src'))\n"
        "from preprocessing import KaggleStudentPreprocessor\n\n"
        "preprocessor = KaggleStudentPreprocessor()\n"
        "X_train_scaled, X_test_scaled, y_train, y_test, X_train_df, X_test_df = preprocessor.fit_transform_train(df)\n"
        "print(f'Train Samples: {X_train_scaled.shape[0]}, Test Samples: {X_test_scaled.shape[0]}')\n"
        "print(f'Number of Features: {X_train_scaled.shape[1]}')"
    ))
    
    # Section 4
    cells.append(nbf.v4.new_markdown_cell("### 4. Model Benchmarking & Evaluation Leaderboard"))
    
    cells.append(nbf.v4.new_code_cell(
        "models = {\n"
        "    'Linear Regression': LinearRegression(),\n"
        "    'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),\n"
        "    'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42),\n"
        "    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)\n"
        "}\n\n"
        "results = []\n"
        "for name, model in models.items():\n"
        "    model.fit(X_train_scaled, y_train)\n"
        "    preds = model.predict(X_test_scaled)\n"
        "    mae = mean_absolute_error(y_test, preds)\n"
        "    mse = mean_squared_error(y_test, preds)\n"
        "    rmse = np.sqrt(mse)\n"
        "    r2 = r2_score(y_test, preds)\n"
        "    results.append({'Model': name, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R2 Score': round(r2, 4)})\n\n"
        "leaderboard = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)\n"
        "display(leaderboard)"
    ))
    
    nb['cells'] = cells
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Jupyter notebook created at: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    nb_path = os.path.join(current_dir, "model_training.ipynb")
    create_model_training_notebook(nb_path)
