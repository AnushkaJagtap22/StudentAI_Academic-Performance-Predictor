# 🎓 Student Performance Prediction

An end-to-end interactive Machine Learning application built using the official **Kaggle Student Performance Factors Dataset** (6,607 student records, 20 features).

The system extends conventional machine learning prediction with **What-If simulation**, **Goal-Based target optimization**, **Single-Factor Counterfactual sensitivity analysis**, and **Personalized academic action planning**.

---

## 📌 Kaggle Dataset Overview

- **Source:** Kaggle Student Performance Factors Dataset
- **Records:** 6,607 real student profiles
- **Attributes:** 20 features covering academic, behavioral, lifestyle, and socio-economic drivers
- **Target Variable:** `Exam_Score` (0–100)

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Hours_Studied` | Continuous | Weekly self-study hours (1 - 44 hrs) |
| `Attendance` | Continuous | Class attendance percentage (50% - 100%) |
| `Previous_Scores` | Continuous | Scores in previous academic exams (40 - 100) |
| `Tutoring_Sessions` | Continuous | Monthly 1-on-1 tutoring sessions (0 - 8) |
| `Sleep_Hours` | Continuous | Nightly sleep duration (4 - 10 hrs) |
| `Physical_Activity` | Continuous | Weekly exercise duration (0 - 6 hrs) |
| `Parental_Involvement` | Categorical | Low / Medium / High |
| `Access_to_Resources` | Categorical | Low / Medium / High |
| `Motivation_Level` | Categorical | Low / Medium / High |
| `Family_Income` | Categorical | Low / Medium / High |
| `Teacher_Quality` | Categorical | Low / Medium / High (Missing imputed) |
| `Parental_Education_Level`| Categorical | High School / College / Postgraduate |
| `Distance_from_Home` | Categorical | Near / Moderate / Far |
| `Peer_Influence` | Categorical | Negative / Neutral / Positive |
| `School_Type` | Categorical | Public / Private |
| `Internet_Access` | Categorical | No / Yes |
| `Learning_Disabilities` | Categorical | No / Yes |
| `Extracurricular_Activities`| Categorical | No / Yes |
| `Gender` | Categorical | Female / Male |
| **`Exam_Score`** | Target | Final Examination Score (0 - 100) |

---

## 📈 ML Benchmark Leaderboard

| Algorithm | MAE | RMSE | R² Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** (Selected Best) | **0.4414** | **1.7987** | **0.7711** |
| **Gradient Boosting Regressor** | 0.7849 | 2.0006 | 0.7169 |
| **Random Forest Regressor** | 1.1472 | 2.1913 | 0.6603 |
| **Decision Tree Regressor** | 1.6446 | 2.6746 | 0.4939 |

---

## ✨ System Features

1. **🎯 Main Prediction Engine**: Real-time exam score prediction, risk level badge (`LOW RISK`, `MODERATE RISK`, `HIGH RISK`), and feature driver attributions.
2. **⚡ What-If Simulator**: Real-time counterfactual model evaluation comparing baseline vs scenario changes on Kaggle features.
3. **🚀 Goal-Based Optimizer**: Grid search optimizer finding minimal-effort habit roadmaps to reach target exam marks.
4. **🌟 Best Improvement Focus**: Single-factor sensitivity analysis isolating individual driver impacts.
5. **💡 Personalized Action Plan**: Contextual recommendation engine matching student performance gaps with actionable study strategies.

---

## 🚀 How to Run

```bash
cd "C:\Users\Anushka\Documents\BVCOEP\PBL\AI PBL\AI-Student-Performance-Prediction"

# Launch Streamlit Application
streamlit run app.py
```
