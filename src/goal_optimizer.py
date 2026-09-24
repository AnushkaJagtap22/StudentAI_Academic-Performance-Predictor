"""
Goal-Based Prediction Optimizer & Recommendation Report Engine for Kaggle Student Performance.
Derives optimal habit roadmaps and generates a detailed Goal Achievement Recommendation Report.
"""

import numpy as np
import pandas as pd

EFFORT_WEIGHTS = {
    'Hours_Studied': 1.5,
    'Attendance': 1.0,
    'Tutoring_Sessions': 1.2,
    'Sleep_Hours': 0.8,
    'Physical_Activity': 0.6,
    'Motivation_Level': 1.0,
    'Access_to_Resources': 1.1
}

class GoalOptimizer:
    """
    Search engine finding feasible metric improvements to reach target exam score
    and generating formal Goal Achievement Recommendation Reports.
    """
    def __init__(self, predictor):
        self.predictor = predictor

    def compute_effort_score(self, current_inputs: dict, candidate_inputs: dict) -> float:
        """Compute weighted effort score required for candidate habit adjustments."""
        effort = 0.0
        
        h_delta = candidate_inputs.get('Hours_Studied', 0.0) - current_inputs.get('Hours_Studied', 0.0)
        if h_delta > 0:
            effort += EFFORT_WEIGHTS['Hours_Studied'] * (h_delta / 10.0)
            
        a_delta = candidate_inputs.get('Attendance', 0.0) - current_inputs.get('Attendance', 0.0)
        if a_delta > 0:
            effort += EFFORT_WEIGHTS['Attendance'] * (a_delta / 20.0)
            
        t_delta = candidate_inputs.get('Tutoring_Sessions', 0.0) - current_inputs.get('Tutoring_Sessions', 0.0)
        if t_delta > 0:
            effort += EFFORT_WEIGHTS['Tutoring_Sessions'] * (t_delta / 4.0)
            
        s_delta = candidate_inputs.get('Sleep_Hours', 7.0) - current_inputs.get('Sleep_Hours', 7.0)
        if s_delta > 0:
            effort += EFFORT_WEIGHTS['Sleep_Hours'] * (s_delta / 3.0)
            
        return round(float(effort), 3)

    def optimize_for_target(self, current_inputs: dict, target_score: float, max_scenarios=5) -> dict:
        """
        Grid search over controllable factor combinations to achieve target score.
        """
        baseline_res = self.predictor.predict(current_inputs)
        baseline_score = baseline_res['predicted_marks']
        
        if baseline_score >= target_score:
            return {
                'status': 'ALREADY_ACHIEVED',
                'message': f"Your current predicted exam score of {baseline_score} already meets or exceeds your target of {target_score}!",
                'baseline_score': baseline_score,
                'target_score': target_score,
                'scenarios': []
            }
            
        # Theoretical maximum achievable score with maxed controllable parameters
        max_inputs = current_inputs.copy()
        max_inputs['Hours_Studied'] = min(max(current_inputs['Hours_Studied'] + 15, 35.0), 44.0)
        max_inputs['Attendance'] = 100.0
        max_inputs['Tutoring_Sessions'] = 8.0
        max_inputs['Sleep_Hours'] = 8.0
        max_inputs['Motivation_Level'] = 'High'
        max_inputs['Access_to_Resources'] = 'High'
        
        max_pred = self.predictor.predict(max_inputs)['predicted_marks']
        
        if max_pred < target_score - 1.0:
            return {
                'status': 'UNFEASIBLE',
                'message': f"Target score of {target_score} exceeds maximum predicted exam capacity ({max_pred}) given previous scores of {current_inputs.get('Previous_Scores', 60)}.",
                'baseline_score': baseline_score,
                'target_score': target_score,
                'max_possible_score': max_pred,
                'scenarios': []
            }

        cur_hrs = float(current_inputs.get('Hours_Studied', 20))
        cur_att = float(current_inputs.get('Attendance', 75))
        cur_tut = float(current_inputs.get('Tutoring_Sessions', 1))

        # Comprehensive search space
        hrs_grid = np.unique(np.clip(np.linspace(cur_hrs, min(cur_hrs + 16, 44.0), num=8), cur_hrs, 44.0))
        att_grid = np.unique(np.clip(np.linspace(cur_att, 100.0, num=6), cur_att, 100.0))
        tut_grid = np.unique(np.clip(np.linspace(cur_tut, min(cur_tut + 5, 8.0), num=5), cur_tut, 8.0))

        candidates = []
        for h in hrs_grid:
            for a in att_grid:
                for t in tut_grid:
                    cand = current_inputs.copy()
                    cand['Hours_Studied'] = round(float(h), 1)
                    cand['Attendance'] = round(float(a), 1)
                    cand['Tutoring_Sessions'] = round(float(t), 1)

                    pred = self.predictor.predict(cand)['predicted_marks']

                    if pred >= target_score - 0.5:
                        effort = self.compute_effort_score(current_inputs, cand)
                        candidates.append({
                            'inputs': cand,
                            'predicted_marks': pred,
                            'effort_score': effort,
                            'delta_hours': round(cand['Hours_Studied'] - cur_hrs, 1),
                            'delta_attendance': round(cand['Attendance'] - cur_att, 1),
                            'delta_tutoring': round(cand['Tutoring_Sessions'] - cur_tut, 1)
                        })

        candidates = sorted(candidates, key=lambda x: (x['effort_score'], abs(x['predicted_marks'] - target_score)))
        
        selected = []
        seen = set()
        for c in candidates:
            key = (c['inputs']['Hours_Studied'], c['inputs']['Attendance'], c['inputs']['Tutoring_Sessions'])
            if key not in seen:
                seen.add(key)
                selected.append(c)
            if len(selected) >= max_scenarios:
                break

        return {
            'status': 'SUCCESS',
            'baseline_score': baseline_score,
            'target_score': target_score,
            'max_possible_score': max_pred,
            'scenarios': selected,
            'disclaimer': "All target plans are model-based estimates derived from Kaggle statistical learning."
        }

    def generate_recommendation_report(self, current_inputs: dict, target_score: float) -> dict:
        """
        Generate a structured, publication-grade Goal Achievement Recommendation Report.
        
        Returns:
            dict containing report metadata, executive summary, strategy breakdown table,
            action milestones, and plain-text markdown report for export.
        """
        opt_res = self.optimize_for_target(current_inputs, target_score)
        baseline_score = opt_res['baseline_score']
        gap = round(target_score - baseline_score, 1)
        
        if opt_res['status'] != 'SUCCESS':
            return {
                'status': opt_res['status'],
                'message': opt_res.get('message', ''),
                'report_md': f"# 🎓 Goal Achievement Recommendation Report\n\n**Status:** {opt_res.get('message', '')}"
            }
            
        top_scenario = opt_res['scenarios'][0]
        effort_level = "LOW EFFORT" if top_scenario['effort_score'] < 0.8 else ("MODERATE EFFORT" if top_scenario['effort_score'] < 1.8 else "HIGH INTENSITY")
        
        # Build Markdown Text Report for Download
        report_md = f"""# 🎓 GOAL ACHIEVEMENT RECOMMENDATION REPORT
*AI-Based Student Performance Prediction System*
*Generated using Kaggle Machine Learning Regression Model*

---

## 📌 Executive Summary
- **Current Predicted Score:** `{baseline_score} / 100`
- **Target Exam Score:** `{target_score} / 100`
- **Required Score Lift (Gap):** `+{gap} Marks`
- **Recommended Strategy Effort:** `{effort_level}` (Effort Rating: {top_scenario['effort_score']:.2f})
- **Primary Model Selected:** `{self.predictor.model.__class__.__name__}`

---

## 🎯 Recommended Habit Adjustments (Primary Plan)

| Controllable Driver | Baseline Value | Recommended Target | Delta Adjustment | Impact Description |
| :--- | :---: | :---: | :---: | :--- |
| **Weekly Study Hours** | {current_inputs.get('Hours_Studied', 0)} hrs/wk | **{top_scenario['inputs']['Hours_Studied']} hrs/wk** | `+{top_scenario['delta_hours']} hrs` | Increases core subject retention & practice |
| **Class Attendance** | {current_inputs.get('Attendance', 0)}% | **{top_scenario['inputs']['Attendance']}%** | `+{top_scenario['delta_attendance']}%` | Ensures full lecture concept coverage |
| **Tutoring Sessions** | {current_inputs.get('Tutoring_Sessions', 0)} /mo | **{top_scenario['inputs']['Tutoring_Sessions']} /mo** | `+{top_scenario['delta_tutoring']} sessions` | 1-on-1 problem-solving mentorship |

**Expected Exam Score Output:** `{top_scenario['predicted_marks']} / 100`

---

## 🚀 Step-by-Step Action Roadmap

### 📅 Phase 1: Attendance & Classroom Engagement (Weeks 1–2)
- Target class attendance at **{top_scenario['inputs']['Attendance']}%** or higher.
- Review lecture notes within 24 hours of attending each session.
- Sit in front rows to minimize classroom distractions.

### ⏱️ Phase 2: Study Time Expansion (Weeks 2–4)
- Increase dedicated weekly study time to **{top_scenario['inputs']['Hours_Studied']} hours per week** (~{round(top_scenario['inputs']['Hours_Studied']/7, 1)} hrs/day).
- Implement 50-minute focused Pomodoro study blocks with 10-minute breaks.
- Prioritize active recall (flashcards, timed practice tests) over passive reading.

### 👨‍🏫 Phase 3: Tutoring & Mentorship Support (Weeks 4+)
- Schedule **{top_scenario['inputs']['Tutoring_Sessions']} monthly tutoring sessions** for weak modules.
- Maintain a running list of specific problem topics to address during tutoring.

---

*Disclaimer: All recommendations are simulated estimates produced by trained ML regression algorithms based on Kaggle statistical distributions.*
"""
        
        return {
            'status': 'SUCCESS',
            'baseline_score': baseline_score,
            'target_score': target_score,
            'gap': gap,
            'effort_level': effort_level,
            'primary_plan': top_scenario,
            'all_scenarios': opt_res['scenarios'],
            'report_md': report_md
        }
