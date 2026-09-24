"""
Recommendation Engine for Kaggle Student Performance Factors System.
Implements:
1. Counterfactual Single-Factor Sensitivity Analysis (Best Improvement Recommendation)
2. Personalized Academic Action Plan Generator
"""

CONTROLLABLE_FACTORS = {
    'Hours_Studied': {
        'name': 'Weekly Study Hours',
        'step': 5.0,
        'max': 44.0,
        'unit': 'hrs/week',
        'icon': '⏱️'
    },
    'Attendance': {
        'name': 'Class Attendance',
        'step': 10.0,
        'max': 100.0,
        'unit': '%',
        'icon': '📅'
    },
    'Tutoring_Sessions': {
        'name': 'Tutoring Sessions',
        'step': 2.0,
        'max': 8.0,
        'unit': 'sessions/mo',
        'icon': '👨‍🏫'
    },
    'Sleep_Hours': {
        'name': 'Sleep Duration',
        'step': 1.5,
        'max': 9.0,
        'unit': 'hrs/night',
        'icon': '🛌'
    },
    'Physical_Activity': {
        'name': 'Physical Activity',
        'step': 2.0,
        'max': 6.0,
        'unit': 'hrs/week',
        'icon': '🏃'
    }
}

SUGGESTIONS_DB = {
    'Hours_Studied': [
        "Increase dedicated self-study by 45-60 minutes daily in quiet environments.",
        "Use active recall techniques (flashcards, past paper practice) rather than passive textbook reading.",
        "Block study hours into focused 50-minute blocks with 10-minute breaks."
    ],
    'Attendance': [
        "Maintain class attendance above 85% to ensure full lecture concept coverage.",
        "Review lecture summary notes within 24 hours of each missed or attended session.",
        "Engage actively in classroom discussion to reinforce memory retention."
    ],
    'Tutoring_Sessions': [
        "Schedule bi-weekly 1-on-1 tutoring sessions to clarify challenging module concepts.",
        "Prepare specific question lists prior to attending tutoring or office hours.",
        "Work through step-by-step problem sets alongside tutors."
    ],
    'Sleep_Hours': [
        "Target 7.5 to 8.5 hours of consistent sleep nightly to optimize memory consolidation.",
        "Establish a regular sleep schedule, avoiding late-night cramming before exams.",
        "Minimize blue light exposure (phones, laptops) 30 minutes before bed."
    ],
    'Physical_Activity': [
        "Incorporate 30 minutes of moderate physical activity 3-4 times per week to reduce stress.",
        "Take short active walks between long study sessions to improve mental alertness."
    ]
}

class RecommendationEngine:
    """
    Engine for generating counterfactual sensitivity analysis and personalized improvement plans on Kaggle data.
    """
    def __init__(self, predictor):
        self.predictor = predictor

    def analyze_best_improvement(self, current_inputs: dict) -> dict:
        """
        Perform actual counterfactual single-factor what-if predictions on Kaggle features.
        Vary ONE factor at a time while holding all other features constant.
        """
        baseline_res = self.predictor.predict(current_inputs)
        baseline_score = baseline_res['predicted_marks']
        
        factor_gains = []
        
        for factor_key, info in CONTROLLABLE_FACTORS.items():
            curr_val = float(current_inputs.get(factor_key, 0.0))
            max_val = info['max']
            step = info['step']
            
            test_val = min(curr_val + step, max_val)
            
            if test_val <= curr_val:
                gain = 0.0
            else:
                test_inputs = current_inputs.copy()
                test_inputs[factor_key] = test_val
                test_res = self.predictor.predict(test_inputs)
                gain = max(0.0, round(test_res['predicted_marks'] - baseline_score, 1))
                
            factor_gains.append({
                'factor': factor_key,
                'name': info['name'],
                'current_value': curr_val,
                'simulated_value': round(test_val, 1),
                'unit': info['unit'],
                'predicted_gain': gain,
                'icon': info['icon']
            })
            
        sorted_gains = sorted(factor_gains, key=lambda x: x['predicted_gain'], reverse=True)
        best_focus = sorted_gains[0] if sorted_gains else None
        
        return {
            'baseline_score': baseline_score,
            'factor_gains': sorted_gains,
            'best_focus': best_focus
        }

    def generate_personalized_plan(self, current_inputs: dict) -> dict:
        """
        Generate customized academic improvement plan based on individual student Kaggle inputs.
        """
        best_analysis = self.analyze_best_improvement(current_inputs)
        baseline_res = self.predictor.predict(current_inputs)
        predicted_score = baseline_res['predicted_marks']
        risk_level = baseline_res['risk_level']
        
        priorities = []
        benchmarks = {
            'Hours_Studied': (25.0, "Weekly Study Hours"),
            'Attendance': (85.0, "Attendance (%)"),
            'Tutoring_Sessions': (3.0, "Tutoring Sessions"),
            'Sleep_Hours': (7.5, "Sleep Duration (hrs)")
        }
        
        for fg in best_analysis['factor_gains']:
            factor = fg['factor']
            curr = fg['current_value']
            if factor in benchmarks:
                target_bm, label = benchmarks[factor]
                gain = fg['predicted_gain']
                if curr < target_bm:
                    urgency = (target_bm - curr) / target_bm + (gain / 10.0)
                    priorities.append({
                        'factor': factor,
                        'name': fg['name'],
                        'current_val': curr,
                        'suggested_target': target_bm,
                        'unit': fg['unit'],
                        'potential_gain': gain,
                        'urgency': urgency,
                        'suggestions': SUGGESTIONS_DB.get(factor, [])
                    })
                    
        priorities = sorted(priorities, key=lambda x: x['urgency'], reverse=True)
        
        if not priorities:
            for fg in best_analysis['factor_gains'][:2]:
                factor = fg['factor']
                priorities.append({
                    'factor': factor,
                    'name': fg['name'],
                    'current_val': fg['current_value'],
                    'suggested_target': min(fg['current_value'] + 5.0, CONTROLLABLE_FACTORS[factor]['max']),
                    'unit': fg['unit'],
                    'potential_gain': fg['predicted_gain'],
                    'urgency': 1.0,
                    'suggestions': SUGGESTIONS_DB.get(factor, [])
                })

        return {
            'predicted_score': predicted_score,
            'risk_level': risk_level,
            'best_focus_factor': best_analysis['best_focus'],
            'priority_action_items': priorities,
            'overall_strategy': self._build_strategy_summary(predicted_score, risk_level, best_analysis['best_focus'])
        }

    def _build_strategy_summary(self, score: float, risk: str, best_focus: dict) -> str:
        if risk == "HIGH RISK":
            return f"Immediate academic intervention required. Primary focus should be on **{best_focus['name']}** to secure an estimated +{best_focus['predicted_gain']} exam score lift."
        elif risk == "MODERATE RISK":
            return f"Targeted optimization required. Concentrating on **{best_focus['name']}** yields the highest predicted return (+{best_focus['predicted_gain']} marks)."
        else:
            return f"High performance student profile. Maintain academic habits while boosting **{best_focus['name']}** for peak exam performance."
