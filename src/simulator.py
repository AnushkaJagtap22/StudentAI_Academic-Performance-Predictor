"""
What-If Performance Simulator Engine for Kaggle Student Performance Factors.
Evaluates counterfactual scenarios dynamically using the trained ML model.
"""

from prediction import StudentPredictor

class WhatIfSimulator:
    """
    Evaluates scenario changes in Kaggle student inputs and quantifies predicted exam score improvements.
    """
    def __init__(self, predictor: StudentPredictor = None):
        self.predictor = predictor if predictor is not None else StudentPredictor()

    def simulate_scenario(self, baseline_inputs: dict, scenario_inputs: dict) -> dict:
        """
        Predict baseline performance and scenario performance using the trained Kaggle ML model.
        """
        baseline_res = self.predictor.predict(baseline_inputs)
        scenario_res = self.predictor.predict(scenario_inputs)
        
        baseline_score = baseline_res['predicted_marks']
        scenario_score = scenario_res['predicted_marks']
        mark_delta = round(scenario_score - baseline_score, 1)
        
        factor_diffs = {}
        for col in self.predictor.preprocessor.feature_cols:
            base_val = baseline_inputs.get(col, None)
            scen_val = scenario_inputs.get(col, None)
            factor_diffs[col] = {
                'baseline': base_val,
                'scenario': scen_val
            }
            
        return {
            'baseline': baseline_res,
            'scenario': scenario_res,
            'baseline_score': baseline_score,
            'scenario_score': scenario_score,
            'mark_delta': mark_delta,
            'is_improvement': mark_delta >= 0,
            'factor_diffs': factor_diffs
        }
