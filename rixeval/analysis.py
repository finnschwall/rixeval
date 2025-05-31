import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional
import krippendorff


def calculate_krippendorff_alpha(df):
    # Create a matrix where rows are datapoints and columns are raters (variations)
    datapoints = df['datapoint_idx'].unique()
    max_variations = df['datapoint_idx'].value_counts().max()

    # Create reliability data matrix
    reliability_data = []

    for datapoint_idx in datapoints:
        decisions = df[df['datapoint_idx'] == datapoint_idx]['llm_decision'].tolist()
        # Convert to numeric (Check-Out = 1, Cancellation = 0)
        numeric_decisions = [1 if d == 'Check-Out' else 0 for d in decisions]

        # Pad with NaN if fewer variations than max
        while len(numeric_decisions) < max_variations:
            numeric_decisions.append(np.nan)

        reliability_data.append(numeric_decisions)

    # Transpose for krippendorff format (raters as rows, items as columns)
    reliability_data = np.array(reliability_data).T

    # Calculate Krippendorff's Alpha
    alpha = krippendorff.alpha(reliability_data, level_of_measurement='nominal')
    return alpha

class ResultAnalyzer:
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df.copy()
        self._encode_outcomes()

    def _encode_outcomes(self):
        outcome_mapping = {'Check-Out': 1, 'Cancellation': 0}
        self.df['true_outcome_encoded'] = self.df['true_outcome'].map(outcome_mapping)
        self.df['predicted_outcome_encoded'] = self.df['predicted_outcome'].map(outcome_mapping)
        self.df['llm_decision_encoded'] = self.df['llm_decision'].map(outcome_mapping)

    def analyze_combination(self, combination_id: int) -> Dict[str, float]:
        combo_data = self.df[self.df['combination_id'] == combination_id]

        if len(combo_data) == 0:
            return {}

        num_correct = len(combo_data[combo_data['true_outcome'] == combo_data['llm_decision']])
        percent_correct = (num_correct / len(combo_data)) * 100

        num_agree_ai = len(combo_data[combo_data['predicted_outcome'] == combo_data['llm_decision']])
        percent_agree_ai = (num_agree_ai / len(combo_data)) * 100

        # Correlations
        corr_true_llm = combo_data['true_outcome_encoded'].corr(combo_data['llm_decision_encoded'])
        corr_predicted_llm = combo_data['predicted_outcome_encoded'].corr(combo_data['llm_decision_encoded'])

        # Decision distribution
        decision_counts = combo_data['llm_decision'].value_counts(normalize=True)
        decision_entropy = -sum(p * np.log2(p) for p in decision_counts if p > 0)

        return {
            "combination_id": combination_id,
            # "n_datapoints": len(combo_data),
            "percent_correct": percent_correct,
            "percent_agree_ai": percent_agree_ai,
            "corr_true_llm": corr_true_llm,
            "corr_predicted_llm": corr_predicted_llm,
            # "decision_entropy": decision_entropy,
            "avg_processing_time": combo_data['processing_time'].mean()
        }

    def analyze_all_combinations(self) -> pd.DataFrame:
        results = []
        for combo_id in self.df['combination_id'].unique():
            analysis = self.analyze_combination(combo_id)
            if analysis:
                # Add parameter information
                combo_params = self.df[self.df['combination_id'] == combo_id].iloc[0]
                param_cols = [col for col in self.df.columns if col not in [
                    'experiment_id', 'combination_id', 'datapoint_idx', 'dataset_index',
                    'confidence', 'true_outcome', 'predicted_outcome', 'llm_decision',
                    'reasoning_length', 'processing_time', 'timestamp',
                    'true_outcome_encoded', 'predicted_outcome_encoded', 'llm_decision_encoded', "tokens_per_second", "total_tokens"
                ]]

                for col in param_cols:
                    analysis[col] = combo_params[col]
                results.append(analysis)

        return pd.DataFrame(results).drop("combination_id", axis=1)

    def filter_by_parameter(self, param_name: str, param_value: Any) -> 'ResultAnalyzer':
        filtered_df = self.df[self.df[param_name] == param_value]
        return ResultAnalyzer(filtered_df)

    def compare_parameters(self, param_name: str) -> pd.DataFrame:
        # Compare results across different values of a parameter
        comparison_results = []

        for param_value in self.df[param_name].unique():
            subset = self.df[self.df[param_name] == param_value]

            if len(subset) == 0:
                continue

            analyzer = ResultAnalyzer(subset)
            summary = analyzer.analyze_all_combinations()

            comparison_results.append({
                param_name: param_value,
                "n_combinations": len(summary),
                "avg_percent_correct": summary['percent_correct'].mean(),
                "std_dev_percent_correct": summary['percent_correct'].std(),
                "avg_percent_agree_ai": summary['percent_agree_ai'].mean(),
                "std_dev_percent_agree_ai": summary['percent_agree_ai'].std(),
                "krippendorff_alpha": calculate_krippendorff_alpha(subset),
                "avg_corr_true_llm": summary['corr_true_llm'].mean(),
                "std_dev_corr_true_llm": summary['corr_true_llm'].std(),
                "avg_corr_predicted_llm": summary['corr_predicted_llm'].mean(),
                "std_dev_corr_predicted_llm": summary['corr_predicted_llm'].std(),
                # "avg_decision_entropy": summary['decision_entropy'].mean()
            })

        return pd.DataFrame(comparison_results)