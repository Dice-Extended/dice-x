"""
Random perturbation implementation for counterfactual instances.
"""

from dice_ml_x.perturbation_interfaces.base_perturbation import _BasePerturbation
import pandas as pd
import numpy as np


class RandomPerturbation(_BasePerturbation):
    """
    Implements random perturbation strategy for counterfactual instances.

    Perturbs the continuous features randomly within a given range and modifies
    the categorical features.
    """
    def generate(self, c_i: pd.DataFrame, continuous_features: list = [],
                  categorical_features: dict = {}, feature_ranges: dict = {}) -> pd.DataFrame:
        """
        Generates random perturbations for both continuous and categorical features.

        Args:
            c_i (pandas.DataFrame): The counterfactual instance to be perturbed.
            continuous_features (list): List of continuous features.
            categorical_features (dict): Categorical features with their possible
                values.
            feature_ranges (dict): Ranges for continuous features as {feature: (min, max)}.
        """
        c_i_prime = c_i.copy()

        for feature in continuous_features:
            if feature in c_i.columns:
                low, high = feature_ranges.get(feature, (0, 1))
                c_i_prime[feature] = np.random.uniform(low, high)

        
        for cat_feature, cats in categorical_features.items():
            if cat_feature in c_i.columns:
                c_i_prime[cat_feature] = np.random.choice(cats)

        return c_i_prime
    

    def validate(self, c_i: pd.DataFrame, c_i_prime: pd.DataFrame, model: any) -> bool:
        """
        Validates that the model outcomes the same output both for c_i the
        counterfactual instance and c_i_prime the perturbed counterfactual.

        Args:
            c_i (pandas.DataFrame): The original counterfactual instance.
            c_i_prime (pandas.DataFrame): The perturbed counterfactual instance.
            model (any): The model to validate against.
        Returns:
            bool: Boolean that indicates the validity of the perturbed counterfactual.
        """
        return model.predict(c_i) == model.predict(c_i_prime)