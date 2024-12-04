"""
Spherical perturbation implementation.
"""

from dice_ml_x.perturbation_interfaces.base_perturbation import _BasePerturbation
import pandas as pd
import numpy as np

class SphericalPerturbation(_BasePerturbation):
    """
    Implements spherical perturbation for counterfactual instances.

    Generates perturbations within a spherical boundary constructed around the
    given counterfactual instance.
    """
    def generate(self, c_i: pd.DataFrame, radius: float = 1.0, continuous_features: list = [],
                 feature_ranges: dict = {}) -> pd.DataFrame:
        """
        Generates perturbations within a spherical boundary around the given counterfactual instance.

        Args:
            c_i (pandas.DataFrame): The counterfactual instance that will be perturbed.
            radius (float): Radius of the sphere that will be constructed around the 
                given counterfactual instance.
            continuous_features (list): List of continuous features.
        Returns:
            pandas.DataFrame: A perturbed version of the given counterfactual explanation.
        """
        c_i_prime = c_i.copy()

        for feature in continuous_features:
            if feature in c_i.columns:
                current_value = c_i[feature].values[0]
                feature_min = max(feature_ranges[feature][0], 0, current_value)
                feature_max = feature_ranges[feature][1]
                scaled_radius = radius * (feature_max - feature_min)
                low = max(current_value - scaled_radius, feature_min)
                high = min(current_value - scaled_radius, feature_max)
                c_i_prime[feature] = np.random.uniform(low, high)

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