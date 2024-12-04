"""
Gaussian perturbation implementation.
"""

from dice_ml_x.perturbation_interfaces.base_perturbation import _BasePerturbation
import numpy as np
import pandas as pd

class GaussianPerturbation(_BasePerturbation):
    """
    Implements Gaussian perturbation strategy for counterfactual instances.
    
    Generates perturbations by adding noise to continuous features and randomly
    changes the categorical features.
    """

    def __init__(self, std_dev=0.1, continuous_features: list = [], categorical_features: dict = {}) -> None:
        """
        Initializes the GaussianPerturbation with features given.

        Args:
            std_dev (float): Standard deviation for Gaussian noise that will be added
                to the continuous features.
            continuous_features (list): List of continuous feature names.
            categorical_features (dict): Dictionary of categorical features and their possible values.

        Returns:
            None
        """
        self.std_dev = std_dev
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features

    def generate(self, c_i: pd.DataFrame, std_dev=0.1) -> pd.DataFrame:
        """
        Generates perturbations for continuous and categorical features.

        Args:
            c_i (pandas.DataFrame): The counterfactual instance that will be perturbed.
            std_dev (float): Standard deviation for Gaussian noise that will be added
                to the continuous features.

        Returns:
            pandas.DataFrame: A perturbed version of the given counterfactual explanation.
        """
        
        c_i_prime = c_i.copy().reset_index(drop=True)
        # Handle the continuous features
        for feature in self.continuous_features:
            if feature in c_i.columns:
                c_i_prime[feature] += np.random.normal(0, std_dev)

        # Handle the categorical features
        for cat_feature, cats in self.categorical_features.items():
            if cat_feature in c_i.columns:
                current_val = c_i[cat_feature].values[0]
                valid_cats = [cat for cat in cats if cat != current_val]
                if valid_cats:
                    c_i_prime.at[0, cat_feature] = np.random.choice(valid_cats)

        c_i_prime = c_i_prime.iloc[:1].reset_index(drop=True)

        return c_i_prime
    
    def validate(self, c_i: pd.DataFrame, c_i_prime: pd.DataFrame, predict_fn: callable) -> bool:
        """
        Validates that the model outcomes the same output both for c_i the
        counterfactual instance and c_i_prime the perturbed counterfactual.

        Args:
            c_i (pandas.DataFrame): The original counterfactual instance.
            c_i_prime (pandas.DataFrame): The perturbed counterfactual instance.
            predict_fn (callable): The prediction function of the model.
        Returns:
            bool: Boolean that indicates the validity of the perturbed counterfactual.
        """
        try:

            # Predictions
            pred_c_i = predict_fn(c_i)

            pred_c_i_prime = predict_fn(c_i_prime)
            return pred_c_i == pred_c_i_prime
        except KeyError as e:
            print(f"KeyError occurred: {e}, for c_i:\n {c_i.head()}\n, c_i_prime:\n {c_i_prime.head()}")
            raise
        """if len(pred_c_i.shape) > 1 and pred_c_i.shape[1] > 1:
            print("validate if")
            return np.argmax(pred_c_i) == np.argmax(pred_c_i_prime)
        else:
            print("validate else")
            return pred_c_i == pred_c_i_prime"""