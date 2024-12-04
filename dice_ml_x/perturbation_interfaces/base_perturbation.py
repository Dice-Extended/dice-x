"""
Abstract base class for various perturbation methods.

`_BasePerturbation` class is defined in the module as a common interface
for implementing various perturbation strategies.
"""

from abc import abstractmethod, ABC
import pandas as pd

class _BasePerturbation(ABC):
    """
    Abstract base class for different perturbation methods
    
    All strategies that will inherit this class is enforced to implement
    `generate` and `validate` methods.
    """

    @abstractmethod
    def generate(self, c_i: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generates perturbations for the given counterfactual c_i

        Args:
            c_i (pd.DataFrame): The counterfactual instance that will be perturbed.
            **kwargs: Additional arguments that will be used to specify the 
                strategy to generate perturbations.
        
        Returns: 
            pandas.DataFrame: The perturbed version of the given counterfactual explanation.
        """
        pass


    @abstractmethod
    def validate(self, c_i: pd.DataFrame, c_i_prime: pd.DataFrame, predict_fn: callable) -> bool:
        """
        Validates that the model outcomes the same output both for c_i the
        counterfactual instance and c_i_prime the perturbed counterfactual.

        Args:
            c_i (pandas.DataFrame): The original counterfactual instance.
            c_i_prime (pandas.DataFrame): The perturbed counterfactual instance
            predict_fn (callable): The prediction method of the model.
    
        Returns:
            bool: Boolean that indicates the validity of the perturbed counterfactual.
        """
        pass

    def reconstruct_categorical(self, c_i: pd.DataFrame, **kwargs):
        """
        Converts one-hot encoded columns to original categorical format for perturbation.

        Args:
            c_i (pandas.DataFrame): The counterfactual instance that will be perturbed.
            **kwargs (dict): Additional arguments such as categorical features with their
                possible values.
            
        Returns:
            pandas.DataFrame: A DataFrame with original categorical values.
        """
        categorical_features = kwargs.get('categorical_features')
        c_i_prime = c_i.copy()
        for feature, categories in categorical_features.items():
            one_hot_cols = [f"{feature}_{cat}" for cat in categories]
            for col in one_hot_cols:
                if col in c_i_prime.columns and c_i_prime[col].iloc[0] == 1:
                    c_i_prime[feature] = col.split(f"{feature}_")[-1]
                    break
        c_i_prime = c_i_prime.drop(columns=[col for feature in categorical_features for col in [f"{feature}_{cat}" for cat in categorical_features[feature]]], errors='ignore')
    
        return c_i_prime

    def expand_to_onehot(self, c_i: pd.DataFrame, **kwargs):
        """
        Converts perturbed categorical features back to one-hot encoded format.

        Args:
            c_i_prime (pandas.DataFrame): Perturbed counterfactual instance.
            **kwargs (dict): Additional arguments such as categorical features with their
                possible values.

        Returns:
            pandas.DataFrame: A DataFrame with one hot encoded categorical values.
        """
        categorical_features = kwargs.get('categorical_features')
        c_i_onehot = c_i.copy()
        for feature, categories in categorical_features.items():
            for category in categories:
                c_i_onehot[f"{feature}_{category}"] = (c_i_onehot[feature] == category).astype(int)
        return c_i_onehot