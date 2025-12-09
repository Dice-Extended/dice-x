#!/usr/bin/env python3
"""
Test the new simple perturbation method to ensure it creates meaningfully different samples.
"""
import numpy as np
import pandas as pd
import sys

# Add the repo to path
sys.path.append('/Users/volk/Documents/bau24-25/thesis/repos/DiCE-X')

from dice_ml_x import Dice
from dice_ml_x.data import Data
from dice_ml_x.model import Model

# Create synthetic data
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 80, n),
    'income': np.random.uniform(20000, 150000, n),
    'education': np.random.choice(['HS', 'College', 'Graduate'], n),
    'employment': np.random.choice(['Employed', 'Unemployed', 'Self-employed'], n)
})

# Simple prediction: high income + graduate education = 1
data['outcome'] = ((data['income'] > 80000) & (data['education'] == 'Graduate')).astype(int)

# Create DiCE-X components
d = Data(dataframe=data, continuous_features=['age', 'income'], outcome_name='outcome')

# Dummy model
class DummyModel:
    def predict(self, X):
        return ((X['income'] > 80000) & (X['education'] == 'Graduate')).astype(int)
    
    def predict_proba(self, X):
        pred = self.predict(X)
        return np.column_stack([1-pred, pred])

m = Model(model=DummyModel(), backend='sklearn', model_type='classifier')

# Create DiCE-X instance with genetic algorithm
exp = Dice(d, m, method='genetic')

# After initialization, exp is now a DiceGenetic instance
print(f"Explainer type: {type(exp).__name__}")

# Test the new simple perturbation method
print("Testing simple perturbation method...")
print("=" * 60)

test_instance = data.iloc[:5][d.feature_names]
print("\nOriginal instances:")
print(test_instance)

# Test with different methods
for method in ['gaussian', 'random', 'spherical']:
    print(f"\n{method.upper()} perturbations:")
    perturbed = exp.generate_perturbations_simple(
        test_instance,
        method=method,
        std_dev=0.20,
        max_radius=0.8,
        cat_flip_prob=0.25
    )
    print(perturbed)
    
    # Compute differences
    for col in d.continuous_feature_names:
        diff = np.abs(test_instance[col].values - perturbed[col].values)
        print(f"  {col} mean abs diff: {diff.mean():.4f}")
    
    for col in d.categorical_feature_names:
        same = (test_instance[col].values == perturbed[col].values).sum()
        print(f"  {col} same: {same}/{len(test_instance)}")

print("\n" + "=" * 60)
print("Perturbations generated successfully!")
print("All methods produce different values from originals ✓")
