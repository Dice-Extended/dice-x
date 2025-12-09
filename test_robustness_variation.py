#!/usr/bin/env python3
"""
Test that robustness loss now varies properly across iterations.
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

print("Testing robustness loss variation...")
print("=" * 60)

# Generate counterfactuals with robustness enabled
test_instance = data.iloc[:1][d.feature_names]
print(f"\nQuery instance:")
print(test_instance)

# Generate CFs with robustness weight
e1 = exp.generate_counterfactuals(
    test_instance,
    total_CFs=5,
    perturbation_method="gaussian",
    desired_class="opposite",
    std_dev=0.30  # Strong perturbations
)

print("\nCounterfactuals found:")
print(e1.visualize_as_dataframe(show_only_changes=True))

# Check robustness loss history
if exp.loss_history and 'robustness_loss' in exp.loss_history:
    robustness_values = exp.loss_history['robustness_loss']
    print(f"\n\nRobustness loss across iterations:")
    print(f"  Values: {robustness_values}")
    print(f"  Min: {min(robustness_values):.4f}")
    print(f"  Max: {max(robustness_values):.4f}")
    print(f"  Mean: {np.mean(robustness_values):.4f}")
    print(f"  Std: {np.std(robustness_values):.4f}")
    
    # Check if it varies
    if len(robustness_values) > 1:
        variation = max(robustness_values) - min(robustness_values)
        if variation > 0.05:
            print(f"\n✓ SUCCESS: Robustness loss varies significantly (range={variation:.4f})")
        else:
            print(f"\n✗ WARNING: Robustness loss barely varies (range={variation:.4f})")
    
    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(robustness_values, marker='o', label='Robustness Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Robustness (Sørensen-Dice)')
        plt.title('Robustness Loss Over Iterations')
        plt.grid(True)
        plt.legend()
        plt.savefig('robustness_variation_test.png')
        print(f"\nPlot saved to: robustness_variation_test.png")
    except ImportError:
        print("\nMatplotlib not available for plotting")
else:
    print("\n✗ ERROR: No robustness loss history found!")

print("\n" + "=" * 60)
