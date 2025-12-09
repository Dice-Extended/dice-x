#!/usr/bin/env python3
"""
Diagnostic tool to understand why robustness values are so high.
"""
import numpy as np
import pandas as pd
import sys

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

data['outcome'] = ((data['income'] > 80000) & (data['education'] == 'Graduate')).astype(int)

d = Data(dataframe=data, continuous_features=['age', 'income'], outcome_name='outcome')


class DummyModel:
    def predict(self, X):
        return ((X['income'] > 80000) & (X['education'] == 'Graduate')).astype(int)
    
    def predict_proba(self, X):
        pred = self.predict(X)
        return np.column_stack([1-pred, pred])


m = Model(model=DummyModel(), backend='sklearn', model_type='classifier')
exp = Dice(d, m, method='genetic')

print("Diagnostic: Understanding Robustness Values")
print("=" * 70)

# Generate a CF
test_instance = data.iloc[:1]
query_df = test_instance[d.continuous_feature_names + d.categorical_feature_names]
print("\nQuery instance:")
print(query_df)

# Generate one CF
e1 = exp.generate_counterfactuals(
    query_df,
    total_CFs=1,
    perturbation_method="gaussian",
    desired_class="opposite",
    std_dev=0.30
)

# Get the generated CF
if exp.final_cfs_df is not None and len(exp.final_cfs_df) > 0:
    cf = exp.final_cfs_df.iloc[0:1]
    print("\nGenerated Counterfactual:")
    print(cf)
    
    # Manually generate and test perturbations
    print("\n" + "=" * 70)
    print("Testing perturbations manually:")
    
    cf_for_pert = cf[d.continuous_feature_names + d.categorical_feature_names]
    
    for strength in [0.1, 0.2, 0.3, 0.5]:
        print(f"\nPerturbation strength (std_dev): {strength}")
        perturbed = exp.generate_perturbations_simple(
            cf_for_pert,
            method="gaussian",
            std_dev=strength,
            max_radius=strength * 2,
            cat_flip_prob=0.25
        )
        
        print("  Original CF:")
        print(f"    {cf_for_pert.to_dict('records')[0]}")
        print("  Perturbed CF:")
        print(f"    {perturbed.to_dict('records')[0]}")
        
        # Compute differences
        for col in d.continuous_feature_names:
            orig = cf_for_pert[col].values[0]
            pert = perturbed[col].values[0]
            diff = abs(orig - pert)
            pct = (diff / (orig + 1e-6)) * 100
            print(f"    {col}: {orig:.2f} → {pert:.2f} (Δ={diff:.2f}, {pct:.1f}%)")
        
        for col in d.categorical_feature_names:
            orig = cf_for_pert[col].values[0]
            pert = perturbed[col].values[0]
            same = "✓ same" if orig == pert else "✗ CHANGED"
            print(f"    {col}: {orig} → {pert} ({same})")
        
        # Compute robustness using the same method as the loss
        # We need to encode the data first
        cf_encoded = exp.label_encode(cf_for_pert.copy())
        pert_encoded = exp.label_encode(perturbed.copy())
        
        robustness_array = exp.compute_robustness_loss_SDS(
            cf_encoded.values,
            perturbed
        )
        robustness = robustness_array[0] if isinstance(robustness_array, np.ndarray) else robustness_array
        
        print(f"  → Sørensen-Dice Coefficient: {robustness:.4f}")
        print(f"     (1.0 = identical, 0.0 = completely different)")

else:
    print("\n✗ ERROR: No counterfactuals generated!")

print("\n" + "=" * 70)
