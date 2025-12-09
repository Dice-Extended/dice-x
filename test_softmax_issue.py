"""
Test to demonstrate the problem with softmax on one-hot encoded features
"""
import torch
import torch.nn.functional as F

print('Problem with softmax on one-hot encoded features:')
print('='*70)

# Original one-hot vectors (different categories)
cat1 = torch.tensor([[1.0, 0.0, 0.0]])  # Category 0
cat2 = torch.tensor([[0.0, 1.0, 0.0]])  # Category 1
cat3 = torch.tensor([[0.0, 0.0, 1.0]])  # Category 2

print('\nOriginal one-hot vectors:')
print(f'Category 0: {cat1}')
print(f'Category 1: {cat2}')
print(f'Category 2: {cat3}')

# Apply softmax with tau=1.0
tau = 1.0
soft1 = F.softmax(cat1 / tau, dim=1)
soft2 = F.softmax(cat2 / tau, dim=1)
soft3 = F.softmax(cat3 / tau, dim=1)

print(f'\nAfter softmax (tau={tau}):')
print(f'Category 0: {soft1}')
print(f'Category 1: {soft2}')
print(f'Category 2: {soft3}')

# Compute differences
diff_01 = torch.abs(soft1 - soft2).sum()
diff_12 = torch.abs(soft2 - soft3).sum()
diff_02 = torch.abs(soft1 - soft3).sum()

print(f'\nL1 differences after softmax:')
print(f'Cat0 vs Cat1: {diff_01:.4f}')
print(f'Cat1 vs Cat2: {diff_12:.4f}')
print(f'Cat0 vs Cat2: {diff_02:.4f}')

print(f'\nOriginal L1 differences (on one-hot):')
print(f'Cat0 vs Cat1: {torch.abs(cat1 - cat2).sum():.4f}')
print(f'Cat1 vs Cat2: {torch.abs(cat2 - cat3).sum():.4f}')
print(f'Cat0 vs Cat2: {torch.abs(cat1 - cat3).sum():.4f}')

print('\n' + '='*70)
print('Issue: Softmax makes ALL categories look similar!')
print('All pairwise distances become ~0.36, losing categorical distinction.')

print('\n' + '='*70)
print('Testing with "soft" one-hot vectors (more realistic during optimization):')
print('='*70)

# During optimization, vectors might be "soft" one-hot
soft_cat1 = torch.tensor([[0.9, 0.05, 0.05]])   # Mostly category 0
soft_cat2 = torch.tensor([[0.05, 0.9, 0.05]])   # Mostly category 1
soft_cat3 = torch.tensor([[0.33, 0.34, 0.33]])  # Uncertain

print('\nSoft one-hot vectors:')
print(f'Soft Cat 0: {soft_cat1}')
print(f'Soft Cat 1: {soft_cat2}')
print(f'Uncertain:  {soft_cat3}')

soft1_softmax = F.softmax(soft_cat1 / tau, dim=1)
soft2_softmax = F.softmax(soft_cat2 / tau, dim=1)
soft3_softmax = F.softmax(soft_cat3 / tau, dim=1)

print(f'\nAfter softmax:')
print(f'Soft Cat 0: {soft1_softmax}')
print(f'Soft Cat 1: {soft2_softmax}')
print(f'Uncertain:  {soft3_softmax}')

print(f'\nL1 differences:')
print(f'Soft0 vs Soft1: {torch.abs(soft1_softmax - soft2_softmax).sum():.4f}')
print(f'Soft0 vs Uncertain: {torch.abs(soft1_softmax - soft3_softmax).sum():.4f}')

print('\n' + '='*70)
print('CONCLUSION:')
print('='*70)
print('✗ Softmax on one-hot vectors loses categorical information')
print('✗ All categories become similar probability distributions')
print('✗ This defeats the purpose of measuring categorical differences')
print('\nBetter alternatives:')
print('1. Keep categorical features as-is (already differentiable)')
print('2. Use Gumbel-Softmax for sampling but compare pre-softmax values')
print('3. Use straight-through estimator')
print('4. Compare argmax indices with a differentiable approximation')
