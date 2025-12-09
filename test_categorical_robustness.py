"""
Test to compare different approaches for handling categorical features in robustness loss
"""
import torch
import torch.nn.functional as F


def test_approaches():
    print('='*70)
    print('Comparing approaches for categorical features in Sørensen-Dice')
    print('='*70)
    
    # Scenario: Original CF has category 0, perturbed CF has category 1
    original_cf = torch.tensor([[1.0, 0.0, 0.0]])
    perturbed_cf = torch.tensor([[0.0, 1.0, 0.0]])
    
    print('\nScenario: Category changed from 0 to 1')
    print(f'Original:  {original_cf}')
    print(f'Perturbed: {perturbed_cf}')
    
    # Approach 1: With Softmax (PROBLEMATIC)
    tau = 1.0
    orig_soft = F.softmax(original_cf / tau, dim=1)
    pert_soft = F.softmax(perturbed_cf / tau, dim=1)
    
    intersection_soft = torch.sum(torch.min(orig_soft, pert_soft))
    union_soft = torch.sum(orig_soft) + torch.sum(pert_soft)
    dice_soft = (2 * intersection_soft) / (union_soft + 1e-8)
    
    print(f'\nApproach 1: WITH Softmax')
    print(f'  Original softmax:  {orig_soft}')
    print(f'  Perturbed softmax: {pert_soft}')
    print(f'  Sørensen-Dice coefficient: {dice_soft:.4f}')
    print(f'  → Problem: High similarity ({dice_soft:.4f}) despite different categories!')
    
    # Approach 2: Without Softmax (BETTER)
    intersection_raw = torch.sum(torch.min(original_cf, perturbed_cf))
    union_raw = torch.sum(original_cf) + torch.sum(perturbed_cf)
    dice_raw = (2 * intersection_raw) / (union_raw + 1e-8)
    
    print(f'\nApproach 2: WITHOUT Softmax (keeping one-hot as-is)')
    print(f'  Original:  {original_cf}')
    print(f'  Perturbed: {perturbed_cf}')
    print(f'  Sørensen-Dice coefficient: {dice_raw:.4f}')
    print(f'  → Better: Low similarity ({dice_raw:.4f}) correctly reflects different categories!')
    
    print('\n' + '='*70)
    print('Scenario: Same category (both category 1)')
    print('='*70)
    
    same_orig = torch.tensor([[0.0, 1.0, 0.0]])
    same_pert = torch.tensor([[0.0, 1.0, 0.0]])
    
    print(f'\nOriginal:  {same_orig}')
    print(f'Perturbed: {same_pert}')
    
    # With softmax
    same_orig_soft = F.softmax(same_orig / tau, dim=1)
    same_pert_soft = F.softmax(same_pert / tau, dim=1)
    intersection_same_soft = torch.sum(torch.min(same_orig_soft, same_pert_soft))
    union_same_soft = torch.sum(same_orig_soft) + torch.sum(same_pert_soft)
    dice_same_soft = (2 * intersection_same_soft) / (union_same_soft + 1e-8)
    
    # Without softmax
    intersection_same_raw = torch.sum(torch.min(same_orig, same_pert))
    union_same_raw = torch.sum(same_orig) + torch.sum(same_pert)
    dice_same_raw = (2 * intersection_same_raw) / (union_same_raw + 1e-8)
    
    print(f'\nWith Softmax:    Dice = {dice_same_soft:.4f}')
    print(f'Without Softmax: Dice = {dice_same_raw:.4f}')
    print(f'→ Both correctly show perfect similarity (1.0)')
    
    print('\n' + '='*70)
    print('Scenario: Soft one-hot (during optimization)')
    print('='*70)
    
    # During gradient descent, one-hot vectors naturally become "soft"
    soft_orig = torch.tensor([[0.85, 0.10, 0.05]])  # Mostly category 0
    soft_pert = torch.tensor([[0.10, 0.80, 0.10]])  # Mostly category 1
    
    print(f'\nOriginal (soft):  {soft_orig}')
    print(f'Perturbed (soft): {soft_pert}')
    
    # With softmax
    soft_orig_softmax = F.softmax(soft_orig / tau, dim=1)
    soft_pert_softmax = F.softmax(soft_pert / tau, dim=1)
    intersection_soft_soft = torch.sum(torch.min(soft_orig_softmax, soft_pert_softmax))
    union_soft_soft = torch.sum(soft_orig_softmax) + torch.sum(soft_pert_softmax)
    dice_soft_soft = (2 * intersection_soft_soft) / (union_soft_soft + 1e-8)
    
    # Without softmax
    intersection_soft_raw = torch.sum(torch.min(soft_orig, soft_pert))
    union_soft_raw = torch.sum(soft_orig) + torch.sum(soft_pert)
    dice_soft_raw = (2 * intersection_soft_raw) / (union_soft_raw + 1e-8)
    
    print(f'\nWith Softmax:')
    print(f'  After softmax: {soft_orig_softmax}, {soft_pert_softmax}')
    print(f'  Dice = {dice_soft_soft:.4f}')
    print(f'\nWithout Softmax:')
    print(f'  As-is: {soft_orig}, {soft_pert}')
    print(f'  Dice = {dice_soft_raw:.4f}')
    print(f'\n→ Without softmax gives better discrimination ({dice_soft_raw:.4f} vs {dice_soft_soft:.4f})')
    
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print('\n✓ WITHOUT Softmax (Recommended):')
    print('  - Preserves categorical distinctiveness')
    print('  - Hard one-hot: [1,0,0] vs [0,1,0] → Dice = 0.0 (totally different)')
    print('  - Soft one-hot naturally emerges during optimization')
    print('  - Better gradient signal for categorical changes')
    print('\n✗ WITH Softmax:')
    print('  - Makes all categories artificially similar')
    print('  - Hard one-hot: [1,0,0] → [0.576, 0.212, 0.212] (loses information)')
    print('  - Weaker discrimination between different categories')
    print('  - Defeats the purpose of measuring categorical differences')


if __name__ == '__main__':
    test_approaches()
