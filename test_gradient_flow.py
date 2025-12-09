"""
Test script to verify gradient flow in perturbation generation process
"""
import torch
import torch.nn as nn


def test_gradient_flow():
    """Test if gradients flow correctly through perturbation generation"""
    
    print("Testing gradient flow in perturbation generation...\n")
    
    # Simulate counterfactual tensors (like self.cfs)
    cfs = []
    for i in range(3):
        cf = torch.randn(10, requires_grad=True)
        cfs.append(cf)
    
    print("1. Initial CFs created with requires_grad=True")
    print(f"   CF[0] requires_grad: {cfs[0].requires_grad}")
    
    # Test 1: Stack and check gradient flow
    cfs_stacked = torch.stack(cfs, dim=0)
    print(f"\n2. After stacking, requires_grad: {cfs_stacked.requires_grad}")
    
    # Test 2: Clone and check gradient flow
    cfs_cloned = cfs_stacked.clone()
    print(f"3. After cloning, requires_grad: {cfs_cloned.requires_grad}")
    
    # Test 3: Add noise and check gradient flow
    noise = cfs_cloned * 0.05
    cfs_perturbed = cfs_cloned + noise
    print(f"4. After adding noise, requires_grad: {cfs_perturbed.requires_grad}")
    
    # Test 4: Convert to Parameter
    cfs_param = nn.Parameter(cfs_perturbed, requires_grad=True)
    print(f"5. After converting to Parameter, requires_grad: {cfs_param.requires_grad}")
    
    # Test 5: Compute loss and backward
    print("\n6. Testing backward pass...")
    loss = cfs_param.sum()
    loss.backward()
    
    # Check if gradients reached original CFs
    has_gradients = all(cf.grad is not None for cf in cfs)
    print(f"   Original CFs have gradients: {has_gradients}")
    if has_gradients:
        print(f"   CF[0] grad shape: {cfs[0].grad.shape}")
        print(f"   CF[0] grad norm: {cfs[0].grad.norm().item():.6f}")
    
    print("\n" + "="*60)
    
    # Test 6: Simulate the BROKEN version (with detach)
    print("\nTesting BROKEN version (with detach):")
    cfs_broken = []
    for i in range(3):
        cf = torch.randn(10, requires_grad=True)
        cfs_broken.append(cf)
    
    cfs_stacked_broken = torch.stack(cfs_broken, dim=0)
    cfs_perturbed_broken = cfs_stacked_broken + cfs_stacked_broken * 0.05
    cfs_param_broken = nn.Parameter(cfs_perturbed_broken.detach(), requires_grad=True)  # DETACH HERE
    
    print(f"   Parameter requires_grad: {cfs_param_broken.requires_grad}")
    
    loss_broken = cfs_param_broken.sum()
    loss_broken.backward()
    
    has_gradients_broken = all(cf.grad is not None for cf in cfs_broken)
    print(f"   Original CFs have gradients: {has_gradients_broken}")
    
    print("\n" + "="*60)
    
    # Test 7: Simulate with torch.no_grad()
    print("\nTesting with torch.no_grad():")
    cfs_nogad = []
    for i in range(3):
        cf = torch.randn(10, requires_grad=True)
        cfs_nogad.append(cf)
    
    cfs_stacked_nograd = torch.stack(cfs_nogad, dim=0)
    
    with torch.no_grad():
        # Simulate model prediction in no_grad context
        pred = cfs_stacked_nograd.sum()
    
    print(f"   Prediction requires_grad: {pred.requires_grad}")
    try:
        pred.backward()
        print("   Backward pass succeeded")
    except RuntimeError as e:
        print(f"   Backward pass failed: {str(e)[:50]}...")
    
    print("\n" + "="*60)
    print("\nSUMMARY:")
    print(f"✓ Correct implementation (no detach): Gradients flow = {has_gradients}")
    print(f"✗ Broken implementation (with detach): Gradients flow = {has_gradients_broken}")
    print("\nConclusion: The detach() call breaks the computational graph!")


if __name__ == "__main__":
    test_gradient_flow()
