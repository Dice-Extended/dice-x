"""
Comprehensive test to verify gradient flow in the perturbation process
This test simulates the actual workflow in dice_pytorch.py
"""
import torch
import torch.nn as nn


class SimpleMockModel(nn.Module):
    """Simple model to simulate predictions"""
    def __init__(self, input_dim=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))


def test_broken_version():
    """Test the BROKEN version with torch.no_grad() and detach()"""
    print("="*70)
    print("Testing BROKEN VERSION (original code)")
    print("="*70)
    
    model = SimpleMockModel()
    
    # Simulate self.cfs
    cfs = []
    for i in range(3):
        cf = torch.randn(10, requires_grad=True)
        cfs.append(cf)
    
    print("\n1. Initial CFs:")
    print(f"   CFs require_grad: {[cf.requires_grad for cf in cfs]}")
    
    # Simulate generate_perturbations() with BROKEN code
    cfs_stacked = torch.stack(cfs, dim=0)
    perturbed_cfs = torch.nn.Parameter(cfs_stacked.clone(), requires_grad=True)
    
    # BROKEN: Using torch.no_grad()
    with torch.no_grad():
        model.eval()
        pred_original = model(cfs_stacked)
        pred_perturbed = model(perturbed_cfs)
    
    print(f"\n2. After no_grad context:")
    print(f"   pred_original requires_grad: {pred_original.requires_grad}")
    print(f"   pred_perturbed requires_grad: {pred_perturbed.requires_grad}")
    
    # Try to compute loss
    try:
        class_loss = torch.mean((pred_original - pred_perturbed) ** 2)
        distance = torch.norm(perturbed_cfs - cfs_stacked, p=2)
        loss = class_loss + 0.01 * distance
        
        print(f"\n3. Loss computation:")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        
        # Try backward
        loss.backward()
        print(f"\n4. After backward:")
        print(f"   Original CFs have gradients: {all(cf.grad is not None for cf in cfs)}")
        
    except RuntimeError as e:
        print(f"\n✗ ERROR: {str(e)}")
    
    # BROKEN: Using detach()
    perturbed_cfs_detached = perturbed_cfs.detach()
    print(f"\n5. After detach():")
    print(f"   perturbed_cfs_detached requires_grad: {perturbed_cfs_detached.requires_grad}")
    
    # Now use in robustness loss
    robustness = torch.mean(cfs_stacked - perturbed_cfs_detached) ** 2
    print(f"   robustness requires_grad: {robustness.requires_grad}")
    
    if robustness.requires_grad:
        robustness.backward()
        print(f"   CFs have gradients after robustness.backward(): {all(cf.grad is not None for cf in cfs)}")
    else:
        print(f"   ✗ Cannot backpropagate through robustness loss!")


def test_fixed_version():
    """Test the FIXED version without torch.no_grad() and detach()"""
    print("\n\n" + "="*70)
    print("Testing FIXED VERSION (without no_grad and detach)")
    print("="*70)
    
    model = SimpleMockModel()
    
    # Simulate self.cfs
    cfs = []
    for i in range(3):
        cf = torch.randn(10, requires_grad=True)
        cfs.append(cf)
    
    print("\n1. Initial CFs:")
    print(f"   CFs require_grad: {[cf.requires_grad for cf in cfs]}")
    
    # Simulate generate_perturbations() with FIXED code
    cfs_stacked = torch.stack(cfs, dim=0)
    perturbed_cfs = cfs_stacked.clone() + torch.randn_like(cfs_stacked) * 0.05
    
    # FIXED: NO torch.no_grad()
    model.eval()
    pred_original = model(cfs_stacked)
    pred_perturbed = model(perturbed_cfs)
    
    print(f"\n2. Without no_grad context:")
    print(f"   pred_original requires_grad: {pred_original.requires_grad}")
    print(f"   pred_perturbed requires_grad: {pred_perturbed.requires_grad}")
    
    # Compute loss
    class_loss = torch.mean((pred_original - pred_perturbed) ** 2)
    distance = torch.norm(perturbed_cfs - cfs_stacked, p=2)
    loss = class_loss + 0.01 * distance
    
    print(f"\n3. Loss computation:")
    print(f"   Loss requires_grad: {loss.requires_grad}")
    
    # Backward through perturbation loss
    loss.backward()
    print(f"\n4. After perturbation loss.backward():")
    print(f"   Original CFs have gradients: {all(cf.grad is not None for cf in cfs)}")
    if cfs[0].grad is not None:
        print(f"   CF[0] gradient norm: {cfs[0].grad.norm().item():.6f}")
    
    # Reset gradients
    for cf in cfs:
        cf.grad = None
    
    # FIXED: NO detach() when returning perturbed_cfs
    perturbed_cfs_no_detach = perturbed_cfs  # Keep gradient connection
    print(f"\n5. Without detach():")
    print(f"   perturbed_cfs_no_detach requires_grad: {perturbed_cfs_no_detach.requires_grad}")
    
    # Now use in robustness loss
    cfs_stacked_new = torch.stack(cfs, dim=0)
    robustness = torch.mean((cfs_stacked_new - perturbed_cfs_no_detach) ** 2)
    print(f"   robustness requires_grad: {robustness.requires_grad}")
    
    if robustness.requires_grad:
        robustness.backward()
        has_grads = all(cf.grad is not None for cf in cfs)
        print(f"   ✓ CFs have gradients after robustness.backward(): {has_grads}")
        if has_grads and cfs[0].grad is not None:
            print(f"   ✓ CF[0] gradient norm: {cfs[0].grad.norm().item():.6f}")
    else:
        print(f"   ✗ Cannot backpropagate through robustness loss!")


def test_complete_workflow():
    """Test the complete workflow with both losses combined"""
    print("\n\n" + "="*70)
    print("Testing COMPLETE WORKFLOW (main loss + robustness loss)")
    print("="*70)
    
    model = SimpleMockModel()
    optimizer_model = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Simulate self.cfs with optimizer
    cfs = []
    for i in range(3):
        cf = torch.randn(10, requires_grad=True)
        cfs.append(cf)
    
    optimizer_cfs = torch.optim.Adam(cfs, lr=0.01)
    
    print("\n1. Setup complete")
    print(f"   Number of CFs: {len(cfs)}")
    print(f"   All CFs require_grad: {all(cf.requires_grad for cf in cfs)}")
    
    # Training loop iteration
    for iteration in range(3):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        optimizer_cfs.zero_grad()
        optimizer_model.zero_grad()
        
        # Stack CFs
        cfs_stacked = torch.stack(cfs, dim=0)
        
        # Main losses (y-loss, proximity, diversity)
        target = torch.ones(3, 1)
        predictions = model(cfs_stacked)
        y_loss = nn.BCELoss()(predictions, target)
        
        # Generate perturbations WITHOUT detach
        perturbed_cfs = cfs_stacked + torch.randn_like(cfs_stacked) * 0.05
        
        # Robustness loss WITHOUT no_grad
        pred_perturbed = model(perturbed_cfs)
        robustness_loss = torch.mean((predictions - pred_perturbed) ** 2)
        
        # Combined loss
        total_loss = y_loss + 0.5 * robustness_loss
        
        print(f"   y_loss: {y_loss.item():.4f}")
        print(f"   robustness_loss: {robustness_loss.item():.4f}")
        print(f"   total_loss: {total_loss.item():.4f}")
        
        # Backward
        total_loss.backward()
        
        # Check gradients
        has_grads = all(cf.grad is not None for cf in cfs)
        print(f"   ✓ CFs have gradients: {has_grads}")
        if has_grads:
            avg_grad_norm = sum(cf.grad.norm().item() for cf in cfs) / len(cfs)
            print(f"   ✓ Average CF gradient norm: {avg_grad_norm:.6f}")
        
        # Update
        optimizer_cfs.step()
    
    print("\n✓ Complete workflow with gradient flow successful!")


if __name__ == "__main__":
    test_broken_version()
    test_fixed_version()
    test_complete_workflow()
    
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n✗ BROKEN VERSION:")
    print("  - torch.no_grad() prevents gradient computation")
    print("  - detach() severs computational graph")
    print("  - Robustness loss cannot influence CF optimization")
    print("\n✓ FIXED VERSION:")
    print("  - Remove torch.no_grad() to enable gradient tracking")
    print("  - Remove detach() to maintain gradient flow")
    print("  - Robustness loss can now backpropagate to CFs")
    print("\n✓ COMPLETE WORKFLOW:")
    print("  - Both main loss and robustness loss contribute to CF optimization")
    print("  - Gradients flow correctly through the entire computation")
