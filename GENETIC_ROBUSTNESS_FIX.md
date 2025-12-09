# Fix: Robustness Loss Always 1.0 in Genetic Algorithm

## Problem

You observed that the robustness loss remained at 1.0 throughout all iterations when using `dice_genetic.py`. This indicated that the counterfactuals were being evaluated as perfectly robust (no change under perturbations).

## Root Cause

The issue was in the `compute_loss` method (line 712):

```python
perturbed_cfs = self.generate_perturbations_fast(input_instance)
```

The `generate_perturbations_fast` method has default parameters:
- `force_flip_constraint=True` (by default)
- `tol=0.05` (very tight tolerance)

**What this means:**
The method was designed to find perturbations that **maintain almost the same model predictions** (within 5% difference). This is useful for some applications, but defeats the purpose of measuring robustness!

### The Constraint Logic

```python
if force_flip_constraint:
    hit_mask = delta <= tol  # Only keep perturbations with prediction change <= 0.05
else:
    hit_mask = np.ones(len(delta), dtype=bool)  # Keep all perturbations
```

When `force_flip_constraint=True`:
- Perturbations where predictions differ by >5% are rejected
- Only "safe" perturbations that barely change anything are kept
- Result: Original and perturbed CFs are almost identical
- Sørensen-Dice coefficient ≈ 1.0 (perfect similarity)
- **Robustness loss = 1.0 always!**

## Why Robustness Should Change

For robustness to be meaningful:
1. **Perturbations should be diverse** - They should change features meaningfully
2. **Predictions can change** - Some perturbations might flip the class
3. **Robustness measures stability** - If a CF is robust, small perturbations shouldn't dramatically change outcomes

The Sørensen-Dice coefficient should vary:
- **High (near 1.0)**: CF is robust - perturbations don't change much
- **Low (near 0.0)**: CF is fragile - perturbations cause significant changes

## The Fix

Changed the call in `compute_loss` method:

### Before (Broken):
```python
perturbed_cfs = self.generate_perturbations_fast(input_instance)
```

### After (Fixed):
```python
perturbed_cfs = self.generate_perturbations_fast(
    input_instance, 
    method=perturbation_method if perturbation_method in ["gaussian", "random", "spherical"] else "gaussian",
    force_flip_constraint=False,  # Allow perturbations with different predictions
    tol=0.2,  # More relaxed tolerance
    std_dev=kwargs.get('std_dev', 0.10),
    max_radius=kwargs.get('max_radius', 0.5)
)
```

### Key Changes:
1. **`force_flip_constraint=False`**: Perturbations can have any prediction change
2. **`tol=0.2`**: More relaxed tolerance (if constraint were enabled)
3. **Pass perturbation method**: Uses the method specified by user (`gaussian`, `random`, `spherical`)
4. **Configurable parameters**: Allows user to control `std_dev` and `max_radius` via kwargs

## Expected Behavior After Fix

After this fix, you should see:
- **Robustness loss varying** across iterations (not stuck at 1.0)
- **Lower robustness** for fragile counterfactuals (those sensitive to perturbations)
- **Higher robustness** for stable counterfactuals (those resistant to perturbations)
- **Meaningful optimization** where the algorithm can trade off between proximity, sparsity, diversity, and robustness

### Example Expected Plot:
```
Robustness Loss: Should vary between ~0.3 to ~0.9
- Early iterations: May be low (fragile CFs)
- Later iterations: Should improve as algorithm finds more robust CFs
```

## Additional Notes

### Perturbation Parameters

You can now control perturbation generation via kwargs:

```python
exp.generate_counterfactuals(
    query_instance,
    total_CFs=5,
    perturbation_method="gaussian",
    std_dev=0.15,  # Standard deviation for Gaussian noise
    max_radius=0.6  # Maximum radius for spherical perturbations
)
```

### Comparison with PyTorch Method

The PyTorch gradient-based method (`dice_pytorch.py`) doesn't have this issue because:
1. It generates perturbations without prediction constraints
2. The perturbation optimization is separate from CF optimization
3. Robustness is computed on the final perturbed state

### Testing the Fix

Run your notebook again with the genetic algorithm and check:

```python
exp = dice_ml_x.Dice(d, m, method="genetic")
exp_result = exp.generate_counterfactuals(
    x_test[0:1], 
    total_CFs=5, 
    perturbation_method="gaussian"
)

# Plot loss history
plot_loss_metrics(exp.loss_history)
```

You should now see the robustness loss line changing across iterations instead of flat at 1.0!

## Summary

**Problem**: Robustness loss was always 1.0 because perturbations were constrained to maintain almost identical predictions.

**Solution**: Removed the `force_flip_constraint` and allow perturbations to vary freely, enabling meaningful robustness measurement.

**Result**: Robustness loss now properly reflects how stable counterfactuals are under perturbations, contributing meaningfully to the optimization process.
