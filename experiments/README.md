# DiCE-X Experiments

This directory contains experimental scripts for evaluating DiCE-X counterfactual explanations.

## Directory Structure

```
experiments/
├── binning_test/                      # Binning sensitivity analysis
│   ├── binning_sensitivity_test.py   # Main experiment script
│   ├── run_aws_experiment.py         # AWS-optimized runner
│   └── test_checkpoint.py            # Checkpoint system tests
├── memory_management.py               # Memory monitoring utilities
├── test_memory_management.py          # Memory management tests
├── CHECKPOINT_GUIDE.md                # Full checkpoint documentation
├── CHECKPOINT_QUICK_REF.md            # Quick reference for checkpointing
├── CHECKPOINT_IMPLEMENTATION_SUMMARY.md # Implementation overview
├── MEMORY_MANAGEMENT.md               # Full memory management guide
├── MEMORY_QUICK_REF.md                # Quick reference for memory
└── MEMORY_IMPLEMENTATION_SUMMARY.md   # Memory implementation overview
```

## Key Features

### 1. Binning Sensitivity Analysis

Comprehensive evaluation of how bin counts affect counterfactual quality across:
- 4 datasets (adult-income, compas, lending-club, german-credit)
- 3 backends (sklearn, PyTorch, TensorFlow 2)
- Multiple binning strategies (fixed, Sturges, Scott, Freedman-Diaconis)

**Metrics evaluated:**
- Validity (CFs satisfy target class)
- Proximity (continuous & categorical)
- Diversity (continuous & categorical)
- Sparsity (continuous features)
- Robustness (stability under perturbation)
- Fidelity (surrogate model quality)
- Computation time

### 2. Memory Management System

Automatic memory monitoring and cleanup for long-running experiments:
- Real-time memory tracking with psutil
- Configurable threshold-based cleanup
- Backend-specific cleanup (TensorFlow, PyTorch)
- Context managers and decorators
- Emergency cleanup for OOM prevention

**See:** `MEMORY_MANAGEMENT.md`, `MEMORY_QUICK_REF.md`

### 3. Checkpoint/Resume System

Fault-tolerant execution with automatic progress saving:
- Incremental checkpoint saves
- Automatic resume from failures
- Configuration-level tracking
- Zero manual intervention
- <0.1% performance overhead

**See:** `CHECKPOINT_GUIDE.md`, `CHECKPOINT_QUICK_REF.md`

## Quick Start

### Local Development

```bash
# Run basic test (small scale)
cd experiments/binning_test
uv run binning_sensitivity_test.py
```

### AWS Deployment

```bash
# Run on specific instance type
uv run run_aws_experiment.py --instance-type r5.xlarge

# Available instance types:
# - t2.medium (4GB RAM)   - minimal config
# - t2.large (8GB RAM)    - moderate config
# - t2.xlarge (16GB RAM)  - large config
# - r5.xlarge (32GB RAM)  - full-scale config
```

### Resume Interrupted Experiment

```bash
# If experiment was interrupted, just run again
# Checkpoint system automatically resumes from last save
uv run run_aws_experiment.py --instance-type r5.xlarge
```

## Configuration

### Basic Configuration

```python
from binning_sensitivity_test import BinningTestConfig

config = BinningTestConfig(
    # Experiment scale
    n_test_points=20,
    fixed_bins=[5, 10, 15, 20],
    test_datasets=['adult-income', 'compas'],
    test_backends=['sklearn', 'PYT'],
    
    # Memory management
    enable_memory_monitoring=True,
    memory_cleanup_threshold_percent=75.0,
    memory_cleanup_after_backend=True,
    
    # Checkpointing
    enable_checkpointing=True,
    checkpoint_every_n_configs=5,
    resume_from_checkpoint=True,
)
```

### AWS Instance Configurations

Pre-configured for optimal performance on each instance type:

| Instance | RAM | Test Points | Datasets | Backends | Checkpoint Every |
|----------|-----|-------------|----------|----------|-----------------|
| t2.medium | 4GB | 5 | 1 | 1 | 3 configs |
| t2.large | 8GB | 10 | 2 | 2 | 5 configs |
| t2.xlarge | 16GB | 15 | 3 | 3 | 10 configs |
| r5.xlarge | 32GB | 20 | 4 | 3 | 10 configs |

## Output Files

### Results

- `binning_sensitivity_results.csv` - Full results
- `binning_summary.csv` - Aggregated summary
- `binning_recommendation.txt` - Pareto-optimal recommendations
- `binning_{dataset}.csv` - Per-dataset results
- `binning_{backend}.csv` - Per-backend results

### Visualizations

Generated in `chart_artefacts/`:
- `binning_validity_by_bins.pdf`
- `binning_proximity_by_bins.pdf`
- `binning_diversity_by_bins.pdf`
- `binning_robustness_by_bins.pdf`
- Heatmaps for all metrics

### Checkpoints

During execution:
- `checkpoint_results.csv` - Incremental saves
- `checkpoint_timestamp.txt` - Progress tracking

After completion:
- `checkpoint_results_completed.csv` - Final archive

## Testing

```bash
# Test checkpoint system
cd experiments/binning_test
uv run python test_checkpoint.py

# Test memory management
cd experiments
uv run python test_memory_management.py
```

## Monitoring

### Memory Usage

Check memory during runs:
```bash
# Monitor memory consumption
watch -n 10 'free -h && ps aux | grep python | head -5'
```

### Checkpoint Progress

Check experiment progress:
```bash
# Count completed configurations
wc -l experiment_artefacts/checkpoint_results.csv

# View timestamp
cat experiment_artefacts/checkpoint_timestamp.txt

# Monitor in real-time
watch -n 60 'wc -l experiment_artefacts/checkpoint_results.csv'
```

## Best Practices

### 1. Start Small, Scale Up

```bash
# Test locally first
uv run binning_sensitivity_test.py  # Uses default config

# Then scale to AWS
uv run run_aws_experiment.py --instance-type t2.medium  # Quick test
uv run run_aws_experiment.py --instance-type r5.xlarge  # Full scale
```

### 2. Use Screen for Long Runs

```bash
# Start screen session
screen -S dice-experiment

# Run experiment
uv run run_aws_experiment.py --instance-type r5.xlarge

# Detach: Ctrl+A, then D
# Reattach later: screen -r dice-experiment
```

### 3. Monitor Logs

```bash
# Save logs to file
uv run run_aws_experiment.py --instance-type r5.xlarge 2>&1 | tee experiment.log

# Watch for checkpoints
tail -f experiment.log | grep checkpoint
```

### 4. Backup Critical Results

```bash
# Backup checkpoints periodically
cp experiment_artefacts/checkpoint_results.csv \
   backup/checkpoint_$(date +%Y%m%d_%H%M%S).csv
```

## Troubleshooting

### Out of Memory

**Symptoms:** Process killed, no error message

**Solutions:**
1. Reduce `n_test_points`
2. Lower `memory_cleanup_threshold_percent`
3. Enable aggressive cleanup
4. Use smaller instance or reduce dataset/backend count

### Checkpoint Not Loading

**Check:**
```bash
ls -lh experiment_artefacts/checkpoint_results.csv
head experiment_artefacts/checkpoint_results.csv
```

**Fix:**
- Verify CSV is valid
- Check file permissions
- Ensure output_dir matches

### Slow Execution

**Common causes:**
- TensorFlow backend (slower than sklearn/PyTorch)
- Large `n_repeat` for robustness
- High `n_samples_fidelity`

**Optimize:**
```python
config = BinningTestConfig(
    n_repeat=30,  # Reduce from 50
    n_samples_fidelity=500,  # Reduce from 1000
    test_backends=['sklearn'],  # Use fastest backend
)
```

## Advanced Usage

### Custom Binning Strategy

```python
# Add custom bins to test
config = BinningTestConfig(
    fixed_bins=[3, 7, 12, 25, 50],  # Custom sequence
)
```

### Selective Dataset Testing

```python
# Test only specific datasets
config = BinningTestConfig(
    test_datasets=['adult-income'],  # Single dataset
    test_backends=['sklearn', 'PYT', 'TF2'],  # All backends
)
```

### Backend Comparison

```python
# Compare backends on same dataset
config = BinningTestConfig(
    test_datasets=['compas'],
    test_backends=['sklearn', 'PYT', 'TF2'],
    n_test_points=20,
)
```

## Citation

If you use these experiments in your research, please cite:

```bibtex
@article{dicex2024,
  title={DiCE-X: Extended Counterfactual Explanations},
  author={...},
  journal={...},
  year={2024}
}
```

## Support

For issues, questions, or contributions:
1. Check documentation: `CHECKPOINT_GUIDE.md`, `MEMORY_MANAGEMENT.md`
2. Run tests: `test_checkpoint.py`, `test_memory_management.py`
3. Review logs for detailed error messages
4. Verify configuration matches your use case

## License

See LICENSE file in repository root.
