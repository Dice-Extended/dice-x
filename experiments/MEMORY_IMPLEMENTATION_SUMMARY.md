# Memory Management Implementation Summary

## What Was Created

### 1. Core Module: `experiments/memory_management.py`
A comprehensive, reusable memory management library with:

**Features:**
- ✅ Real-time memory monitoring (system + process)
- ✅ Automatic cleanup when thresholds exceeded
- ✅ Backend-specific cleanup (TensorFlow, PyTorch, sklearn)
- ✅ Memory profiling and visualization
- ✅ Context managers and decorators
- ✅ Emergency cleanup procedures
- ✅ Adaptive batch sizing

**Key Components:**
- `MemoryMonitor` - Context manager for monitoring code blocks
- `@memory_checkpoint` - Decorator for automatic monitoring
- `MemoryProfiler` - Track memory over time with plots
- `clear_session_memory()` - Manual cleanup
- `get_memory_stats()` - Get current memory info
- `emergency_cleanup()` - Aggressive cleanup

### 2. Integration: `binning_sensitivity_test.py`
Updated to use memory management:

**Changes Made:**
- ✅ Imported memory management utilities
- ✅ Added memory config to `BinningTestConfig`
- ✅ Wrapped tasks with `MemoryMonitor`
- ✅ Added periodic memory checkpoints
- ✅ Cleanup after each backend/dataset
- ✅ Logging at key points

**New Config Options:**
```python
enable_memory_monitoring: bool = True
memory_cleanup_threshold_percent: float = 75.0
memory_cleanup_after_backend: bool = True
memory_checkpoint_interval: int = 5
```

### 3. AWS Helper: `run_aws_experiment.py`
Convenience script with pre-configured settings for different AWS instance types:

**Features:**
- ✅ Instance-specific configurations (t2.medium, t2.large, t2.xlarge)
- ✅ Automatic memory profiling
- ✅ Single dataset mode for constrained memory
- ✅ Command-line interface
- ✅ Error handling and recovery suggestions

**Usage:**
```bash
# Full experiment on t2.large
python experiments/binning_test/run_aws_experiment.py --instance-type t2.large

# Single dataset
python experiments/binning_test/run_aws_experiment.py --single-dataset compas
```

### 4. Testing: `test_memory_management.py`
Comprehensive test suite to verify everything works:

**Tests:**
- ✅ Basic memory monitoring
- ✅ Context manager functionality
- ✅ Decorator functionality
- ✅ Memory profiling
- ✅ Stats collection
- ✅ Backend-specific cleanup

**Usage:**
```bash
python experiments/test_memory_management.py
```

### 5. Documentation

**`MEMORY_MANAGEMENT.md`** (Full guide)
- Complete API reference
- AWS-specific instructions
- Troubleshooting guide
- Example configurations
- Best practices

**`MEMORY_QUICK_REF.md`** (Quick reference)
- Copy-paste code snippets
- Common commands
- Quick configs
- Troubleshooting tips

## How to Use on AWS

### Step 1: Choose Instance Size

| Instance | RAM | Recommended Config |
|----------|-----|-------------------|
| t2.medium | 4 GB | 1 dataset, sklearn only, 5 test points |
| t2.large | 8 GB | 2 datasets, sklearn + TF2, 10 test points |
| t2.xlarge | 16 GB | 3 datasets, all backends, 15 test points |

### Step 2: Run Experiment

```bash
# Option A: Use helper script (easiest)
python experiments/binning_test/run_aws_experiment.py --instance-type t2.large

# Option B: Run directly with custom config
python experiments/binning_test/binning_sensitivity_test.py

# Option C: Single dataset mode
python experiments/binning_test/run_aws_experiment.py --single-dataset compas
```

### Step 3: Monitor

```bash
# Terminal 1: Run experiment
python experiments/binning_test/run_aws_experiment.py --instance-type t2.large

# Terminal 2: Monitor memory
watch -n 5 free -h
```

### Step 4: Results

Memory management automatically:
- Logs memory at key points
- Cleans up when threshold exceeded
- Saves memory profile to `experiment_artefacts/memory_profile_*.csv`
- Generates plot `experiment_artefacts/memory_profile_*.png`

## Key Benefits

### 1. Prevents OOM Kills
Automatic cleanup before system runs out of memory

### 2. Visibility
Know exactly what's using memory and when

### 3. Reusable
Can be used in any experiment, not just binning tests:

```python
from experiments.memory_management import MemoryMonitor

with MemoryMonitor(name="My Experiment", auto_cleanup=True) as m:
    # Your code here
    m.checkpoint("after processing")
```

### 4. Adaptive
Automatically adjusts to available memory

### 5. Debuggable
Profiling helps identify memory leaks

## Example: Complete AWS Workflow

```python
from experiments.binning_test.binning_sensitivity_test import (
    binning_sensitivity_flow,
    BinningTestConfig
)
from experiments.memory_management import (
    setup_memory_logging,
    log_memory_usage,
    clear_session_memory
)

# Setup
setup_memory_logging()
log_memory_usage("Started")

# Configure for t2.large
config = BinningTestConfig(
    n_test_points=10,
    test_datasets=['adult-income', 'compas'],
    test_backends=['sklearn', 'TF2'],
    enable_memory_monitoring=True,
    memory_cleanup_threshold_percent=75.0,
    memory_cleanup_after_backend=True,
    memory_checkpoint_interval=5,
)

# Run
try:
    result = binning_sensitivity_flow(config=config)
    print(f"✅ Results: {result}")
finally:
    clear_session_memory(backend=None, aggressive=True)
    log_memory_usage("Done")
```

## Troubleshooting

### Still Getting OOM?

1. **Add swap space:**
   ```bash
   sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **Lower threshold:**
   ```python
   config.memory_cleanup_threshold_percent = 65.0  # More aggressive
   ```

3. **Reduce batch sizes:**
   ```python
   config.n_test_points = 5
   config.n_repeat = 30
   config.n_samples_fidelity = 500
   ```

4. **Process one at a time:**
   ```bash
   # Run separately for each dataset
   python run_aws_experiment.py --single-dataset compas
   python run_aws_experiment.py --single-dataset adult-income
   ```

### Memory Leak?

```python
from experiments.memory_management import MemoryProfiler

profiler = MemoryProfiler()
# Run your code with profiler.record() calls
profiler.plot("leak_analysis.png")  # Visual inspection
```

## Files Created

```
experiments/
├── memory_management.py          # Core library (reusable)
├── test_memory_management.py     # Test suite
├── MEMORY_MANAGEMENT.md          # Full documentation
├── MEMORY_QUICK_REF.md           # Quick reference
└── binning_test/
    ├── binning_sensitivity_test.py  # Updated with memory mgmt
    └── run_aws_experiment.py        # AWS helper script
```

## Next Steps

1. **Test locally:**
   ```bash
   python experiments/test_memory_management.py
   ```

2. **Test binning (local):**
   ```bash
   python experiments/binning_test/binning_sensitivity_test.py
   ```

3. **Deploy to AWS:**
   ```bash
   # Copy code to AWS instance
   scp -r DiCE-X ec2-user@your-instance:/home/ec2-user/
   
   # SSH in
   ssh ec2-user@your-instance
   
   # Run
   cd DiCE-X
   python experiments/binning_test/run_aws_experiment.py --instance-type t2.large
   ```

## Support

- Full docs: `experiments/MEMORY_MANAGEMENT.md`
- Quick ref: `experiments/MEMORY_QUICK_REF.md`
- Test suite: `python experiments/test_memory_management.py`
