# Memory Management Quick Reference

## 🚀 Quick Setup

```python
from experiments.memory_management import setup_memory_logging, log_memory_usage
setup_memory_logging()
log_memory_usage("Started")
```

## 📊 Check Current Memory

```python
from experiments.memory_management import get_memory_stats
stats = get_memory_stats()
print(f"System: {stats.percent:.1f}%, Process: {stats.process_mb:.0f} MB")
```

## 🧹 Manual Cleanup

```python
from experiments.memory_management import clear_session_memory

# Light cleanup
clear_session_memory(backend='TF2')

# Aggressive cleanup
clear_session_memory(backend=None, aggressive=True)
```

## 🎯 Monitor a Code Block

```python
from experiments.memory_management import MemoryMonitor

with MemoryMonitor(
    name="My Task",
    threshold_percent=80.0,
    auto_cleanup=True,
    backend='TF2'
) as monitor:
    # Your code
    for i in range(100):
        process_data(i)
        monitor.checkpoint(f"iteration_{i}")
```

## 🏷️ Decorator for Functions

```python
from experiments.memory_management import memory_checkpoint

@memory_checkpoint(threshold_mb=2000, backend='TF2', auto_cleanup=True)
def my_function(data):
    # Your code
    return result
```

## 📈 Profile Memory Over Time

```python
from experiments.memory_management import MemoryProfiler

profiler = MemoryProfiler()
profiler.record("start")

for i in range(100):
    process(data[i])
    profiler.record(f"step_{i}")

profiler.summary()
profiler.save("profile.csv")
profiler.plot("profile.png")
```

## ⚙️ Configure Binning Test

```python
from experiments.binning_test.binning_sensitivity_test import BinningTestConfig

# For t2.medium (4 GB)
config = BinningTestConfig(
    n_test_points=5,
    test_datasets=['compas'],
    test_backends=['sklearn'],
    enable_memory_monitoring=True,
    memory_cleanup_threshold_percent=70.0,
    memory_checkpoint_interval=2,
)

# For t2.large (8 GB)
config = BinningTestConfig(
    n_test_points=10,
    test_datasets=['adult-income', 'compas'],
    test_backends=['sklearn', 'TF2'],
    enable_memory_monitoring=True,
    memory_cleanup_threshold_percent=75.0,
    memory_checkpoint_interval=5,
)

# For t2.xlarge (16 GB)
config = BinningTestConfig(
    n_test_points=15,
    test_datasets=['adult-income', 'compas', 'lending-club'],
    test_backends=['sklearn', 'PYT', 'TF2'],
    enable_memory_monitoring=True,
    memory_cleanup_threshold_percent=80.0,
    memory_checkpoint_interval=10,
)
```

## 🚨 Emergency Cleanup

```python
from experiments.memory_management import emergency_cleanup
emergency_cleanup()  # Very aggressive - use as last resort
```

## 🐧 AWS Commands

```bash
# Check memory
free -h
watch -n 5 free -h  # Monitor every 5 seconds

# Add swap (4GB)
sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Run experiment
python experiments/binning_test/run_aws_experiment.py --instance-type t2.large

# Run single dataset
python experiments/binning_test/run_aws_experiment.py --single-dataset compas --backends sklearn

# Test memory management
python experiments/test_memory_management.py
```

## 🔍 Troubleshooting

**Memory too high?**
- Lower `memory_cleanup_threshold_percent` (e.g., 65%)
- Reduce `n_test_points`
- Test fewer datasets/backends
- Add swap space

**OOM killed?**
```bash
# Check logs
sudo dmesg | grep -i "killed process"

# Use smaller batches or larger instance
```

**Memory leak?**
```python
# Profile to find the issue
profiler = MemoryProfiler()
# ... run code ...
profiler.plot("leak_analysis.png")
```

## 📖 Full Documentation

See `experiments/MEMORY_MANAGEMENT.md` for complete guide.
