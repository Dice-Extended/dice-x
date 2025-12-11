# Memory Management for Experiments

This guide explains how to use the memory management utilities when running experiments on AWS or other cloud instances with limited memory.

## Quick Start

### 1. Basic Usage in Your Script

```python
from experiments.memory_management import (
    MemoryMonitor,
    memory_checkpoint,
    clear_session_memory,
    log_memory_usage,
    setup_memory_logging
)

# Setup logging at the start
setup_memory_logging()

# Monitor memory during execution
with MemoryMonitor(
    name="My Experiment",
    threshold_percent=80.0,  # Trigger cleanup at 80% memory
    auto_cleanup=True,
    backend='TF2'  # or 'PYT', 'sklearn'
) as monitor:
    # Your code here
    for i in range(100):
        # Do work
        monitor.checkpoint(f"iteration_{i}")
```

### 2. Decorator for Tasks (Prefect)

```python
@task
@memory_checkpoint(threshold_mb=2000, backend='TF2', auto_cleanup=True)
def my_expensive_task(data):
    # Your code here
    return results
```

### 3. Manual Cleanup

```python
# Log current memory
log_memory_usage("Before processing")

# Do some work...

# Clean up when done
clear_session_memory(backend='TF2', aggressive=True)
log_memory_usage("After cleanup")
```

## Configuration for AWS

### Recommended Settings for Different Instance Types

#### t2.medium (4 GB RAM)
```python
config = BinningTestConfig(
    enable_memory_monitoring=True,
    memory_cleanup_threshold_percent=70.0,  # Aggressive
    memory_cleanup_after_backend=True,
    memory_checkpoint_interval=3,  # Check every 3 test points
    n_test_points=5,  # Smaller batches
)
```

#### t2.large (8 GB RAM)
```python
config = BinningTestConfig(
    enable_memory_monitoring=True,
    memory_cleanup_threshold_percent=75.0,
    memory_cleanup_after_backend=True,
    memory_checkpoint_interval=5,
    n_test_points=10,
)
```

#### t2.xlarge (16 GB RAM)
```python
config = BinningTestConfig(
    enable_memory_monitoring=True,
    memory_cleanup_threshold_percent=80.0,
    memory_cleanup_after_backend=True,
    memory_checkpoint_interval=10,
    n_test_points=20,
)
```

## Features

### 1. MemoryMonitor (Context Manager)

Automatically monitors memory at start/end and optionally during execution.

```python
with MemoryMonitor(
    name="Training Phase",
    threshold_percent=75.0,      # System memory threshold
    threshold_mb=1500,            # Process memory threshold
    auto_cleanup=True,            # Auto-cleanup when exceeded
    check_interval=5,             # Check every 5 checkpoints
    backend='TF2'                 # Backend to clear
) as monitor:
    for batch in batches:
        process(batch)
        monitor.checkpoint("batch processed")
```

### 2. memory_checkpoint Decorator

Monitors memory before/after function execution and cleans up if needed.

```python
@memory_checkpoint(
    threshold_percent=80.0,
    threshold_mb=2000,
    auto_cleanup=True,
    backend='PYT'
)
def train_model(data):
    # Your training code
    return model
```

### 3. Manual Memory Management

```python
# Get current stats
from experiments.memory_management import get_memory_stats
stats = get_memory_stats()
print(f"System: {stats.used_mb:.0f}/{stats.total_mb:.0f} MB ({stats.percent:.1f}%)")
print(f"Process: {stats.process_mb:.0f} MB")

# Clear memory
clear_session_memory(backend='TF2', aggressive=True)

# Emergency cleanup (very aggressive)
from experiments.memory_management import emergency_cleanup
emergency_cleanup()
```

### 4. Memory Profiling

Track memory usage over time for analysis:

```python
from experiments.memory_management import MemoryProfiler

profiler = MemoryProfiler()

for i in range(100):
    # Do work
    result = process_batch(data[i])
    profiler.record(f"batch_{i}")

# Show summary
profiler.summary()

# Save to CSV
profiler.save("memory_profile.csv")

# Plot
profiler.plot("memory_usage.png")
```

### 5. Adaptive Batch Sizing

Automatically reduce batch size when memory is high:

```python
from experiments.memory_management import memory_limited_batch

batch_size = 32

for i in range(0, len(data), batch_size):
    with memory_limited_batch(
        batch_size=batch_size,
        max_memory_percent=75.0,
        min_batch_size=4
    ) as actual_batch_size:
        batch = data[i:i+actual_batch_size]
        process(batch)
```

## AWS-Specific Tips

### 1. CloudWatch Integration

Monitor your instance memory from CloudWatch:

```bash
# Install CloudWatch agent
sudo yum install amazon-cloudwatch-agent

# Configure to send memory metrics
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -s \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/config.json
```

### 2. Swap Space (Emergency Buffer)

Add swap space for emergency situations:

```bash
# Create 4GB swap file
sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 3. Running Experiments

```bash
# SSH to AWS instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Navigate to project
cd DiCE-X

# Run with memory monitoring
python experiments/binning_test/binning_sensitivity_test.py 2>&1 | tee experiment.log

# Monitor memory in another terminal
watch -n 5 free -h
```

### 4. Auto-recovery Script

Create a script to automatically restart if OOM killed:

```bash
#!/bin/bash
# run_with_recovery.sh

MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "Attempt $((RETRY_COUNT + 1)) of $MAX_RETRIES"
    
    python experiments/binning_test/binning_sensitivity_test.py
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Success!"
        exit 0
    elif [ $EXIT_CODE -eq 137 ]; then
        echo "OOM killed. Retrying with more aggressive memory management..."
        RETRY_COUNT=$((RETRY_COUNT + 1))
        sleep 10
    else
        echo "Failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
done

echo "Max retries reached. Giving up."
exit 1
```

## Troubleshooting

### Memory Still Too High?

1. **Reduce batch sizes**:
   ```python
   config.n_test_points = 5  # Smaller
   ```

2. **More aggressive cleanup**:
   ```python
   config.memory_cleanup_threshold_percent = 65.0  # Lower threshold
   config.memory_checkpoint_interval = 2  # Check more often
   ```

3. **Process datasets sequentially**:
   ```python
   config.test_datasets = ['adult-income']  # One at a time
   ```

4. **Disable certain backends**:
   ```python
   config.test_backends = ['sklearn']  # Skip TF2/PYT if too heavy
   ```

### OOM Killed?

If your process is killed by the OOM killer:

1. Check system logs:
   ```bash
   sudo dmesg | grep -i "killed process"
   ```

2. Reduce memory footprint:
   - Use smaller models
   - Reduce `n_repeat` for robustness tests
   - Lower `n_samples_fidelity`
   - Process one backend at a time

3. Consider larger instance or more swap space

### Memory Leaks?

If memory keeps growing:

1. **Enable profiling**:
   ```python
   from experiments.memory_management import MemoryProfiler
   
   profiler = MemoryProfiler()
   # Run your code
   profiler.plot("leak_analysis.png")
   ```

2. **Check for TensorFlow/PyTorch sessions**:
   - TF2: Ensure `tf.keras.backend.clear_session()` is called
   - PyTorch: Clear cache with `torch.cuda.empty_cache()`

3. **Look for large object retention**:
   ```python
   import gc
   import sys
   
   # Find largest objects
   def get_largest_objects(n=10):
       gc.collect()
       objects = gc.get_objects()
       return sorted(objects, key=sys.getsizeof, reverse=True)[:n]
   ```

## API Reference

### Functions

- **`get_memory_stats()`**: Get current memory statistics
- **`log_memory_usage(context, level)`**: Log memory with context
- **`clear_session_memory(backend, aggressive)`**: Clear memory
- **`emergency_cleanup()`**: Aggressive emergency cleanup
- **`setup_memory_logging(level)`**: Configure logging

### Classes

- **`MemoryMonitor`**: Context manager for monitoring
- **`MemoryProfiler`**: Track memory over time
- **`MemoryStats`**: Memory statistics dataclass

### Decorators

- **`@memory_checkpoint(...)`**: Monitor function execution

### Context Managers

- **`memory_limited_batch(...)`**: Adaptive batch sizing

## Example: Complete AWS Workflow

```python
from pathlib import Path
from experiments.binning_test.binning_sensitivity_test import (
    binning_sensitivity_flow,
    BinningTestConfig,
    DefaultPaths
)
from experiments.memory_management import setup_memory_logging, log_memory_usage

# Setup
setup_memory_logging()
log_memory_usage("AWS Experiment Started")

# Configure for AWS instance (e.g., t2.large with 8GB RAM)
config = BinningTestConfig(
    n_test_points=10,
    fixed_bins=[5, 10, 15, 20],
    test_datasets=['adult-income', 'compas'],  # Start with 2
    test_backends=['sklearn', 'TF2'],  # Skip PyTorch if memory constrained
    
    # Memory settings
    enable_memory_monitoring=True,
    memory_cleanup_threshold_percent=75.0,
    memory_cleanup_after_backend=True,
    memory_checkpoint_interval=5,
)

# Run
try:
    result = binning_sensitivity_flow(config=config)
    print(f"✅ Success! Results: {result}")
except Exception as e:
    print(f"❌ Failed: {e}")
    log_memory_usage("Error occurred", level="ERROR")
finally:
    from experiments.memory_management import clear_session_memory
    clear_session_memory(backend=None, aggressive=True)
    log_memory_usage("Final cleanup complete")
```
