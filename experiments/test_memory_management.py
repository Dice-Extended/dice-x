"""
Simple test to verify memory management is working correctly.
Run this before deploying to AWS.
"""

import sys
from pathlib import Path

# Add DiCE-X to path
dice_x_path = Path(__file__).parent.parent.parent
if str(dice_x_path) not in sys.path:
    sys.path.insert(0, str(dice_x_path))

from memory_management import (
    MemoryMonitor,
    memory_checkpoint,
    clear_session_memory,
    log_memory_usage,
    get_memory_stats,
    setup_memory_logging,
    MemoryProfiler
)
import numpy as np
import time


def test_basic_monitoring():
    """Test basic memory monitoring."""
    print("\n" + "="*80)
    print("TEST 1: Basic Memory Monitoring")
    print("="*80)
    
    setup_memory_logging()
    
    log_memory_usage("Test started")
    
    # Allocate some memory
    data = []
    for i in range(5):
        data.append(np.random.rand(1000, 1000))
        log_memory_usage(f"After allocation {i+1}")
    
    # Cleanup
    del data
    clear_session_memory()
    log_memory_usage("After cleanup")
    
    print("✅ Test 1 passed\n")


def test_context_manager():
    """Test MemoryMonitor context manager."""
    print("\n" + "="*80)
    print("TEST 2: Context Manager")
    print("="*80)
    
    with MemoryMonitor(
        name="Test Operation",
        threshold_percent=75.0,
        auto_cleanup=True,
        check_interval=2
    ) as monitor:
        
        for i in range(10):
            # Simulate work
            _ = np.random.rand(500, 500)
            time.sleep(0.1)
            monitor.checkpoint(f"iteration_{i}")
    
    print("✅ Test 2 passed\n")


@memory_checkpoint(threshold_percent=90.0, auto_cleanup=True)
def test_decorator():
    """Test memory_checkpoint decorator."""
    print("\n" + "="*80)
    print("TEST 3: Decorator")
    print("="*80)
    
    # Simulate some work
    data = [np.random.rand(1000, 1000) for _ in range(3)]
    result = sum(d.sum() for d in data)
    
    print(f"Computation result: {result:.2e}")
    print("✅ Test 3 passed\n")
    
    return result


def test_profiler():
    """Test MemoryProfiler."""
    print("\n" + "="*80)
    print("TEST 4: Memory Profiler")
    print("="*80)
    
    profiler = MemoryProfiler()
    profiler.record("start")
    
    for i in range(5):
        _ = np.random.rand(500, 500)
        time.sleep(0.1)
        profiler.record(f"iteration_{i}")
    
    profiler.record("end")
    profiler.summary()
    
    # Save results
    output_dir = Path("experiment_artefacts")
    output_dir.mkdir(exist_ok=True)
    
    profiler.save(output_dir / "test_memory_profile.csv")
    
    try:
        profiler.plot(output_dir / "test_memory_profile.png")
        print("📊 Memory plot saved")
    except Exception as e:
        print(f"⚠️  Could not save plot: {e}")
    
    print("✅ Test 4 passed\n")


def test_memory_stats():
    """Test get_memory_stats function."""
    print("\n" + "="*80)
    print("TEST 5: Memory Statistics")
    print("="*80)
    
    stats = get_memory_stats()
    
    print(f"System Memory:")
    print(f"  Total:     {stats.total_mb:,.0f} MB")
    print(f"  Used:      {stats.used_mb:,.0f} MB ({stats.percent:.1f}%)")
    print(f"  Available: {stats.available_mb:,.0f} MB")
    print(f"\nProcess Memory:")
    print(f"  Used:      {stats.process_mb:,.0f} MB")
    
    print("\n✅ Test 5 passed\n")


def test_backend_cleanup():
    """Test backend-specific cleanup."""
    print("\n" + "="*80)
    print("TEST 6: Backend Cleanup")
    print("="*80)
    
    # Test TensorFlow cleanup
    try:
        import tensorflow as tf
        print("Testing TensorFlow cleanup...")
        
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,))
        ])
        
        log_memory_usage("After TF model creation")
        
        # Cleanup
        clear_session_memory(backend='TF2', aggressive=True)
        log_memory_usage("After TF cleanup")
        
        print("✅ TensorFlow cleanup OK")
    except ImportError:
        print("⚠️  TensorFlow not available, skipping")
    
    # Test PyTorch cleanup
    try:
        import torch
        print("Testing PyTorch cleanup...")
        
        # Create tensor
        tensor = torch.rand(1000, 1000)
        
        log_memory_usage("After PyTorch tensor creation")
        
        # Cleanup
        del tensor
        clear_session_memory(backend='PYT', aggressive=True)
        log_memory_usage("After PyTorch cleanup")
        
        print("✅ PyTorch cleanup OK")
    except ImportError:
        print("⚠️  PyTorch not available, skipping")
    
    print("\n✅ Test 6 passed\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("MEMORY MANAGEMENT TEST SUITE")
    print("="*80)
    
    tests = [
        ("Basic Monitoring", test_basic_monitoring),
        ("Context Manager", test_context_manager),
        ("Decorator", test_decorator),
        ("Profiler", test_profiler),
        ("Memory Stats", test_memory_stats),
        ("Backend Cleanup", test_backend_cleanup),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ Test '{name}' failed: {e}\n")
            failed += 1
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ All tests passed! Memory management is ready for AWS.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review errors above.")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    run_all_tests()
