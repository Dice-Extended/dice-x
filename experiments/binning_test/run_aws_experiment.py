"""
Example: Running Binning Test with Memory Management on AWS

This script demonstrates how to run the binning sensitivity test
with proper memory management for AWS instances.
"""

from pathlib import Path
import logging

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s'
)

from experiments.binning_test.binning_sensitivity_test import (
    binning_sensitivity_flow,
    BinningTestConfig,
    DefaultPaths
)
from experiments.memory_management import (
    setup_memory_logging,
    log_memory_usage,
    clear_session_memory,
    MemoryProfiler
)


def run_aws_experiment(instance_type: str = 't2.large'):
    """
    Run binning sensitivity test optimized for AWS instance type.
    
    Args:
        instance_type: AWS instance type (t2.medium, t2.large, t2.xlarge, etc.)
    """
    # Setup memory monitoring
    setup_memory_logging()
    log_memory_usage(f"Starting experiment on {instance_type}")
    
    # Start profiler
    profiler = MemoryProfiler()
    profiler.record("start")
    
    # Configure based on instance type
    configs = {
        't2.medium': BinningTestConfig(
            n_test_points=5,
            fixed_bins=[5, 10, 15],
            test_datasets=['compas'],  # Smallest dataset
            test_backends=['sklearn'],  # Lightest backend
            n_repeat=30,  # Reduce robustness samples
            n_samples_fidelity=500,  # Reduce fidelity samples
            enable_memory_monitoring=True,
            memory_cleanup_threshold_percent=70.0,
            memory_cleanup_after_backend=True,
            memory_checkpoint_interval=2,
            enable_checkpointing=True,
            checkpoint_every_n_configs=3,
            resume_from_checkpoint=True,
        ),
        't2.large': BinningTestConfig(
            n_test_points=10,
            fixed_bins=[5, 10, 15, 20],
            test_datasets=['adult-income', 'compas'],
            test_backends=['sklearn', 'TF2'],
            n_repeat=50,
            enable_memory_monitoring=True,
            memory_cleanup_threshold_percent=75.0,
            memory_cleanup_after_backend=True,
            memory_checkpoint_interval=5,
            enable_checkpointing=True,
            checkpoint_every_n_configs=5,
            resume_from_checkpoint=True,
        ),
        't2.xlarge': BinningTestConfig(
            n_test_points=15,
            fixed_bins=[5, 10, 15, 20],
            test_datasets=['adult-income', 'compas', 'lending-club'],
            test_backends=['sklearn', 'PYT', 'TF2'],
            n_repeat=50,
            enable_memory_monitoring=True,
            memory_cleanup_threshold_percent=80.0,
            memory_cleanup_after_backend=True,
            memory_checkpoint_interval=10,
            enable_checkpointing=True,
            checkpoint_every_n_configs=10,
            resume_from_checkpoint=True,
        ),
        'r5.xlarge': BinningTestConfig(
            n_test_points=20,
            fixed_bins=[5, 10, 15, 20],
            test_datasets=['adult-income', 'compas', 'lending-club', 'german-credit'],
            test_backends=['sklearn', 'PYT', 'TF2'],
            n_repeat=50,
            fidelity_radii=[0.5, 1.0, 2.0],
            n_samples_fidelity=1000,
            enable_memory_monitoring=True,
            memory_cleanup_threshold_percent=80.0,
            memory_cleanup_after_backend=True,
            memory_checkpoint_interval=10,
            enable_checkpointing=True,
            checkpoint_every_n_configs=10,
            resume_from_checkpoint=True,
        ),
    }
    
    # Get config or default to t2.large
    config = configs.get(instance_type, configs['t2.large'])
    
    print("\n" + "="*80)
    print(f"EXPERIMENT CONFIGURATION FOR {instance_type.upper()}")
    print("="*80)
    print(f"Test points:     {config.n_test_points}")
    print(f"Bins:            {config.fixed_bins}")
    print(f"Datasets:        {config.test_datasets}")
    print(f"Backends:        {config.test_backends}")
    print(f"Memory monitor:  {config.enable_memory_monitoring}")
    print(f"Memory threshold: {config.memory_cleanup_threshold_percent}%")
    print(f"Checkpointing:   {config.enable_checkpointing}")
    print(f"Checkpoint every: {config.checkpoint_every_n_configs} configs")
    print(f"Auto-resume:     {config.resume_from_checkpoint}")
    print("="*80 + "\n")
    
    # Run experiment
    try:
        profiler.record("before_experiment")
        
        result = binning_sensitivity_flow(config=config)
        
        profiler.record("after_experiment")
        
        print("\n" + "="*80)
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {result}")
        print("="*80 + "\n")
        
        return result
        
    except MemoryError as e:
        print("\n" + "="*80)
        print("❌ MEMORY ERROR - Instance too small for this configuration")
        print("="*80)
        print(f"Error: {e}")
        print("\nSuggestions:")
        print("1. Use a larger instance type")
        print("2. Reduce n_test_points")
        print("3. Test fewer datasets/backends")
        print("4. Add swap space (see MEMORY_MANAGEMENT.md)")
        print("="*80 + "\n")
        raise
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ EXPERIMENT FAILED")
        print("="*80)
        print(f"Error: {e}")
        print("="*80 + "\n")
        log_memory_usage("Error occurred", level="ERROR")
        raise
        
    finally:
        # Cleanup and save profiling data
        profiler.record("cleanup")
        
        print("\n[Memory] Running final cleanup...")
        clear_session_memory(backend=None, aggressive=True)
        
        log_memory_usage("Final state")
        
        # Save memory profile
        output_dir = Path("experiment_artefacts")
        output_dir.mkdir(exist_ok=True)
        
        profile_csv = output_dir / f"memory_profile_{instance_type}.csv"
        profile_png = output_dir / f"memory_profile_{instance_type}.png"
        
        profiler.save(profile_csv)
        
        try:
            profiler.plot(profile_png)
        except Exception as e:
            print(f"[Warning] Could not save memory plot: {e}")
        
        profiler.summary()


def run_single_dataset(dataset_name: str, backends: list = None):
    """
    Run test for a single dataset (useful for very constrained memory).
    
    Args:
        dataset_name: Dataset to test (e.g., 'compas', 'adult-income')
        backends: List of backends (default: ['sklearn'])
    """
    if backends is None:
        backends = ['sklearn']
    
    setup_memory_logging()
    log_memory_usage(f"Starting single dataset test: {dataset_name}")
    
    config = BinningTestConfig(
        n_test_points=10,
        fixed_bins=[5, 10, 15, 20],
        test_datasets=[dataset_name],
        test_backends=backends,
        enable_memory_monitoring=True,
        memory_cleanup_threshold_percent=75.0,
        memory_cleanup_after_backend=True,
        memory_checkpoint_interval=3,
    )
    
    try:
        result = binning_sensitivity_flow(config=config)
        print(f"\n✅ Completed {dataset_name}")
        return result
    finally:
        clear_session_memory(backend=None, aggressive=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run binning sensitivity test with memory management"
    )
    parser.add_argument(
        '--instance-type',
        type=str,
        default='t2.large',
        choices=['t2.medium', 't2.large', 't2.xlarge', 'r5.xlarge'],
        help='AWS instance type (determines memory configuration)'
    )
    parser.add_argument(
        '--single-dataset',
        type=str,
        help='Run only a single dataset (e.g., compas, adult-income)'
    )
    parser.add_argument(
        '--backends',
        nargs='+',
        default=None,
        help='Backends to test (sklearn, PYT, TF2)'
    )
    
    args = parser.parse_args()
    
    if args.single_dataset:
        print(f"\n🔬 Running single dataset mode: {args.single_dataset}\n")
        run_single_dataset(args.single_dataset, args.backends)
    else:
        print(f"\n🔬 Running full experiment for {args.instance_type}\n")
        run_aws_experiment(args.instance_type)
