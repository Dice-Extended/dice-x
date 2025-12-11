"""
Memory Management Utilities for Experiment Flows
=================================================

Provides tools for monitoring and managing memory usage during long-running
experiments on cloud instances (AWS, etc.). Includes garbage collection,
memory profiling, automatic cleanup, and emergency handling.

Usage:
    from experiments.memory_management import (
        MemoryMonitor,
        memory_checkpoint,
        clear_session_memory,
        log_memory_usage
    )

    # Option 1: Decorator for tasks
    @task
    @memory_checkpoint(threshold_mb=1000)
    def my_task():
        pass

    # Option 2: Context manager
    with MemoryMonitor(check_interval=10) as monitor:
        # Your code here
        pass

    # Option 3: Manual cleanup
    clear_session_memory(backend='TF2')
    log_memory_usage("After cleanup")
"""

import gc
import os
import psutil
import logging
from typing import Optional, Callable, Any, Literal
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_mb: float
    available_mb: float
    used_mb: float
    percent: float
    process_mb: float
    
    def __str__(self):
        return (f"Memory: {self.used_mb:.1f}/{self.total_mb:.1f} MB "
                f"({self.percent:.1f}%), Process: {self.process_mb:.1f} MB")


def get_memory_stats() -> MemoryStats:
    """
    Get current memory statistics.
    
    Returns:
        MemoryStats object with system and process memory info
    """
    # System memory
    mem = psutil.virtual_memory()
    
    # Current process memory
    process = psutil.Process(os.getpid())
    process_mem = process.memory_info().rss / 1024 / 1024  # MB
    
    return MemoryStats(
        total_mb=mem.total / 1024 / 1024,
        available_mb=mem.available / 1024 / 1024,
        used_mb=mem.used / 1024 / 1024,
        percent=mem.percent,
        process_mb=process_mem
    )


def log_memory_usage(context: str = "", level: str = "INFO"):
    """
    Log current memory usage with context.
    
    Args:
        context: Description of when this is being called
        level: Logging level (INFO, WARNING, ERROR)
    """
    stats = get_memory_stats()
    msg = f"[Memory] {context}: {stats}"
    
    if level == "WARNING":
        logger.warning(msg)
    elif level == "ERROR":
        logger.error(msg)
    else:
        logger.info(msg)


def clear_session_memory(
    backend: Optional[Literal['TF2', 'PYT', 'sklearn']] = None,
    aggressive: bool = False
):
    """
    Clear memory by running garbage collection and clearing backend sessions.
    
    Args:
        backend: ML backend to clear ('TF2', 'PYT', 'sklearn', or None for all)
        aggressive: If True, perform more aggressive cleanup
    """
    logger.info(f"Clearing memory (backend={backend}, aggressive={aggressive})...")
    
    # Python garbage collection
    collected = gc.collect()
    logger.debug(f"Garbage collection freed {collected} objects")
    
    # Backend-specific cleanup
    if backend in [None, 'TF2']:
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            logger.debug("Cleared TensorFlow session")
            
            if aggressive:
                # Reset default graph (TF 1.x compatibility)
                try:
                    tf.compat.v1.reset_default_graph()
                except AttributeError:
                    pass
        except ImportError:
            pass
    
    if backend in [None, 'PYT']:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared PyTorch CUDA cache")
        except ImportError:
            pass
    
    if aggressive:
        # Additional garbage collection passes
        for _ in range(2):
            gc.collect()
        
        # Force collection of generation 2 objects
        gc.collect(2)
    
    log_memory_usage("After cleanup", level="INFO")


class MemoryMonitor:
    """
    Context manager for monitoring memory usage during execution.
    
    Automatically logs memory at start/end and optionally checks during execution.
    Can trigger cleanup if memory usage exceeds threshold.
    
    Example:
        with MemoryMonitor(threshold_percent=80, auto_cleanup=True) as monitor:
            # Your code here
            monitor.checkpoint("After processing batch")
    """
    
    def __init__(
        self,
        name: str = "Operation",
        threshold_percent: Optional[float] = None,
        threshold_mb: Optional[float] = None,
        auto_cleanup: bool = False,
        check_interval: Optional[int] = None,
        backend: Optional[str] = None
    ):
        """
        Initialize memory monitor.
        
        Args:
            name: Name of the operation being monitored
            threshold_percent: Memory usage % that triggers warning/cleanup
            threshold_mb: Process memory MB that triggers warning/cleanup
            auto_cleanup: Automatically run cleanup when threshold exceeded
            check_interval: If set, check memory every N iterations (use with checkpoint())
            backend: Backend to clear if cleanup is triggered
        """
        self.name = name
        self.threshold_percent = threshold_percent
        self.threshold_mb = threshold_mb
        self.auto_cleanup = auto_cleanup
        self.check_interval = check_interval
        self.backend = backend
        
        self.start_stats: Optional[MemoryStats] = None
        self.checkpoint_count = 0
        
    def __enter__(self):
        """Enter context - log starting memory."""
        self.start_stats = get_memory_stats()
        logger.info(f"[Memory] Starting '{self.name}': {self.start_stats}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - log final memory and delta."""
        end_stats = get_memory_stats()
        
        if self.start_stats:
            delta_mb = end_stats.process_mb - self.start_stats.process_mb
            logger.info(
                f"[Memory] Finished '{self.name}': {end_stats} "
                f"(Δ Process: {delta_mb:+.1f} MB)"
            )
        else:
            logger.info(f"[Memory] Finished '{self.name}': {end_stats}")
        
        return False  # Don't suppress exceptions
    
    def checkpoint(self, label: str = ""):
        """
        Log memory at a checkpoint and optionally trigger cleanup.
        
        Args:
            label: Description of the checkpoint
        """
        self.checkpoint_count += 1
        
        # Check if we should monitor this checkpoint
        should_check = (
            self.check_interval is None or 
            self.checkpoint_count % self.check_interval == 0
        )
        
        if not should_check:
            return
        
        stats = get_memory_stats()
        
        # Determine if threshold exceeded
        threshold_exceeded = False
        reason = ""
        
        if self.threshold_percent and stats.percent > self.threshold_percent:
            threshold_exceeded = True
            reason = f"System memory {stats.percent:.1f}% > {self.threshold_percent}%"
        
        if self.threshold_mb and stats.process_mb > self.threshold_mb:
            threshold_exceeded = True
            reason = f"Process memory {stats.process_mb:.1f} MB > {self.threshold_mb} MB"
        
        # Log
        level = "WARNING" if threshold_exceeded else "INFO"
        context = f"'{self.name}' checkpoint #{self.checkpoint_count}"
        if label:
            context += f" ({label})"
        log_memory_usage(context, level=level)
        
        # Auto cleanup if enabled
        if threshold_exceeded:
            if reason:
                logger.warning(f"[Memory] Threshold exceeded: {reason}")
            
            if self.auto_cleanup:
                logger.info("[Memory] Auto-cleanup triggered")
                clear_session_memory(backend=self.backend, aggressive=True)


def memory_checkpoint(
    threshold_percent: Optional[float] = None,
    threshold_mb: Optional[float] = None,
    auto_cleanup: bool = True,
    backend: Optional[str] = None
):
    """
    Decorator for functions/tasks that automatically monitors memory.
    Runs cleanup before and after if thresholds exceeded.
    
    Args:
        threshold_percent: System memory % threshold
        threshold_mb: Process memory MB threshold
        auto_cleanup: Run cleanup if threshold exceeded
        backend: Backend to clear during cleanup
    
    Example:
        @task
        @memory_checkpoint(threshold_mb=1000, backend='TF2')
        def process_batch(data):
            # Your code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Check memory before
            before_stats = get_memory_stats()
            logger.info(f"[Memory] Before {func_name}: {before_stats}")
            
            # Check thresholds before
            should_cleanup_before = False
            if threshold_percent and before_stats.percent > threshold_percent:
                logger.warning(
                    f"[Memory] Pre-execution cleanup: "
                    f"memory at {before_stats.percent:.1f}% > {threshold_percent}%"
                )
                should_cleanup_before = True
            
            if threshold_mb and before_stats.process_mb > threshold_mb:
                logger.warning(
                    f"[Memory] Pre-execution cleanup: "
                    f"process at {before_stats.process_mb:.1f} MB > {threshold_mb} MB"
                )
                should_cleanup_before = True
            
            if should_cleanup_before and auto_cleanup:
                clear_session_memory(backend=backend, aggressive=True)
            
            # Execute function
            try:
                result = func(*args, **kwargs)
            finally:
                # Check memory after
                after_stats = get_memory_stats()
                delta_mb = after_stats.process_mb - before_stats.process_mb
                
                logger.info(
                    f"[Memory] After {func_name}: {after_stats} "
                    f"(Δ: {delta_mb:+.1f} MB)"
                )
                
                # Cleanup after if needed
                should_cleanup_after = False
                if threshold_percent and after_stats.percent > threshold_percent:
                    should_cleanup_after = True
                if threshold_mb and after_stats.process_mb > threshold_mb:
                    should_cleanup_after = True
                
                if should_cleanup_after and auto_cleanup:
                    logger.info("[Memory] Post-execution cleanup")
                    clear_session_memory(backend=backend, aggressive=True)
            
            return result
        
        return wrapper
    return decorator


@contextmanager
def memory_limited_batch(
    batch_size: int,
    max_memory_percent: float = 80.0,
    min_batch_size: int = 1,
    backend: Optional[str] = None
):
    """
    Context manager that suggests batch size reduction if memory is high.
    
    Args:
        batch_size: Initial batch size
        max_memory_percent: Maximum acceptable memory %
        min_batch_size: Minimum batch size (won't go below this)
        backend: Backend to cleanup if needed
    
    Yields:
        Adjusted batch size
    
    Example:
        for i in range(0, len(data), batch_size):
            with memory_limited_batch(batch_size, max_memory_percent=75) as actual_batch:
                batch = data[i:i+actual_batch]
                process(batch)
    """
    stats = get_memory_stats()
    
    # Adjust batch size based on memory
    adjusted_batch = batch_size
    
    if stats.percent > max_memory_percent:
        logger.warning(
            f"[Memory] High memory usage ({stats.percent:.1f}%), "
            f"cleaning up before batch"
        )
        clear_session_memory(backend=backend, aggressive=True)
        
        # Suggest smaller batch
        reduction_factor = max_memory_percent / stats.percent
        adjusted_batch = max(min_batch_size, int(batch_size * reduction_factor * 0.8))
        
        if adjusted_batch < batch_size:
            logger.warning(
                f"[Memory] Reducing batch size {batch_size} → {adjusted_batch} "
                f"due to memory pressure"
            )
    
    yield adjusted_batch


class MemoryProfiler:
    """
    Track memory usage over time for profiling and debugging.
    
    Example:
        profiler = MemoryProfiler()
        
        for i in range(100):
            # Do work
            profiler.record(f"iteration_{i}")
        
        profiler.summary()
        profiler.save("memory_profile.csv")
    """
    
    def __init__(self):
        self.records = []
        self.start_time = time.time()
    
    def record(self, label: str = ""):
        """Record current memory usage."""
        stats = get_memory_stats()
        elapsed = time.time() - self.start_time
        
        self.records.append({
            'timestamp': elapsed,
            'label': label,
            'system_used_mb': stats.used_mb,
            'system_percent': stats.percent,
            'process_mb': stats.process_mb,
        })
    
    def summary(self):
        """Print summary statistics."""
        if not self.records:
            logger.info("[MemoryProfiler] No records")
            return
        
        import pandas as pd
        df = pd.DataFrame(self.records)
        
        logger.info("\n" + "="*80)
        logger.info("MEMORY PROFILE SUMMARY")
        logger.info("="*80)
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Duration: {df['timestamp'].max():.1f} seconds")
        logger.info(f"\nProcess Memory (MB):")
        logger.info(f"  Min:  {df['process_mb'].min():.1f}")
        logger.info(f"  Max:  {df['process_mb'].max():.1f}")
        logger.info(f"  Mean: {df['process_mb'].mean():.1f}")
        logger.info(f"  Std:  {df['process_mb'].std():.1f}")
        logger.info(f"\nSystem Memory (%):")
        logger.info(f"  Min:  {df['system_percent'].min():.1f}%")
        logger.info(f"  Max:  {df['system_percent'].max():.1f}%")
        logger.info(f"  Mean: {df['system_percent'].mean():.1f}%")
        logger.info("="*80 + "\n")
    
    def save(self, filepath: str):
        """Save records to CSV."""
        import pandas as pd
        df = pd.DataFrame(self.records)
        df.to_csv(filepath, index=False)
        logger.info(f"[MemoryProfiler] Saved to {filepath}")
    
    def plot(self, filepath: Optional[str] = None):
        """
        Plot memory usage over time.
        
        Args:
            filepath: If provided, save plot to this file
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        
        if not self.records:
            logger.warning("[MemoryProfiler] No records to plot")
            return
        
        df = pd.DataFrame(self.records)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Process memory
        axes[0].plot(df['timestamp'], df['process_mb'], marker='o', markersize=3)
        axes[0].set_ylabel('Process Memory (MB)')
        axes[0].set_title('Memory Usage Over Time')
        axes[0].grid(alpha=0.3)
        
        # System memory percentage
        axes[1].plot(df['timestamp'], df['system_percent'], marker='o', markersize=3, color='orange')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('System Memory (%)')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"[MemoryProfiler] Plot saved to {filepath}")
        else:
            plt.show()
        
        plt.close()


def emergency_cleanup():
    """
    Emergency memory cleanup when system is critically low.
    Runs the most aggressive cleanup possible.
    """
    logger.error("[Memory] EMERGENCY CLEANUP INITIATED")
    
    stats_before = get_memory_stats()
    logger.error(f"[Memory] Before emergency cleanup: {stats_before}")
    
    # Multiple aggressive GC passes
    for i in range(3):
        collected = gc.collect(2)
        logger.debug(f"[Memory] Emergency GC pass {i+1}: freed {collected} objects")
    
    # Clear all backend sessions
    clear_session_memory(backend=None, aggressive=True)
    
    # Force additional cleanup
    import sys
    if hasattr(sys, 'modules'):
        # Clear module cache (careful - may break things)
        import importlib
        for module_name in list(sys.modules.keys()):
            if 'test' in module_name.lower() or 'temp' in module_name.lower():
                try:
                    del sys.modules[module_name]
                except:
                    pass
    
    stats_after = get_memory_stats()
    freed_mb = stats_before.process_mb - stats_after.process_mb
    
    logger.error(
        f"[Memory] After emergency cleanup: {stats_after} "
        f"(freed {freed_mb:.1f} MB)"
    )


# Convenience function for setting up memory logging
def setup_memory_logging(level: int = logging.INFO):
    """
    Configure logging for memory management module.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s')
    )
    logger.addHandler(handler)
    logger.setLevel(level)
