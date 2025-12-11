"""
Test checkpoint/resume functionality.

Tests that checkpoints are saved correctly and can be loaded to resume experiments.
"""

import sys
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from binning_sensitivity_test import (
    load_checkpoint,
    save_checkpoint,
    BinningTestConfig
)


def test_checkpoint_save_and_load():
    """Test basic checkpoint save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Create sample results
        results = [
            {
                'dataset': 'adult-income',
                'backend': 'sklearn',
                'num_bins': 10,
                'bin_method': 'fixed_10',
                'validity_mean': 0.95,
            },
            {
                'dataset': 'adult-income',
                'backend': 'sklearn',
                'num_bins': 20,
                'bin_method': 'fixed_20',
                'validity_mean': 0.97,
            },
            {
                'dataset': 'compas',
                'backend': 'TF2',
                'num_bins': 10,
                'bin_method': 'fixed_10',
                'validity_mean': 0.92,
            },
        ]
        
        # Save checkpoint
        save_checkpoint(results, output_dir)
        
        # Verify files exist
        checkpoint_file = output_dir / 'checkpoint_results.csv'
        timestamp_file = output_dir / 'checkpoint_timestamp.txt'
        
        assert checkpoint_file.exists(), "Checkpoint CSV should exist"
        assert timestamp_file.exists(), "Timestamp file should exist"
        
        # Load checkpoint
        loaded_results, completed_configs = load_checkpoint(output_dir)
        
        # Verify loaded data
        assert len(loaded_results) == 3, f"Should load 3 results, got {len(loaded_results)}"
        assert len(completed_configs) == 3, f"Should have 3 completed configs, got {len(completed_configs)}"
        
        # Verify config keys
        expected_keys = {
            'adult-income|sklearn|10',
            'adult-income|sklearn|20',
            'compas|TF2|10',
        }
        assert completed_configs == expected_keys, f"Config keys mismatch: {completed_configs} != {expected_keys}"
        
        print("✓ Checkpoint save/load test passed")


def test_checkpoint_resume_logic():
    """Test that resume logic correctly skips completed configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Simulate partial completion
        initial_results = [
            {'dataset': 'adult-income', 'backend': 'sklearn', 'num_bins': 5, 'validity_mean': 0.9},
            {'dataset': 'adult-income', 'backend': 'sklearn', 'num_bins': 10, 'validity_mean': 0.95},
        ]
        save_checkpoint(initial_results, output_dir)
        
        # Load checkpoint
        results, completed_configs = load_checkpoint(output_dir)
        
        # Define what would be run
        all_configs = [
            ('adult-income', 'sklearn', 5),
            ('adult-income', 'sklearn', 10),
            ('adult-income', 'sklearn', 15),  # Not completed yet
            ('adult-income', 'sklearn', 20),  # Not completed yet
        ]
        
        # Simulate resume logic
        skipped = []
        to_run = []
        
        for dataset, backend, num_bins in all_configs:
            config_key = f"{dataset}|{backend}|{num_bins}"
            if config_key in completed_configs:
                skipped.append(config_key)
            else:
                to_run.append(config_key)
        
        # Verify
        assert len(skipped) == 2, f"Should skip 2 configs, skipped {len(skipped)}"
        assert len(to_run) == 2, f"Should run 2 configs, got {len(to_run)}"
        assert 'adult-income|sklearn|5' in skipped
        assert 'adult-income|sklearn|10' in skipped
        assert 'adult-income|sklearn|15' in to_run
        assert 'adult-income|sklearn|20' in to_run
        
        print("✓ Resume logic test passed")


def test_checkpoint_incremental_save():
    """Test incremental checkpoint saves."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        results = []
        checkpoint_interval = 3
        configs_since_checkpoint = 0
        
        # Simulate running 10 configs with checkpointing every 3
        for i in range(10):
            # Add result
            results.append({
                'dataset': 'test',
                'backend': 'sklearn',
                'num_bins': i,
                'validity_mean': 0.9 + i * 0.01,
            })
            configs_since_checkpoint += 1
            
            # Save checkpoint
            if configs_since_checkpoint >= checkpoint_interval:
                save_checkpoint(results, output_dir)
                configs_since_checkpoint = 0
                
                # Verify checkpoint has correct number of results
                loaded, _ = load_checkpoint(output_dir)
                assert len(loaded) == len(results), \
                    f"Checkpoint should have {len(results)} results, got {len(loaded)}"
        
        # Save final checkpoint (to capture last 1 config: 9%3 = 0)
        save_checkpoint(results, output_dir)
        
        # Final state: should have 10 results
        final_results, final_configs = load_checkpoint(output_dir)
        assert len(final_results) == 10, f"Should have 10 results, got {len(final_results)}"
        
        print("✓ Incremental save test passed")


def test_checkpoint_no_file():
    """Test loading when no checkpoint exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Load from empty directory
        results, completed = load_checkpoint(output_dir)
        
        assert results == [], "Should return empty list"
        assert completed == set(), "Should return empty set"
        
        print("✓ No checkpoint file test passed")


def test_checkpoint_config_defaults():
    """Test that config has correct checkpoint defaults."""
    config = BinningTestConfig()
    
    assert config.enable_checkpointing == True, "Checkpointing should be enabled by default"
    assert config.checkpoint_every_n_configs == 5, "Default checkpoint interval should be 5"
    assert config.resume_from_checkpoint == True, "Resume should be enabled by default"
    
    print("✓ Config defaults test passed")


if __name__ == '__main__':
    print("="*80)
    print("CHECKPOINT SYSTEM TESTS")
    print("="*80)
    
    test_checkpoint_save_and_load()
    test_checkpoint_resume_logic()
    test_checkpoint_incremental_save()
    test_checkpoint_no_file()
    test_checkpoint_config_defaults()
    
    print("\n" + "="*80)
    print("ALL CHECKPOINT TESTS PASSED ✓")
    print("="*80)
