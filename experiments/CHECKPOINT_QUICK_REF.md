# Checkpoint System - Quick Reference

## Enable Checkpointing

```python
config = BinningTestConfig(
    enable_checkpointing=True,
    checkpoint_every_n_configs=5,
    resume_from_checkpoint=True,
)
```

## Files Created

- `checkpoint_results.csv` - Incremental results (deleted on success)
- `checkpoint_timestamp.txt` - Progress tracking (deleted on success)
- `checkpoint_results_completed.csv` - Final archive (kept after success)

## Auto-Resume

```bash
# First run - gets interrupted
uv run run_aws_experiment.py --instance-type r5.xlarge

# Just run again - automatically resumes
uv run run_aws_experiment.py --instance-type r5.xlarge
```

## Check Progress

```bash
# Count completed configs
wc -l experiment_artefacts/checkpoint_results.csv

# View timestamp
cat experiment_artefacts/checkpoint_timestamp.txt
```

## Force Fresh Start

```bash
rm experiment_artefacts/checkpoint_results.csv
rm experiment_artefacts/checkpoint_timestamp.txt
```

## Instance-Specific Settings

| Instance | RAM | Checkpoint Every N |
|----------|-----|-------------------|
| t2.medium | 4GB | 3 configs |
| t2.large | 8GB | 5 configs |
| t2.xlarge | 16GB | 10 configs |
| r5.xlarge | 32GB | 10 configs |

## Checkpoint Logs

Look for these in output:

```
=== Checking for Existing Checkpoint ===
✓ Loaded 42 results from checkpoint
✓ Resuming from 42 completed configurations
Total configurations: 84
Already completed: 42
Remaining: 42

⏭️  Skipping adult-income/sklearn/bins=10 (already completed)
💾 Saving checkpoint (47 results)...
✓ Archived checkpoint file (experiment completed successfully)
```

## Common Commands

```bash
# Check if checkpoint exists
ls -lh experiment_artefacts/checkpoint_*

# View recent checkpoints
tail -20 experiment_artefacts/checkpoint_results.csv

# Monitor progress during run (in another terminal)
watch -n 60 'wc -l experiment_artefacts/checkpoint_results.csv'

# Backup checkpoint
cp experiment_artefacts/checkpoint_results.csv checkpoint_backup_$(date +%Y%m%d_%H%M).csv
```

## Failure Recovery

1. **Instance terminated** → Just restart, auto-resumes
2. **Out of memory** → Reduce config, restart, auto-resumes
3. **Code error** → Fix bug, restart, auto-resumes
4. **Network disconnect** → Use `screen` or `nohup`, can reconnect

## Overhead

- Checkpoint save: ~50ms per checkpoint
- Total overhead: <0.1% of experiment time
- No performance impact on computation
