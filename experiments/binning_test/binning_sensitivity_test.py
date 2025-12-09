import sys
from pathlib import Path

dice_x_path  = "/Users/volk/Documents/bau24-25/thesis/repos/DiCE-X"

if dice_x_path not in sys.path:
    sys.path.insert(0, dice_x_path)

import dice_ml_x
from dataclasses import dataclass, field
from typing import Optional
from prefect import flow, task, get_run_logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from enum import Enum
import time

from experiments.cf_eval_selected_flow import (
    compute_continuous_proximity,
    compute_categorical_proximity,
    compute_continuous_diversity,
    compute_categorical_diversity,
    compute_sparsity,
    robustness_flip_rate,
    one_nn_fidelity,
)

from experiments.grid_search_experiment import (
    load_datasets,
    load_dice_x_models,
    gen_kwargs_for_method,
    make_dice_objects
)


class BinningMethod(Enum):
    STURGES = 'sturges'
    SCOTT = 'scott'
    FREEDMAN_DIACONIS = 'freedman_diaconis'


def compute_adaptive_bins(data: pd.DataFrame, continuous_features: list,
                          method: BinningMethod=BinningMethod.STURGES):
    """
        Compute adaptive bin count using specified rule.
        Returns maximum across all continuous features.
    """
    n_samples = len(data)
    bin_counts = []

    for feat in continuous_features:
        feat_data = data[feat].values
        range_val = feat_data.max() - feat_data.min()

        if method == BinningMethod.STURGES:
            n_bins = int(np.ceil(np.log2(n_samples) + 1))

        elif method == BinningMethod.SCOTT:
            std = np.std(feat_data)
            if std == 0:
                n_bins = 1
            else:
                h = 3.5 * std / np.cbrt(n_samples)
                n_bins = int(np.ceil(range_val / h))

        elif method == BinningMethod.FREEDMAN_DIACONIS:
            iqr = np.percentile(feat_data, 75) - np.percentile(feat_data, 25)
            if iqr == 0:
                n_bins = 1
            else:
                h = 2 * iqr / np.cbrt(n_samples)
                if h == 0:
                    n_bins = 1
                else:
                    n_bins = max(1, int(np.ceil(range_val / h)))

        bin_counts.append(n_bins)

    final_bins = max(bin_counts)
    return final_bins


@dataclass
class BinningTestConfig:
    n_test_points: int = 10
    lambda_p: float = 1.0
    lambda_d: float = 1.0
    lambda_r: float = 1.0
    fixed_bins: list = field(default_factory=lambda: [5, 10, 15, 20])
    k_cfs: int = 5
    random_seed: int = 42
    test_datasets: list = field(default_factory=lambda: ['adult-income', 'compas',
                                                         'lending-club', 'german-credit'])
    test_backends: list = field(default_factory=lambda: ['sklearn', 'PYT', 'TF2'])
    noise_sd: float = 0.10
    cat_flip_p: float = 0.20
    n_repeat: int = 50
    fidelity_radii: list = field(default_factory=lambda: [0.5, 1.0, 2.0])
    n_samples_fidelity: int = 1000


@dataclass
class DefaultPaths:
    out_dir: Path = Path("experiment_artefacts")
    dice_x_pickle: Optional[Path] = Path("../../docs/source/notebooks/benchmarking_results_23_01_2025-01_05.pkl")


@task
def evaluate_with_bins(
    df: pd.DataFrame,
    target: str, ds_name: str, backend: str, num_bins: int,
    bin_label: str, config: BinningTestConfig, dice_x_models: dict,
):
    """Generate CFs and compute ALL metrics (SEPARATE continuous and categorical)."""
    logger = get_run_logger()

    # Split data
    y = df[target]

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=config.random_seed, stratify=y
    )

    # Sample test points
    test_subset = test_df.sample(
        n=min(config.n_test_points, len(test_df)),
        random_state=config.random_seed
    )

    # Create DiCE objects
    try:
        _, exp, data_iface, _, model_pred = make_dice_objects.fn(
            train_df, target, ds_name, backend, 
            'DiCE-Ext',
            dice_x_models, None
        )
    except Exception as e:
        logger.error(f"Failed to create DiCE objects for {ds_name}/{backend}: {e}")
        return None

    # Generate CFs with fixed lambdas and specified bins
    gen_kwargs = gen_kwargs_for_method.fn(
        method_label='DiCE-Ext', total_cfs=config.k_cfs, lam1=config.lambda_p,
        lam2=config.lambda_d, lam3=config.lambda_r, num_bins=num_bins, backend=backend,
    )

    # Evaluate on test subset
    metrics_list = []
    failed = 0
    rng = np.random.default_rng(config.random_seed)

    for idx in range(len(test_subset)):
        x_query = test_subset.iloc[idx:idx+1].drop(columns=[target])
        x_full = test_subset.iloc[idx:idx+1]

        try:
            start = time.time()
            dice_exp = exp.generate_counterfactuals(x_query, **gen_kwargs)
            gen_time = time.time() - start

            C = dice_exp.to_dataframe()

            if C.empty:
                failed += 1
                continue

            # Fill NaNs with original values
            na_cols = C.columns[C.isna().any()]
            if len(na_cols):
                C[na_cols] = C[na_cols].fillna(x_query[na_cols].iloc[0])

            # ========== COMPUTE ALL METRICS (SEPARATE continuous and categorical) ==========

            # 1. Validity
            validity = exp.get_validity_percentage() / 100.0    # type: ignore

            # 2. Proximity - SEPARATE continuous and categorical (NO aggregation)
            proximity_cont = compute_continuous_proximity(C, x_full, data_iface)
            proximity_cat = compute_categorical_proximity(C, x_full, data_iface)

            # 3. Diversity - SEPARATE continuous and categorical (NO aggregation)
            diversity_cont = compute_continuous_diversity(C, data_iface)
            diversity_cat = compute_categorical_diversity(C, data_iface)

            # 4. Sparsity (continuous only)
            sparsity_cont = compute_sparsity(C, x_full, data_iface)

            # 5. Robustness (flip rate under perturbations)
            robustness = robustness_flip_rate(
                C, target, data_iface, backend, model_pred,
                noise_sd=config.noise_sd, cat_flip_p=config.cat_flip_p,
                n_repeat=config.n_repeat, rng=rng
            )

            # 6. Fidelity (1-NN for multiple radii)
            fidelity_scores = {}
            for radius in config.fidelity_radii:
                fid = one_nn_fidelity(
                    x_query, C, data_iface, model_pred, backend,
                    radius_mad=radius,
                    n_samples=config.n_samples_fidelity,
                    rng=rng
                )
                fidelity_scores[f'fidelity_{radius}'] = fid

            # Store all metrics (SEPARATE, no aggregation)
            point_metrics = {
                'validity': validity,
                'proximity_cont': proximity_cont,
                'proximity_cat': proximity_cat,
                'diversity_cont': diversity_cont,
                'diversity_cat': diversity_cat,
                'sparsity_cont': sparsity_cont,
                'robustness': robustness,
                'generation_time': gen_time,
                'n_cfs': len(C),
            }
            point_metrics.update(fidelity_scores)

            metrics_list.append(point_metrics)

        except Exception as e:
            logger.warning(f"Failed {ds_name}/{backend}/bins={num_bins}, idx={idx}: {str(e)[:100]}")
            failed += 1
            continue

    # Aggregate
    if not metrics_list:
        logger.error(f"All test points failed for {ds_name}/{backend}/bins={num_bins}")
        return None

    df_metrics = pd.DataFrame(metrics_list)

    # Create result dictionary with mean and std for all metrics
    result = {
        'dataset': ds_name,
        'backend': backend,
        'num_bins': num_bins,
        'bin_label': bin_label,
        'n_successful': len(metrics_list),
        'n_failed': failed,
    }

    # Add mean and std for each metric
    for col in df_metrics.columns:
        result[f'{col}_mean'] = df_metrics[col].mean()
        result[f'{col}_std'] = df_metrics[col].std()

    logger.info(f"✓ {ds_name}/{backend}/{bin_label:12s}: "
                f"valid={result['validity_mean']:.3f}, "
                f"prox_cont={result['proximity_cont_mean']:.3f}, "
                f"prox_cat={result['proximity_cat_mean']:.3f}, "
                f"div_cont={result['diversity_cont_mean']:.3f}, "
                f"div_cat={result['diversity_cat_mean']:.3f}, "
                f"robust={result['robustness_mean']:.3f}, "
                f"time={result['generation_time_mean']:.1f}s")

    return result


@task
def save_results(results: list, output_dir: Path) -> Path:
    """Save results to CSV with proper structure."""
    df_results = pd.DataFrame(results)

    # Sort for readability
    df_results = df_results.sort_values(['dataset', 'backend', 'num_bins'])

    csv_file = output_dir / 'binning_sensitivity_results.csv'
    df_results.to_csv(csv_file, index=False)

    # Also save per-dataset CSVs
    for ds_name in df_results['dataset'].unique():
        df_ds = df_results[df_results['dataset'] == ds_name]
        ds_csv = output_dir / f'binning_{ds_name}.csv'
        df_ds.to_csv(ds_csv, index=False)

    # Save per-backend CSVs
    for backend in df_results['backend'].unique():
        df_be = df_results[df_results['backend'] == backend]
        be_csv = output_dir / f'binning_{backend}.csv'
        df_be.to_csv(be_csv, index=False)

    return csv_file


@task
def create_plots(results_csv: Path, output_dir: Path):
    """Create comprehensive visualization plots - SEPARATE continuous and categorical."""
    df = pd.read_csv(results_csv)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    datasets = df['dataset'].unique()
    backends = df['backend'].unique()

    # ===== Plot 1: Core Metrics (Validity, Robustness, Sparsity) =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    core_metrics = [
        ('validity_mean', 'Validity', [0.85, 1.01]),
        ('robustness_mean', 'Robustness (keep rate)', [0, 1.01]),
        ('sparsity_cont_mean', 'Sparsity (continuous)', [0, 1.01]),
    ]

    for metric_idx, (metric, title, ylim) in enumerate(core_metrics):
        ax = axes[metric_idx]

        for backend in backends:
            df_be = df[df['backend'] == backend]

            for ds_name in datasets:
                df_ds = df_be[df_be['dataset'] == ds_name]

                # Separate fixed and adaptive
                fixed_mask = df_ds['num_bins'].isin([5, 10, 15, 20])
                df_fixed = df_ds[fixed_mask].sort_values('num_bins')
                df_adaptive = df_ds[~fixed_mask]

                # Plot fixed bins with error bars
                std_col = metric.replace('_mean', '_std')
                line = ax.errorbar(
                    df_fixed['num_bins'],
                    df_fixed[metric],
                    yerr=df_fixed[std_col] if std_col in df_fixed.columns else None,
                    marker='o',
                    label=f'{ds_name}/{backend}',
                    capsize=4,
                    linewidth=2,
                    alpha=0.8
                )

                color = line.lines[0].get_color()

                # Mark adaptive bins
                for _, row in df_adaptive.iterrows():
                    ax.scatter(
                        row['num_bins'], row[metric], marker='x',
                        s=100, linewidth=2, color=color, zorder=10
                    )

        ax.set_xlabel('Number of Bins', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_dir / 'binning_core_metrics.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'binning_core_metrics.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'binning_core_metrics.eps', format='eps')
    plt.close()

    # ===== Plot 2: Proximity - Continuous vs Categorical (SEPARATE) =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    proximity_metrics = [
        ('proximity_cont_mean', 'Proximity (Continuous)'),
        ('proximity_cat_mean', 'Proximity (Categorical)')
    ]

    for idx, (metric, title) in enumerate(proximity_metrics):
        ax = axes[idx]

        for backend in backends:
            df_be = df[df['backend'] == backend]

            for ds_name in datasets:
                df_ds = df_be[df_be['dataset'] == ds_name]
                fixed_mask = df_ds['num_bins'].isin([5, 10, 15, 20])
                df_fixed = df_ds[fixed_mask].sort_values('num_bins')
                df_adaptive = df_ds[~fixed_mask]

                std_col = metric.replace('_mean', '_std')
                line = ax.errorbar(
                    df_fixed['num_bins'], df_fixed[metric],
                    yerr=df_fixed[std_col] if std_col in df_fixed.columns else None,
                    marker='o', label=f'{ds_name}/{backend}', linewidth=2,
                    capsize=4
                )

                color = line.lines[0].get_color()
                # Mark adaptive bins
                for _, row in df_adaptive.iterrows():
                    ax.scatter(row['num_bins'], row[metric], marker='x',
                               s=100, linewidth=2, color=color, zorder=10)

        ax.set_xlabel('Number of Bins', fontsize=11)
        ax.set_ylabel(title + ' (lower = closer)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'binning_proximity.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'binning_proximity.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'binning_proximity.eps', format='eps')
    plt.close()

    # ===== Plot 3: Diversity - Continuous vs Categorical (SEPARATE) =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    diversity_metrics = [
        ('diversity_cont_mean', 'Diversity (Continuous)'),
        ('diversity_cat_mean', 'Diversity (Categorical)')
    ]

    for idx, (metric, title) in enumerate(diversity_metrics):
        ax = axes[idx]

        for backend in backends:
            df_be = df[df['backend'] == backend]

            for ds_name in datasets:
                df_ds = df_be[df_be['dataset'] == ds_name]
                fixed_mask = df_ds['num_bins'].isin([5, 10, 15, 20])
                df_fixed = df_ds[fixed_mask].sort_values('num_bins')
                df_adaptive = df_ds[~fixed_mask]

                std_col = metric.replace('_mean', '_std')
                line = ax.errorbar(
                    df_fixed['num_bins'], 
                    df_fixed[metric],
                    yerr=df_fixed[std_col] if std_col in df_fixed.columns else None,
                    marker='o',
                    label=f'{ds_name}/{backend}',
                    linewidth=2,
                    capsize=4
                )

                color = line.lines[0].get_color()
                # Mark adaptive bins
                for _, row in df_adaptive.iterrows():
                    ax.scatter(row['num_bins'], row[metric], marker='x',
                               s=100, linewidth=2, color=color, zorder=10)

        ax.set_xlabel('Number of Bins', fontsize=11)
        ax.set_ylabel(title + ' (higher = diverse)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'binning_diversity.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'binning_diversity.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'binning_diversity.eps', format='eps')
    plt.close()

    # ===== Plot 4: Computation Time =====
    fig, ax = plt.subplots(figsize=(10, 6))

    for backend in backends:
        df_be = df[df['backend'] == backend]

        for ds_name in datasets:
            df_ds = df_be[df_be['dataset'] == ds_name]
            fixed_mask = df_ds['num_bins'].isin([5, 10, 15, 20])
            df_fixed = df_ds[fixed_mask].sort_values('num_bins')
            df_adaptive = df_ds[~fixed_mask]

            line = ax.plot(
                df_fixed['num_bins'], df_fixed['generation_time_mean'],
                marker='o', label=f'{ds_name}/{backend}',
                linewidth=2
            )
            color = line[0].get_color()
            # Mark adaptive bins
            for _, row in df_adaptive.iterrows():
                ax.scatter(row['num_bins'], row['generation_time_mean'],
                           marker='x', s=100, linewidth=2, color=color,
                           zorder=10)

    ax.set_xlabel('Number of Bins', fontsize=12)
    ax.set_ylabel('Generation Time (seconds)', fontsize=12)
    ax.set_title('Computation Time vs. Bins', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'binning_time.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'binning_time.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'binning_time.eps', format='eps')
    plt.close()

    # ===== Plot 5: Heatmaps for Each Backend (SEPARATE metrics) =====
    for backend in backends:
        df_be = df[df['backend'] == backend]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        heatmap_metrics = [
            ('validity_mean', 'Validity'),
            ('proximity_cont_mean', 'Proximity (Cont)'),
            ('proximity_cat_mean', 'Proximity (Cat)'),
            ('diversity_cont_mean', 'Diversity (Cont)'),
            ('diversity_cat_mean', 'Diversity (Cat)'),
            ('robustness_mean', 'Robustness')
        ]

        for idx, (metric, title) in enumerate(heatmap_metrics):
            pivot = df_be.pivot_table(
                index='dataset',
                columns='num_bins',
                values=metric
            )

            # Choose colormap: green=good for validity/robustness/diversity, reversed for proximity
            cmap = 'RdYlGn' if metric in ['validity_mean', 'robustness_mean', 'diversity_cont_mean', 'diversity_cat_mean'] else 'RdYlGn_r'

            sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                cbar_kws={'label': title},
                ax=axes[idx]
            )
            axes[idx].set_title(f'{title} - {backend}', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Number of Bins')
            axes[idx].set_ylabel('Dataset')

        plt.tight_layout()
        plt.savefig(output_dir / f'binning_heatmap_{backend}.pdf', bbox_inches='tight')
        plt.savefig(output_dir / f'binning_heatmap_{backend}.png', bbox_inches='tight', dpi=300)
        plt.savefig(output_dir / f'binning_heatmap_{backend}.eps', format='eps')
        plt.close()

    # ===== Plot 6: Fidelity at Multiple Radii =====
    fidelity_cols = [col for col in df.columns if col.startswith('fidelity_') and col.endswith('_mean')]

    if fidelity_cols:
        fig, ax = plt.subplots(figsize=(10, 6))

        for fid_col in fidelity_cols:
            radius = fid_col.replace('fidelity_', '').replace('_mean', '')

            for backend in backends:
                df_be = df[df['backend'] == backend]
                avg_fidelity = df_be.groupby('num_bins')[fid_col].mean()

                ax.plot(
                    avg_fidelity.index,
                    avg_fidelity.values,
                    marker='o',
                    label=f'radius={radius}, {backend}',
                    linewidth=2
                )

        ax.set_xlabel('Number of Bins', fontsize=12)
        ax.set_ylabel('1-NN Fidelity', fontsize=12)
        ax.set_title('Fidelity vs. Bins (Multiple Radii)', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'binning_fidelity.pdf', bbox_inches='tight')
        plt.savefig(output_dir / 'binning_fidelity.png', bbox_inches='tight', dpi=300)
        plt.savefig(output_dir / 'binning_fidelity.eps', format='eps')
        plt.close()

    logger = get_run_logger()
    logger.info(f"⚙️ Created 6 comprehensive plots in {output_dir}/")


@task
def create_summary_table(results_csv: Path, output_dir: Path):
    """Create a comprehensive summary table - SEPARATE continuous and categorical."""
    df = pd.read_csv(results_csv)

    # Group by bins and compute statistics for key metrics (SEPARATE)
    key_metrics = [
        'validity_mean',
        'proximity_cont_mean',
        'proximity_cat_mean',
        'diversity_cont_mean',
        'diversity_cat_mean',
        'sparsity_cont_mean',
        'robustness_mean',
        'generation_time_mean',
    ]

    summary = df.groupby(['num_bins', 'bin_label'])[key_metrics].agg(['mean', 'std']).round(3)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Add interpretation column
    def interpret_bins(row):
        if row['num_bins'] == 10:
            return 'Recommended'
        elif row['num_bins'] < 10:
            return 'Under-discretized'
        elif row['num_bins'] <= 15:
            return 'Acceptable'
        else:
            return 'Over-discretized'

    summary['interpretation'] = summary.apply(interpret_bins, axis=1)

    # Save as CSV
    summary_csv = output_dir / 'binning_summary_table.csv'
    summary.to_csv(summary_csv, index=False)

    # Also save as LaTeX
    latex_file = output_dir / 'binning_summary_table.tex'
    with open(latex_file, 'w') as f:
        f.write(summary.to_latex(index=False, float_format='%.3f'))

    logger = get_run_logger()
    logger.info(f"⚙️ Created summary table: {summary_csv}")

    return summary_csv


@flow(name="Binning Sensitivity Analysis")
def binning_sensitivity_flow(
    config: BinningTestConfig = BinningTestConfig(),
    output_dir: Path = DefaultPaths.out_dir
):
    """
    Run comprehensive binning sensitivity analysis.
    Evaluates ALL metrics SEPARATELY (no aggregation of continuous + categorical).
    """
    logger = get_run_logger()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load resources
    logger.info("Loading datasets and DiCE-Extended models...")
    datasets = load_datasets()
    dice_x_models = load_dice_x_models.fn(DefaultPaths.dice_x_pickle)    # type: ignore
    
    # Filter to test datasets
    datasets = [d for d in datasets if d[2] in config.test_datasets]
    
    # Compute adaptive bins
    logger.info("\n=== Computing Adaptive Bin Counts ===")
    adaptive_bins = {}
    
    for df, target, ds_name in datasets:
        cont_features = df.select_dtypes(include=[np.number]).columns.difference([target]).tolist()
        
        sturges = compute_adaptive_bins(df, cont_features, BinningMethod.STURGES)
        scott = compute_adaptive_bins(df, cont_features, BinningMethod.SCOTT)
        fd = compute_adaptive_bins(df, cont_features, BinningMethod.FREEDMAN_DIACONIS)
        
        adaptive_bins[ds_name] = {
            'sturges': sturges,
            'scott': scott,
            'fd': fd,
        }
        
        logger.info(f"{ds_name}: Sturges={sturges}, Scott={scott}, FD={fd}")
    
    # Save adaptive bins info
    adaptive_df = pd.DataFrame([
        {'dataset': ds, 'method': method, 'num_bins': bins}
        for ds, methods in adaptive_bins.items()
        for method, bins in methods.items()
    ])
    adaptive_df.to_csv(output_dir / 'adaptive_bins_computed.csv', index=False)
    
    # Run experiments
    logger.info("\n=== Running Binning Experiments ===")
    logger.info(f"Metrics (SEPARATE): validity, proximity_cont, proximity_cat, diversity_cont, diversity_cat, sparsity_cont, robustness, fidelity, time")
    logger.info(f"Testing {len(config.test_backends)} backends: {config.test_backends}")
    logger.info(f"Testing {len(datasets)} datasets: {[d[2] for d in datasets]}")
    
    total_configs = len(datasets) * len(config.test_backends) * (len(config.fixed_bins) + 3)
    logger.info(f"Total configurations: {total_configs}")
    
    results = []
    completed = 0
    
    for df, target, ds_name in datasets:
        for backend in config.test_backends:
            # Test fixed bins
            for num_bins in config.fixed_bins:
                result = evaluate_with_bins(
                    df, target, ds_name, backend, num_bins,
                    f"fixed_{num_bins}", config, dice_x_models
                )
                if result:
                    results.append(result)
                completed += 1
                logger.info(f"Progress: {completed}/{total_configs} ({100*completed/total_configs:.1f}%)")
            
            # Test adaptive bins
            for method_name, num_bins in adaptive_bins[ds_name].items():
                result = evaluate_with_bins(
                    df, target, ds_name, backend, num_bins,
                    method_name, config, dice_x_models
                )
                if result:
                    results.append(result)
                completed += 1
                logger.info(f"Progress: {completed}/{total_configs} ({100*completed/total_configs:.1f}%)")
    
    # Save and visualize
    if results:
        csv_file = save_results(results, output_dir)
        logger.info(f"\n✓ Saved results to {csv_file}")
        plot_path: Path = output_dir / "chart_artefacts"
        plot_path.mkdir(parents=True, exist_ok=True)
        create_plots(csv_file, plot_path)
        create_summary_table(csv_file, output_dir)

        # Print final summary
        logger.info("\n" + "="*60)
        logger.info("=== BINNING SENSITIVITY ANALYSIS COMPLETE ===")
        logger.info("="*60)
        logger.info(f"\nResults saved to: {output_dir}/")
        logger.info("  - Main results: binning_sensitivity_results.csv")
        logger.info("    (proximity_cont, proximity_cat, diversity_cont, diversity_cat SEPARATE)")
        logger.info("  - Per-dataset CSVs: binning_<dataset>.csv")
        logger.info("  - Per-backend CSVs: binning_<backend>.csv")
        logger.info("  - Summary table: binning_summary_table.csv")
        logger.info("  - 6 comprehensive plots (*.pdf and *.png)")
        
        return csv_file
    else:
        logger.error("No results generated!")
        return None


if __name__ == "__main__":
    # Quick test first
    config = BinningTestConfig(
        n_test_points=10,
        fixed_bins=[5, 10, 15, 20],
        test_datasets=['adult-income', 'german-credit', 'lending-club', 'compas'],
        test_backends=['sklearn', 'PYT', 'TF2'],
    )

    binning_sensitivity_flow(config=config)
