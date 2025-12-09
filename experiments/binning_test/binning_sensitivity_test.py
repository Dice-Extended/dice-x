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


def compute_pareto_ranks(df, objectives):
    """
    Compute Pareto ranks using iterative fronts:
    - Rank 1 = non-dominated (Pareto frontier)
    - Rank 2 = dominated only by rank 1
    - etc.

    Args:
        df: DataFrame with objective columns
        objectives: dict mapping column names to 'maximize' or 'minimize'

    Returns:
        np.array of ranks (1-indexed)
    """
    # Prepare objectives matrix (convert all to maximization)
    obj_matrix = np.zeros((len(df), len(objectives)))

    for idx, (col, direction) in enumerate(objectives.items()):
        if direction == 'minimize':
            obj_matrix[:, idx] = -df[col].values  # negate to maximize
        else:
            obj_matrix[:, idx] = df[col].values

    ranks = np.zeros(len(df), dtype=int)
    remaining = list(range(len(df)))
    current_rank = 1

    while remaining:
        # Find non-dominated solutions in remaining set
        pareto_front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                # Check if j dominates i (better or equal on all, strictly better on at least one)
                if np.all(obj_matrix[j] >= obj_matrix[i]) and np.any(obj_matrix[j] > obj_matrix[i]):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(i)

        # Assign rank to current front
        for i in pareto_front:
            ranks[i] = current_rank
            remaining.remove(i)

        current_rank += 1

    return ranks


def select_best_from_pareto(pareto_df, objectives):
    """
    From Pareto-optimal set, select solution closest to ideal point.

    Ideal point defined as:
    - validity = 1.0 (perfect validity)
    - proximity = min observed (closest to query)
    - diversity = max observed (most diverse)
    - robustness = 1.0 (perfect robustness)

    Args:
        pareto_df: DataFrame containing only Pareto-optimal solutions
        objectives: dict mapping column names to 'maximize' or 'minimize'

    Returns:
        Single row (Series) representing the best solution
    """
    if len(pareto_df) == 0:
        return None

    if len(pareto_df) == 1:
        return pareto_df.iloc[0]

    # Define ideal point for each objective
    ideal = {}
    for col, direction in objectives.items():
        if col == 'validity_mean_mean':
            ideal[col] = 1.0  # perfect validity
        elif col == 'robustness_mean_mean':
            ideal[col] = 1.0  # perfect robustness
        elif direction == 'minimize':
            ideal[col] = pareto_df[col].min()  # best observed value
        else:  # maximize
            ideal[col] = pareto_df[col].max()  # best observed value

    # Compute normalized Euclidean distance to ideal point
    distances = []
    for _, row in pareto_df.iterrows():
        dist_sq = 0
        for col, direction in objectives.items():
            # Get normalized objective value [0, 1]
            min_val = pareto_df[col].min()
            max_val = pareto_df[col].max()

            if max_val == min_val:
                obj_norm = 1.0
            else:
                obj_norm = (row[col] - min_val) / (max_val - min_val)

            # Get normalized ideal value [0, 1]
            if max_val == min_val:
                ideal_norm = 1.0
            else:
                ideal_norm = (ideal[col] - min_val) / (max_val - min_val)

            # For minimization objectives, invert the scale
            if direction == 'minimize':
                obj_norm = 1.0 - obj_norm
                ideal_norm = 1.0 - ideal_norm

            dist_sq += (obj_norm - ideal_norm) ** 2

        distances.append(np.sqrt(dist_sq))

    pareto_df = pareto_df.copy()
    pareto_df['distance_to_ideal'] = distances

    return pareto_df.loc[pareto_df['distance_to_ideal'].idxmin()]


@task
def create_summary_table(results_csv: Path, output_dir: Path):
    """
    Create comprehensive summary table using Pareto dominance analysis.
    NO arbitrary weights - uses multi-objective optimization principles.
    """
    logger = get_run_logger()
    df = pd.read_csv(results_csv)

    # Group by bins and compute statistics for key metrics
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

    # ========== PARETO DOMINANCE ANALYSIS (NO ARBITRARY WEIGHTS) ==========

    # Define objectives for Pareto analysis
    objectives = {
        'validity_mean_mean': 'maximize',
        'proximity_cont_mean_mean': 'minimize',  # lower is better
        'diversity_cont_mean_mean': 'maximize',
        'robustness_mean_mean': 'maximize',
    }

    logger.info("\n" + "="*80)
    logger.info("PARETO DOMINANCE ANALYSIS")
    logger.info("="*80)
    logger.info("Objectives:")
    for obj, direction in objectives.items():
        logger.info(f"  - {obj}: {direction}")

    # Compute Pareto ranks
    summary['pareto_rank'] = compute_pareto_ranks(summary, objectives)

    # Identify Pareto-optimal solutions (rank 1)
    pareto_optimal = summary[summary['pareto_rank'] == 1].copy()
    summary['is_pareto_optimal'] = summary['pareto_rank'] == 1

    logger.info(f"\nFound {len(pareto_optimal)} Pareto-optimal configurations:")
    for _, row in pareto_optimal.iterrows():
        logger.info(f"  - {row['bin_label']:15s} ({int(row['num_bins']):2d} bins): "
                   f"validity={row['validity_mean_mean']:.3f}, "
                   f"proximity={row['proximity_cont_mean_mean']:.3f}, "
                   f"diversity={row['diversity_cont_mean_mean']:.3f}, "
                   f"robustness={row['robustness_mean_mean']:.3f}")

    # Select best from Pareto frontier (closest to ideal point)
    if len(pareto_optimal) > 0:
        best_config = select_best_from_pareto(pareto_optimal, objectives)
        summary['is_recommended'] = False
        summary.loc[summary['num_bins'] == best_config['num_bins'], 'is_recommended'] = True

        logger.info("\n" + "-"*80)
        logger.info("RECOMMENDED CONFIGURATION (minimum distance to ideal point):")
        logger.info("-"*80)
        logger.info(f"Configuration: {best_config['bin_label']} ({int(best_config['num_bins'])} bins)")
        logger.info(f"  Validity:   {best_config['validity_mean_mean']:.3f} ± {best_config['validity_mean_std']:.3f}")
        logger.info(f"  Proximity:  {best_config['proximity_cont_mean_mean']:.3f}")
        logger.info(f"  Diversity:  {best_config['diversity_cont_mean_mean']:.3f}")
        logger.info(f"  Robustness: {best_config['robustness_mean_mean']:.3f} ± {best_config['robustness_mean_std']:.3f}")
        logger.info(f"  Time:       {best_config['generation_time_mean_mean']:.1f}s")
        if 'distance_to_ideal' in best_config:
            logger.info(f"  Distance to ideal: {best_config['distance_to_ideal']:.3f}")
    else:
        logger.warning("No Pareto-optimal solutions found!")
        summary['is_recommended'] = False

    # ========== ADDITIONAL ANALYSIS ==========

    # Compute relative performance vs baseline (10 bins) if it exists
    baseline_10 = summary[summary['num_bins'] == 10]
    if len(baseline_10) > 0:
        baseline = baseline_10.iloc[0]

        for col in ['validity_mean_mean', 'proximity_cont_mean_mean', 
                    'diversity_cont_mean_mean', 'robustness_mean_mean', 
                    'generation_time_mean_mean']:
            if baseline[col] != 0:
                summary[f'{col}_pct_change'] = ((summary[col] - baseline[col]) / abs(baseline[col])) * 100
            else:
                summary[f'{col}_pct_change'] = 0.0

    # Create interpretation column
    def interpret_config(row):
        interpretations = []

        if row['is_recommended']:
            interpretations.append("✓ RECOMMENDED")
        elif row['is_pareto_optimal']:
            interpretations.append("Pareto-optimal")

        if row['pareto_rank'] <= 2:
            interpretations.append(f"Rank {int(row['pareto_rank'])}")

        if row['validity_mean_mean'] < 0.95:
            interpretations.append("Low validity (<95%)")

        if row['robustness_mean_mean'] < 0.5:
            interpretations.append("Low robustness (<50%)")

        # Check if significantly slower than median
        median_time = summary['generation_time_mean_mean'].median()
        if row['generation_time_mean_mean'] > median_time * 1.5:
            interpretations.append("Slow (>1.5× median)")

        if not interpretations:
            interpretations.append("Acceptable")

        return "; ".join(interpretations)

    summary['interpretation'] = summary.apply(interpret_config, axis=1)

    # ========== SAVE RESULTS ==========

    # Sort by Pareto rank (best first), then by robustness
    summary = summary.sort_values(['pareto_rank', 'robustness_mean_mean'],
                                   ascending=[True, False])

    # Save comprehensive CSV
    summary_csv = output_dir / 'binning_summary_table.csv'
    summary.to_csv(summary_csv, index=False)
    logger.info(f"\n✓ Saved comprehensive summary: {summary_csv}")

    # Save simplified LaTeX table
    latex_cols = [
        'num_bins', 'bin_label',
        'validity_mean_mean',
        'proximity_cont_mean_mean',
        'diversity_cont_mean_mean',
        'robustness_mean_mean',
        'generation_time_mean_mean',
        'pareto_rank',
        'interpretation'
    ]

    latex_summary = summary[latex_cols].copy()
    latex_summary.columns = [
        'Bins', 'Method', 'Validity', 'Proximity', 'Diversity', 
        'Robustness', 'Time (s)', 'Rank', 'Interpretation'
    ]

    latex_file = output_dir / 'binning_summary_table.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_summary.to_latex(
            index=False, 
            float_format='%.3f',
            escape=False,
            column_format='c' * len(latex_cols)
        ))

    logger.info(f"⚙️ Saved LaTeX table: {latex_file}")

    return summary_csv


# ============================================================================
# ALSO ADD this new task for detailed recommendation report:
# ============================================================================

@task
def create_recommendation_report(results_csv: Path, summary_csv: Path, output_dir: Path):
    """
    Create detailed recommendation report for thesis.
    This explains the Pareto analysis and justifies the choice.
    """
    logger = get_run_logger()

    df = pd.read_csv(results_csv)
    summary = pd.read_csv(summary_csv)

    # Get recommended configuration
    recommended = summary[summary.get('is_recommended', False)]
    if len(recommended) == 0:
        logger.warning("No recommended configuration found in summary table!")
        return None

    recommended = recommended.iloc[0]

    # Get all Pareto-optimal configs
    pareto = summary[summary.get('is_pareto_optimal', False)]

    # Get baseline (10 bins) if exists
    baseline = summary[summary['num_bins'] == 10]
    baseline = baseline.iloc[0] if len(baseline) > 0 else None

    # ========== BUILD REPORT ==========

    report = []
    report.append("="*80)
    report.append("BINNING SENSITIVITY ANALYSIS - RECOMMENDATION REPORT")
    report.append("="*80)
    report.append("")
    report.append("METHODOLOGY:")
    report.append("-" * 80)
    report.append("This analysis uses Pareto dominance (Deb et al., 2002) to identify")
    report.append("configurations where no alternative simultaneously improves all objectives.")
    report.append("No arbitrary preference weights are imposed.")
    report.append("")
    report.append("Objectives:")
    report.append("  1. Maximize validity (CFs must satisfy target class)")
    report.append("  2. Minimize proximity (CFs should be close to query)")
    report.append("  3. Maximize diversity (CFs should explore solution space)")
    report.append("  4. Maximize robustness (CFs should be stable under perturbation)")
    report.append("")

    report.append("PARETO-OPTIMAL CONFIGURATIONS:")
    report.append("-" * 80)
    report.append(f"Found {len(pareto)} non-dominated solutions:\n")

    for _, row in pareto.iterrows():
        report.append(f"  {row['bin_label']:15s} ({int(row['num_bins']):2d} bins)")
        report.append(f"    Validity:   {row['validity_mean_mean']:.3f} ± {row['validity_mean_std']:.3f}")
        report.append(f"    Proximity:  {row['proximity_cont_mean_mean']:.3f}")
        report.append(f"    Diversity:  {row['diversity_cont_mean_mean']:.3f}")
        report.append(f"    Robustness: {row['robustness_mean_mean']:.3f} ± {row['robustness_mean_std']:.3f}")
        report.append(f"    Time:       {row['generation_time_mean_mean']:.1f}s")
        report.append("")

    report.append("RECOMMENDED CONFIGURATION:")
    report.append("-" * 80)
    report.append(f"Configuration: {recommended['bin_label']} ({int(recommended['num_bins'])} bins)")
    report.append("")
    report.append("Selection criteria: Minimum distance to ideal point")
    report.append("  (validity=1.0, proximity=min, diversity=max, robustness=1.0)")
    report.append("")
    report.append("Performance:")
    report.append(f"  Validity:   {recommended['validity_mean_mean']:.3f} ± {recommended['validity_mean_std']:.3f}")
    report.append(f"  Proximity:  {recommended['proximity_cont_mean_mean']:.3f}")
    report.append(f"  Diversity:  {recommended['diversity_cont_mean_mean']:.3f}")
    report.append(f"  Robustness: {recommended['robustness_mean_mean']:.3f} ± {recommended['robustness_mean_std']:.3f}")
    report.append(f"  Time:       {recommended['generation_time_mean_mean']:.1f}s")
    report.append("")

    if baseline is not None and int(baseline['num_bins']) != int(recommended['num_bins']):
        report.append("COMPARISON TO BASELINE (10 bins):")
        report.append("-" * 80)

        def pct_change(new, old):
            if old == 0:
                return 0.0
            return ((new - old) / abs(old)) * 100

        val_change = pct_change(recommended['validity_mean_mean'], baseline['validity_mean_mean'])
        prox_change = pct_change(recommended['proximity_cont_mean_mean'], baseline['proximity_cont_mean_mean'])
        div_change = pct_change(recommended['diversity_cont_mean_mean'], baseline['diversity_cont_mean_mean'])
        rob_change = pct_change(recommended['robustness_mean_mean'], baseline['robustness_mean_mean'])
        time_change = pct_change(recommended['generation_time_mean_mean'], baseline['generation_time_mean_mean'])

        report.append(f"  Validity:   {val_change:+.1f}%")
        report.append(f"  Proximity:  {prox_change:+.1f}% {'(better)' if prox_change < 0 else '(worse)'}")
        report.append(f"  Diversity:  {div_change:+.1f}%")
        report.append(f"  Robustness: {rob_change:+.1f}%")
        report.append(f"  Time:       {time_change:+.1f}%")
        report.append("")

    report.append("JUSTIFICATION:")
    report.append("-" * 80)
    report.append(f"The {int(recommended['num_bins'])}-bin configuration is recommended because:")
    report.append("  1. It is Pareto-optimal (non-dominated)")
    report.append("  2. It achieves the minimum distance to the ideal point")
    report.append(f"  3. Validity: {recommended['validity_mean_mean']:.1%} (target: ≥95%)")
    report.append(f"  4. Robustness: {recommended['robustness_mean_mean']:.1%} (primary contribution)")

    if int(recommended['num_bins']) == 10:
        report.append("  5. Aligns with Sturges' rule and prior CF literature")

    report.append("")
    report.append("="*80)

    # Write to file
    report_file = output_dir / 'binning_recommendation_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    logger.info(f"⚙️ Created recommendation report: {report_file}")

    # Also print key findings to console
    logger.info("\n" + "="*80)
    logger.info("KEY FINDINGS:")
    logger.info("="*80)
    for line in report[-15:]:  # Print last 15 lines (justification section)
        logger.info(line)

    return report_file


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

    if results:
        csv_file = save_results(results, output_dir)
        logger.info(f"\n🚀 Saved results to {csv_file}")
        plot_path: Path = output_dir / "chart_artefacts"
        plot_path.mkdir(parents=True, exist_ok=True)
        create_plots(csv_file, plot_path)

        # NEW ORDER: Create summary first, then report uses it
        summary_csv = create_summary_table(csv_file, output_dir)
        recommendation_report = create_recommendation_report(csv_file, summary_csv, output_dir)

        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"Results:        {csv_file}")
        logger.info(f"Summary:        {summary_csv}")
        logger.info(f"Recommendation: {recommendation_report}")
        logger.info(f"Plots:          {output_dir}/binning_*.pdf")        
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
