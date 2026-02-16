import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_split_plots(results_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(results_csv)

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    datasets = df['dataset'].unique()
    backends = df['backend'].unique()

    _, axes = plt.subplots(1, 3, figsize=(18, 5))
        
    core_metrics = [
            ('validity_mean', 'Validity', [0.85, 1.01]),
            ('robustness_mean', 'Robustness (keep rate)', [0, 1.01]),
            ('sparsity_cont_mean', 'Sparsity (continuous)', [0, 1.01]),
        ]
    file_name = 'binning_core_metrics_fixed'

    for metric_idx, (metric, title, ylim) in enumerate(core_metrics):
        ax = axes[metric_idx]
        for backend in backends:
            df_be = df[df['backend'] == backend]

            for ds_name in datasets:
                df_ds = df_be[df_be['dataset'] == ds_name]

                fixed_mask = df_ds['num_bins'].isin([5, 10, 15, 20])
                df_fixed = df_ds[fixed_mask].sort_values('num_bins')

                std_col = metric.replace('_mean', '_std')
                ax.errorbar(
                    df_fixed['num_bins'],
                    df_fixed[metric],
                    yerr=df_fixed[std_col] if std_col in df_fixed.columns else None,
                    marker='o',
                    label=f'{ds_name}/{backend}',
                    capsize=4,
                    linewidth=2,
                    alpha=0.8
                )
        ax.set_xlabel('Number of Bins', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xticks(ticks=[5, 10, 15, 20])
        if ylim:
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(out_dir / f'{file_name}.pdf', bbox_inches='tight')
    plt.savefig(out_dir / f'{file_name}.png', bbox_inches='tight', dpi=300)
    plt.savefig(out_dir / f'{file_name}.eps', format='eps')
    plt.close()

    file_name = 'binning_core_metrics_adaptive'
    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    for metric_idx, (metric, title, ylim) in enumerate(core_metrics):
        ax = axes[metric_idx]
        for backend in backends:
            df_be = df[df['backend'] == backend]

            for ds_name in datasets:
                df_ds = df_be[df_be['dataset'] == ds_name]

                adaptive_mask = ~df_ds['num_bins'].isin([5, 10, 15, 20])
                df_adaptive = df_ds[adaptive_mask].sort_values('num_bins')

                std_col = metric.replace('_mean', '_std')

                if not df_adaptive.empty:
                    ax.scatter(
                        df_adaptive['num_bins'],
                        df_adaptive[metric],
                        label=f'{ds_name}/{backend}',
                        s=100,
                        alpha=0.7
                    )

                """ ax.errorbar(
                    df_adaptive['num_bins'],
                    df_adaptive[metric],
                    yerr=df_adaptive[std_col] if std_col in df_adaptive.columns else None,
                    marker='o',
                    label=f'{ds_name}/{backend}',
                    capsize=4,
                    linewidth=2,
                    alpha=0.8
                ) """
        ax.set_xlabel('Number of Bins', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xticks(ticks=df_adaptive['num_bins'].unique().tolist())    # type: ignore
        if ylim:
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(out_dir / f'{file_name}.pdf', bbox_inches='tight')
    plt.savefig(out_dir / f'{file_name}.png', bbox_inches='tight', dpi=300)
    plt.savefig(out_dir / f'{file_name}.eps', format='eps')
    plt.close()


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    root_path = current_dir.parent.parent
    out_path = root_path / "experiments/binning_test/experiment_artefacts/chart_artefacts/splitted_charts"

    out_path.mkdir(parents=True, exist_ok=True)

    results_path = root_path / "experiments/binning_test/experiment_artefacts/binning_sensitivity_results.csv"

    create_split_plots(results_path, out_path)