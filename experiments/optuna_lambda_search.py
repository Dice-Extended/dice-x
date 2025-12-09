# optuna_lambda_search.py
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import pandas as pd

import optuna
from optuna.samplers import NSGAIISampler
from optuna.trial import TrialState

from prefect import flow, get_run_logger

from pathlib import Path

# ---- Import pieces from your grid_search_experiment (Prefect tasks etc.)
from grid_search_experiment import (
    GridConfig,
    load_datasets,
    load_dice_x_models,
    load_dice_baseline_models,
    make_dice_objects,
    gen_kwargs_for_method,
    evaluate_one_setting,
    DefaultPaths,
)

# ============================================================
# Config
# ============================================================

@dataclass
class OptunaConfig:
    n_trials: int = 150
    timeout_s: Optional[int] = None
    seed: int = 42
    # λ search space (log-uniform in [0.1, 10])
    lam_low: float = 0.1
    lam_high: float = 10.0
    # experiment size (kept small per-trial)
    n_test_points: int = 5
    k_cfs: int = 3
    n_repeat: int = 3
    # persistence
    save_every_trials: int = 10    # append to CSV every N trials

@dataclass
class PlotConfig:
    enable: bool = True
    save_html: bool = True
    save_png: bool = True
    figsize: tuple = (8, 6)

# ============================================================
# Helpers: Pareto + utopia pick + remaining trials
# ============================================================

def _is_dominated(a, b) -> bool:
    """True if point a is dominated by b (maximize all dims)."""
    return all(bi >= ai for ai, bi in zip(a, b)) and any(bi > ai for ai, bi in zip(a, b))

def _pareto_front(points: List[tuple]) -> List[int]:
    """Return indices of nondominated points (maximize all dims)."""
    idxs = []
    for i, p in enumerate(points):
        if not any(i != j and _is_dominated(p, q) for j, q in enumerate(points)):
            idxs.append(i)
    return idxs

def _select_utopia(points: List[tuple]) -> Optional[int]:
    """
    points: list of (robust, div_cont, prox_cont_negmean, sparsity)
    Min-max normalize per axis, then pick the point with min distance to (1,1,1,1).
    """
    if not points:
        return None
    arr = np.asarray(points, dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    den = np.where(maxs > mins, (maxs - mins), 1.0)
    norm = (arr - mins) / den
    d = np.sqrt(((1.0 - norm) ** 2).sum(axis=1))
    return int(np.argmin(d))

def _remaining_trials(study: optuna.study.Study, target_total: int) -> int:
    # count only COMPLETED trials (ignore pruned/failed/running)
    completed = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
    return max(0, target_total - completed)

def _open_study_and_remaining(
    storage_url: str, study_name: str, directions: List[str], sampler, ocfg
) -> tuple[optuna.study.Study, int]:
    study = optuna.create_study(
        directions=directions,
        sampler=sampler,
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )
    rem = _remaining_trials(study, target_total=ocfg.n_trials)
    return study, rem

# ============================================================
# Shared evaluation (one λ-triple → metrics means on n test pts)
# ============================================================

def _evaluate_lambdas_for_group(
    lam1: float, lam2: float, lam3: float,
    train_df: pd.DataFrame, test_df_full: pd.DataFrame,
    target: str, ds_name: str, backend: str, method_label: str,
    models_x, models_base, cfg_like: GridConfig
) -> Dict[str, float]:
    """
    Runs CF generation + metric suite on a fixed test subset for one (λ1, λ2, λ3).
    Returns dict with *_mean keys for the four objectives.
    """
    try:
        # Build once per trial
        _, exp, data_iface, _, model_pred = make_dice_objects.fn(
            train_df, target, ds_name, backend, method_label, models_x, models_base
        )

        gen_kwargs = gen_kwargs_for_method.fn(
            method_label=method_label,
            total_cfs=cfg_like.k_cfs,
            lam1=lam1, lam2=lam2, lam3=lam3,
            backend=backend,
        )

        rows = evaluate_one_setting.fn(
            exp=exp, data_iface=data_iface, target=target, backend=backend,
            model_for_pred=model_pred, test_df=test_df_full, gen_kwargs=gen_kwargs,
            robustness_cfg=(cfg_like.noise_sd, cfg_like.cat_flip_p, cfg_like.n_repeat, cfg_like.random_seed),
        )

        rows_df = pd.DataFrame(rows)
        ok = rows_df[rows_df["error"].fillna("") == ""]
        if ok.empty:
            # steer the search away from this region
            return dict(
                robustness_keep_rate_mean=0.0,
                div_cont_mad_mean=0.0,
                prox_cont_mad_negmean_mean=-1e9,
                sparsity_cont_mean=0.0,
            )

        agg = ok.agg({
            "robustness_keep_rate": "mean",
            "div_cont_mad": "mean",
            "prox_cont_mad_negmean": "mean",
            "sparsity_cont": "mean",
        })
        return dict(
            robustness_keep_rate_mean=float(agg["robustness_keep_rate"]),
            div_cont_mad_mean=float(agg["div_cont_mad"]),
            prox_cont_mad_negmean_mean=float(agg["prox_cont_mad_negmean"]),
            sparsity_cont_mean=float(agg["sparsity_cont"]),
        )
    except Exception:
        # very defensive: never crash the whole optimization
        return dict(
            robustness_keep_rate_mean=0.0,
            div_cont_mad_mean=0.0,
            prox_cont_mad_negmean_mean=-1e9,
            sparsity_cont_mean=0.0,
        )


# ============================================================
# CSV helpers
# ============================================================

def _optuna_trials_csv(out_dir: Path, ds: str, backend: str, method: str) -> Path:
    phase_dir = out_dir / "optuna_mo"
    phase_dir.mkdir(parents=True, exist_ok=True)
    return phase_dir / f"rows_{ds}__{backend}__{method}__TRIALS.csv"

def _optuna_pareto_csv(out_dir: Path, ds: str, backend: str, method: str) -> Path:
    phase_dir = out_dir / "optuna_mo"
    return phase_dir / f"rows_{ds}__{backend}__{method}__PARETO.csv"

def _optuna_selected_csv(out_dir: Path, ds: str, backend: str, method: str) -> Path:
    phase_dir = out_dir / "optuna_mo"
    return phase_dir / f"rows_{ds}__{backend}__{method}__SELECTED.csv"

def _append_trials(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = (not path.exists()) or (path.stat().st_size == 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=header, index=False)

def _dump_full_replace(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ============================================================
# Small util: fixed test subset per group
# ============================================================

def _make_test_subset(df: pd.DataFrame, target: str, n: int, seed: int) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split
    target_col = df[target]
    _, test = train_test_split(df, test_size=0.2, random_state=seed, stratify=target_col)
    if len(test) > n:
        test = test.sample(n=n, random_state=seed)
    return test


# ============================================================
# Visualisation helpers
# ============================================================

from pathlib import Path
def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save_figure(fig, path_base: Path):
    # Save PNG and SVG
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    # fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")

def plot_optuna_pareto_html(study: optuna.study.Study, out_dir: Path, ds: str, backend: str, method: str):
    """Interactive Pareto (Plotly) saved as HTML."""
    try:
        from optuna.visualization import plot_pareto_front
        fig = plot_pareto_front(study)
        html_path = _safe_mkdir(out_dir / "optuna_mo" / "figs") / f"{ds}__{backend}__{method}__pareto.html"
        fig.write_html(str(html_path))
        print(f"[viz] wrote {html_path}")
    except Exception as e:
        print(f"[viz] (plot_pareto_front) skipped: {e}")

def plot_static_tradespace(trials_df: pd.DataFrame, pareto_df: pd.DataFrame, out_dir: Path, ds: str, backend: str, method: str):
    """
    Static 2D Pareto projections (PNG+SVG): robustness vs {diversity, proximity, sparsity}.
    Pareto points highlighted. Color by lambda3 (robustness weight), size by lambda2, marker alpha by lambda1.
    """
    import matplotlib.pyplot as plt
    metrics = {
        "robustness_keep_rate_mean": "Robustness ↑",
        "div_cont_mad_mean": "Diversity (cont. MAD) ↑",
        "prox_cont_mad_negmean_mean": "Proximity (−MAD) ↑",
        "sparsity_cont_mean": "Sparsity ↑",
    }

    x = "robustness_keep_rate_mean"
    ys = ["div_cont_mad_mean", "prox_cont_mad_negmean_mean", "sparsity_cont_mean"]

    figs_dir = _safe_mkdir(out_dir / "optuna_mo" / "figs")
    base = f"{ds}__{backend}__{method}"

    # Common styling handles missing lambda3 for baseline DiCE (fixed 0.0)
    trials_df = trials_df.copy()
    trials_df["lambda3"] = trials_df.get("lambda3", 0.0).fillna(0.0)
    pareto_df = pareto_df.copy()
    pareto_df["lambda3"] = pareto_df.get("lambda3", 0.0).fillna(0.0)

    for y in ys:
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(
            trials_df[x], trials_df[y],
            c=trials_df["lambda3"],            # color ~ robustness weight
            s=40 + 25*np.log10(trials_df["lambda2"]),  # bubble size ~ lambda2
            alpha=np.clip(0.25 + 0.15*np.log10(trials_df["lambda1"]), 0.15, 0.9),    # type: ignore
            edgecolor="none"
        )
        # Pareto overlay
        ax.scatter(
            pareto_df[x], pareto_df[y],
            facecolors="none", edgecolors="black", s=80, linewidths=1.0, label="Pareto"
        )

        ax.set_xlabel(metrics[x])
        ax.set_ylabel(metrics[y])
        ax.set_title(f"{ds} / {backend} / {method}\nPareto projections")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label("λ3 (robustness weight)")

        ax.legend(loc="best", frameon=False)
        _save_figure(fig, figs_dir / f"{base}__{x}_vs_{y}")
        plt.close(fig)

def plot_parallel_coordinates(trials_df: pd.DataFrame, pareto_df: pd.DataFrame, out_dir: Path, ds: str, backend: str, method: str):
    """
    Static parallel coordinates over the 4 objectives (normalized 0-1).
    Pareto points in bold.
    """
    import matplotlib.pyplot as plt

    cols = [
        "robustness_keep_rate_mean",
        "div_cont_mad_mean",
        "prox_cont_mad_negmean_mean",
        "sparsity_cont_mean",
    ]
    labels = ["Robustness", "Diversity", "Proximity", "Sparsity"]

    def _norm(df):
        arr = df[cols].to_numpy(dtype=float)
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        den  = np.where(maxs > mins, (maxs - mins), 1.0)
        return (arr - mins) / den

    figs_dir = _safe_mkdir(out_dir / "optuna_mo" / "figs")
    base = f"{ds}__{backend}__{method}"

    arr_all = _norm(trials_df)
    arr_pf  = _norm(pareto_df)

    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(8, 4))

    # all trials (light)
    for row in arr_all:
        ax.plot(x, row, linewidth=0.6, alpha=0.2)

    # pareto (highlight)
    for row in arr_pf:
        ax.plot(x, row, linewidth=2.0, alpha=0.9, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized objective (↑ better)")
    ax.set_title(f"{ds} / {backend} / {method}\nParallel coordinates (Pareto bold)")
    _save_figure(fig, figs_dir / f"{base}__parallel")
    plt.close(fig)

def _fig_dir(base: Path, ds: str, backend: str, method: str) -> Path:
    d = base / "optuna_mo" / "figs" / f"{ds}__{backend}__{method}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _save_plotly(fig, path_html: Path, path_png: Optional[Path], save_html: bool, save_png: bool):
    if save_html:
        fig.write_html(str(path_html), include_plotlyjs="cdn")
    if save_png:
        try:
            fig.write_image(str(path_png))
        except Exception as e:
            # kaleido not installed or another export issue
            print(f"[plot] PNG export skipped: {e}")

def _plot_optuna_plotly(study: optuna.study.Study, out_dir: Path, ds: str, backend: str, method: str, cfg: PlotConfig):
    import optuna.visualization as ov
    fd = _fig_dir(out_dir, ds, backend, method)

    # Pareto front (multi-objective)
    fig_pf = ov.plot_pareto_front(
        study,
        target_names=["robustness", "diversity", "proximity", "sparsity"],
    )
    _save_plotly(fig_pf, fd / "pareto_front.html", fd / "pareto_front.png", cfg.save_html, cfg.save_png)

    # Parallel coordinates across params & objectives
    fig_pc = ov.plot_parallel_coordinate(
        study,
        params=["lambda1", "lambda2", "lambda3"],   # λ3 is absent for baseline trials but fine
        target_names=["robustness", "diversity", "proximity", "sparsity"],
    )
    _save_plotly(fig_pc, fd / "parallel_coordinates.html", fd / "parallel_coordinates.png", cfg.save_html, cfg.save_png)

def _plot_static_tradespace(trials_df: pd.DataFrame, pareto_df: pd.DataFrame, out_dir: Path, ds: str, backend: str, method: str, cfg: PlotConfig):
    import matplotlib.pyplot as plt

    fd = _fig_dir(out_dir, ds, backend, method)
    cols = [
        ("robustness_keep_rate_mean", "prox_cont_mad_negmean_mean", "robustness vs proximity"),
        ("robustness_keep_rate_mean", "div_cont_mad_mean",         "robustness vs diversity"),
        ("prox_cont_mad_negmean_mean","div_cont_mad_mean",         "proximity vs diversity"),
        ("prox_cont_mad_negmean_mean","sparsity_cont_mean",        "proximity vs sparsity"),
    ]

    for x_col, y_col, title in cols:
        fig, ax = plt.subplots(figsize=cfg.figsize)
        ax.scatter(
            trials_df[x_col].values, trials_df[y_col].values,
            s=18, alpha=0.35, label="trials"
        )
        if not pareto_df.empty:
            ax.scatter(
                pareto_df[x_col].values, pareto_df[y_col].values,
                s=32, alpha=0.9, marker="x", label="pareto"
            )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{title}  •  {ds}/{backend}/{method}")
        ax.legend(loc="best")
        fig.tight_layout()
        png = fd / f"{title.replace(' ', '_')}.png"
        fig.savefig(png, dpi=200)
        plt.close(fig)



# ============================================================
# One (dataset, backend, method) Optuna run
# ============================================================

def _run_optuna_for_group(
    out_dir: Path,
    ds_tuple: Tuple[pd.DataFrame, str, str],
    backend: str,
    method_label: str,
    models_x,
    models_base,
    gcfg: GridConfig,
    ocfg: OptunaConfig,
    plot_cfg: PlotConfig=PlotConfig()
):
    df, target, ds_name = ds_tuple
    logger = get_run_logger()

    # Fix test subset for this study to keep trials comparable
    test_df = _make_test_subset(df, target, n=ocfg.n_test_points, seed=ocfg.seed)

    # 4 objectives: maximize all (robustness, diversity, proximity, sparsity)
    directions = ["maximize", "maximize", "maximize", "maximize"]
    sampler = NSGAIISampler(
        population_size=48,
        crossover_prob=0.9,
        mutation_prob=None,
        seed=ocfg.seed,
    )

    trials_path = _optuna_trials_csv(out_dir, ds_name, backend, method_label)
    batched_rows: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial):
        # --- search space (λ3 fixed for baseline DiCE) ---
        lam1 = trial.suggest_float("lambda1", ocfg.lam_low, ocfg.lam_high, log=True)
        lam2 = trial.suggest_float("lambda2", ocfg.lam_low, ocfg.lam_high, log=True)
        lam3 = 0.0 if method_label == "DiCE" else trial.suggest_float("lambda3", ocfg.lam_low, ocfg.lam_high, log=True)

        # --- evaluate metrics ---
        metrics = _evaluate_lambdas_for_group(
            lam1, lam2, lam3,
            train_df=df, test_df_full=test_df,
            target=target, ds_name=ds_name, backend=backend, method_label=method_label,
            models_x=models_x, models_base=models_base, cfg_like=gcfg
        )

        # --- persist this trial (buffered) ---
        row = dict(
            dataset=ds_name, backend=backend, method=method_label,
            lambda1=lam1, lambda2=lam2, lambda3=lam3,
            **metrics
        )
        batched_rows.append(row)
        if len(batched_rows) >= ocfg.save_every_trials:
            _append_trials(trials_path, batched_rows)
            batched_rows.clear()

        # --- return order must match directions ---
        return (
            metrics["robustness_keep_rate_mean"],
            metrics["div_cont_mad_mean"],
            metrics["prox_cont_mad_negmean_mean"],
            metrics["sparsity_cont_mean"],
        )
    
    OPTUNA_DB = Path.home() / "Documents/bau24-25/thesis/repos/DiCE-X/experiments/lambda_grid_results" / "optuna_studies.db"
    OPTUNA_DB.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{OPTUNA_DB.resolve()}"

    study_name = f"{ds_name}__{backend}__{method_label}"
    study, n_to_run = _open_study_and_remaining(storage_url, study_name, directions, sampler, ocfg)

    completed = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
    
    logger.info(f"[optuna_mo] {study_name}: completed={completed}, target={ocfg.n_trials}, remaining={n_to_run}")

    existing = [t for t in study.trials if t.values is not None]
    
    logger.info(
        f"[optuna_mo] Resuming study '{study.study_name}' "
        f"at {storage_url} with {len(existing) - 1} completed trials."
    )

    if n_to_run > 0:
        study.optimize(
            objective,
            n_trials=n_to_run,
            timeout=ocfg.timeout_s,
            gc_after_trial=True,
            catch=(Exception,)
        )
    else:
        logger.info(f"[optuna_mo] {study_name}: already finished; skipping optimize()")

    

    # Flush remaining buffered rows
    if batched_rows:
        _append_trials(trials_path, batched_rows)
        batched_rows.clear()

    # Build consolidated DataFrame from study (authoritative)
    trials_data = []
    for t in study.trials:
        if t.values is None:
            continue
        vals = list(t.values)
        lam1 = t.params.get("lambda1")
        lam2 = t.params.get("lambda2")
        lam3 = 0.0 if method_label == "DiCE" else t.params.get("lambda3")
        trials_data.append(dict(
            dataset=ds_name, backend=backend, method=method_label,
            lambda1=lam1, lambda2=lam2, lambda3=lam3,
            robustness_keep_rate_mean=vals[0],
            div_cont_mad_mean=vals[1],
            prox_cont_mad_negmean_mean=vals[2],
            sparsity_cont_mean=vals[3],
        ))
    trials_df = pd.DataFrame(trials_data)

    pareto_df = pd.DataFrame()
    if not trials_df.empty:
        # overwrite consolidated TRIALS csv (nice to have)
        _dump_full_replace(trials_path, trials_df)

        # Pareto front
        pts = trials_df[
            ["robustness_keep_rate_mean", "div_cont_mad_mean", "prox_cont_mad_negmean_mean", "sparsity_cont_mean"]
        ].to_numpy()
        pf_idx = _pareto_front([tuple(p) for p in pts])
        pareto_df = trials_df.iloc[pf_idx].reset_index(drop=True)
        _dump_full_replace(_optuna_pareto_csv(out_dir, ds_name, backend, method_label), pareto_df)

        # Select a single utopia-closest configuration
        sel_idx = _select_utopia(
            pareto_df[
                ["robustness_keep_rate_mean", "div_cont_mad_mean", "prox_cont_mad_negmean_mean", "sparsity_cont_mean"]
            ].to_numpy().tolist()
        )

        if sel_idx is not None:
            selected_df = pareto_df.iloc[[sel_idx]].reset_index(drop=True)
            _dump_full_replace(_optuna_selected_csv(out_dir, ds_name, backend, method_label), selected_df)
        if plot_cfg.enable:
            try:
                _plot_optuna_plotly(study, out_dir, ds_name, backend, method_label, plot_cfg)
            except Exception as e:
                print(f"[plotly] skipped: {e}")
            try:
                _plot_static_tradespace(trials_df, pareto_df, out_dir, ds_name, backend, method_label, plot_cfg)
            except Exception as e:
                print(f"[matplotlib] skipped: {e}")

    logger.info(
        f"[optuna_mo] {ds_name}/{backend}/{method_label}: "
        f"{len(trials_df)} trials, Pareto {len(pareto_df)}"
    )


# ============================================================
# Main Prefect flow
# ============================================================

def _all_groups(datasets, backends):
    for df, target, ds_name in datasets:
        for backend in backends:
            for method_label in ("DiCE-Ext", "DiCE"):
                yield (df, target, ds_name, backend, method_label)


@flow(name="DiCE λ Multi-Objective (Optuna)")
def optuna_multi_objective_flow(
    out_dir: Path = DefaultPaths().out_dir,
    dice_x_pickle: Optional[Path] = DefaultPaths().dice_x_pickle,
    dice_baseline_pickle: Optional[Path] = DefaultPaths().dice_baseline_pickle,
    backends: List[str] = ["sklearn", "PYT", "TF2"],
    gcfg: GridConfig = GridConfig(),
    ocfg: OptunaConfig = OptunaConfig(),
):
    logger = get_run_logger()
    np.random.seed(ocfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets/models
    datasets = load_datasets()
    if gcfg.limit_datasets:
        datasets = [d for d in datasets if d[2] in set(gcfg.limit_datasets)]
    if gcfg.limit_backends:
        backends = [b for b in backends if b in set(gcfg.limit_backends)]
    models_x = load_dice_x_models(dice_x_pickle)
    models_base = load_dice_baseline_models(dice_baseline_pickle)
    plot_cfg = PlotConfig(enable=True, save_html=True, save_png=True)

    # Build ordered list of all groups
    groups: List[Tuple[pd.DataFrame, str, str, str, str]] = []
    for ds_tuple in datasets:
        df, target, ds_name = ds_tuple
        for backend in backends:
            for method_label in ("DiCE-Ext", "DiCE"):
                groups.append((df, target, ds_name, backend, method_label))

    # Find first unfinished group from the study DB
    OPTUNA_DB = out_dir / "optuna_studies.db"
    OPTUNA_DB.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{OPTUNA_DB.resolve()}"
    directions = ["maximize", "maximize", "maximize", "maximize"]

    # Use a fixed sampler spec for probing
    probe_sampler = NSGAIISampler(
        population_size=48, crossover_prob=0.9, mutation_prob=None, seed=ocfg.seed
    )

    start_idx = 0
    for i, (_, _, ds_name, backend, method_label) in enumerate(groups):
        study_name = f"{ds_name}__{backend}__{method_label}"
        try:
            study = optuna.create_study(
                directions=directions,
                sampler=probe_sampler,
                study_name=study_name,
                storage=storage_url,
                load_if_exists=True,
            )
            rem = _remaining_trials(study, ocfg.n_trials)
            if rem > 0:
                start_idx = i
                break
        except Exception:
            # study doesn't exist yet → definitely unfinished
            start_idx = i
            break

    if start_idx > 0:
        logger.info(f"[optuna_mo] Skipping {start_idx} finished group(s); resuming from index {start_idx}.")

    # Run from first unfinished group to the end
    for i in range(start_idx, len(groups)):
        df, target, ds_name, backend, method_label = groups[i]

        # Skip baseline if models missing
        if method_label == "DiCE" and models_base is None:
            logger.warning(f"[optuna_mo] Baseline models missing; skipping DiCE for {ds_name}/{backend}.")
            continue

        # Align per-trial size with ocfg
        gcfg_local = GridConfig(
            n_test_points=ocfg.n_test_points,
            k_cfs=ocfg.k_cfs,
            n_repeat=ocfg.n_repeat,
            random_seed=ocfg.seed,
        )

        # Open study and compute remaining trials *now* (fresh, per-group)
        sampler = NSGAIISampler(
            population_size=48, crossover_prob=0.9, mutation_prob=None, seed=ocfg.seed
        )
        study_name = f"{ds_name}__{backend}__{method_label}"
        study, n_to_run = _open_study_and_remaining(
            storage_url, study_name, directions, sampler, ocfg
        )
        completed = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
        logger.info(
            f"[optuna_mo] {study_name}: completed={completed}, target={ocfg.n_trials}, remaining={n_to_run}"
        )

        if n_to_run == 0:
            logger.info(f"[optuna_mo] {study_name}: already finished; skipping optimize().")
            continue

        # Run this group; _run_optuna_for_group will call create_study again
        # with the same storage+name and just run objective for 'remaining' trials.
        _run_optuna_for_group(
            out_dir=out_dir,
            ds_tuple=(df, target, ds_name),
            backend=backend,
            method_label=method_label,
            models_x=models_x,
            models_base=models_base,
            gcfg=gcfg_local,
            ocfg=ocfg,
            plot_cfg=plot_cfg,
        )


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    # put this at the very top of optuna_lambda_search.py, BEFORE importing Prefect
    # import os, tempfile, uuid
    # os.environ.setdefault("PREFECT_PROFILE", "default")
    # os.environ["PREFECT_API_URL"] = ""  # force ephemeral API
    # os.environ["PREFECT_API_DATABASE_CONNECTION_URL"] = \
    #     f"sqlite+aiosqlite:///{tempfile.gettempdir()}/prefect-{uuid.uuid4().hex}.db"

    optuna_multi_objective_flow(
        gcfg=GridConfig(n_test_points=5, k_cfs=3, n_repeat=3, random_seed=42),
        ocfg=OptunaConfig(n_trials=200, seed=42),
    )
