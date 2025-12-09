# Prefect-powered λ-grid search for DiCE / DiCE-Extended
# =====================================================
#
# Two-phase sweep (coarse logspace 0.1→10; fine = local neighborhoods around top coarse seeds)
# for four datasets and three backends, generating counterfactuals with DiCE (baseline) and
# DiCE-Extended (w/ robustness), computing a metric suite, and APPENDING progress to per-group
# PART files. Run-resumable: we skip λ-triples that already have all test points, and continue
# partial λ from the next missing q_idx. If a group's PART/final file already has the full
# expected row count, we SKIP the whole group/phase.

from __future__ import annotations

import sys
dice_path    = "/Users/volk/Documents/bau24-25/thesis/repos/DiCE"
dice_x_path  = "/Users/volk/Documents/bau24-25/thesis/repos/DiCE-X"
for p in (dice_x_path, dice_path):
    if p not in sys.path:
        sys.path.insert(0, p)

import os
import time
import math
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import os, tempfile, uuid
os.environ.setdefault("PREFECT_PROFILE", "default")
os.environ.setdefault("PREFECT_API_URL", "")  # empty => ephemeral API
os.environ.setdefault(
    "PREFECT_API_DATABASE_CONNECTION_URL",
    f"sqlite+aiosqlite:///{tempfile.gettempdir()}/prefect-{uuid.uuid4().hex}.db"
)

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.task_runners import SequentialTaskRunner

# External dependencies expected in the user's environment
import torch  # for PYT model loading
import tensorflow as tf  # for TF2 model loading

# Your local modules
import dice_ml  # baseline
import dice_ml_x  # extended (with robustness term)
from dice_ml_x.utils import neuralnetworks, helpers


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class DefaultPaths:
    out_dir: Path = Path("lambda_grid_results")
    dice_x_pickle: Optional[Path] = Path("../docs/source/notebooks/benchmarking_results_23_01_2025-01_05.pkl")
    dice_baseline_pickle: Optional[Path] = Path("/Users/volk/Documents/bau24-25/thesis/repos/DiCE/docs/source/notebooks/final_benchmarking_results.pkl")


@dataclass
class GridConfig:
    # Search spaces
    coarse_log_min: float = -1.0   # log10(0.1)
    coarse_log_max: float = 1.0   # log10(10)
    coarse_num: int = 10            # 5^3 = 125

    # Neighborhood refinement around top coarse seeds
    n_top_from_coarse: int = 6
    fine_scale: Tuple[float, float, float] = (0.8, 1.0, 1.2)  # ±20% steps
    fine_num: int = 3  # unused (neighborhood instead)

    # Experiment size (light defaults)
    n_test_points: int = 5
    k_cfs: int = 3
    desired_class: str = "opposite"

    # Robustness probe
    noise_sd: float = 0.10
    cat_flip_p: float = 0.20
    n_repeat: int = 3              # ↓ from 30 for coarse/fine

    # Hi-fidelity (optional final pass on top-3)
    run_hi_fidelity: bool = False
    hi_top_n: int = 3
    hi_n_test_points: int = 20
    hi_k_cfs: int = 5
    hi_n_repeat: int = 30

    # Execution
    random_seed: int = 42
    save_every_rows: int = 20      # append exactly N rows when hit
    n_lambdas_limit: Optional[int] = None  # extra hard cap if needed

    # Optional filters
    limit_datasets: Optional[List[str]] = None  # e.g., ["compas-recidivism"]
    limit_backends: Optional[List[str]] = None  # e.g., ["sklearn"]


def _nz_norm(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.max() == s.min():
        return pd.Series(0.5, index=s.index)
    return ((s - s.min()) / (s.max() - s.min())).fillna(0.5)

def composite_score(df: pd.DataFrame) -> pd.Series:
    return (
        0.35 * _nz_norm(df["robustness_keep_rate_mean"]) +
        0.30 * _nz_norm(df["div_cont_mad_mean"]) +
        0.20 * _nz_norm(df["prox_cont_mad_negmean_mean"]) +
        0.15 * _nz_norm(df["sparsity_cont_mean"])
    )

def pick_top_seeds(agg: pd.DataFrame, cfg: GridConfig) -> Dict[Tuple[str,str,str], List[Tuple[float,float,float]]]:
    seeds: Dict[Tuple[str,str,str], List[Tuple[float,float,float]]] = {}
    if agg.empty:
        return seeds
    ranked = (agg.assign(score=composite_score(agg))
                .sort_values(["dataset","backend","method","score"], ascending=[True,True,True,False])
                .groupby(["dataset","backend","method"], as_index=False, group_keys=False)
                .head(cfg.n_top_from_coarse))
    for _, row in ranked.iterrows():
        key = (row["dataset"], row["backend"], row["method"])
        seeds.setdefault(key, []).append((float(row["lambda1"]), float(row["lambda2"]), float(row["lambda3"])))
    return seeds

def make_local_neighborhood(center: Tuple[float,float,float], scales: Tuple[float,float,float]) -> List[Tuple[float,float,float]]:
    l1, l2, l3 = center
    grid = []
    for a in scales:
        for b in scales:
            for c in scales:
                t = (max(1e-6, l1*a), max(1e-6, l2*b), max(1e-6, l3*c))
                grid.append(t)
    return list({(round(x,12),round(y,12),round(z,12)) for x,y,z in grid})


# -----------------------------
# RESUME / APPEND HELPERS
# -----------------------------


def _rank_top_k(agg: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()
    ranked = (agg.assign(score=composite_score(agg))
                .sort_values(["dataset","backend","method","score"],
                             ascending=[True, True, True, False])
                .groupby(["dataset","backend","method"], as_index=False, group_keys=False)
                .head(k))
    return ranked

def _finalize_and_rank(out_dir: Path, all_agg: List[pd.DataFrame], logger, cfg: GridConfig):
    # Re-load and save all rows and aggregates (keeps every step)
    coarse_all = _load_all_phase_rows(out_dir, "coarse")
    coarse_rows_path = out_dir / f"coarse__rows_{int(time.time())}.csv"
    save_df(coarse_all, coarse_rows_path)
    coarse_agg = aggregate_rows(coarse_all) if not coarse_all.empty else pd.DataFrame()
    if not coarse_agg.empty:
        coarse_agg_path = out_dir / f"coarse__agg_{int(time.time())}.csv"
        save_df(coarse_agg, coarse_agg_path)
        all_agg.append(coarse_agg)
        save_df(_rank_top_k(coarse_agg, 10), out_dir / f"coarse__top10_{int(time.time())}.csv")
        save_df(_rank_top_k(coarse_agg, 1),  out_dir / f"coarse__top1_{int(time.time())}.csv")

    fine_all = _load_all_phase_rows(out_dir, "fine")
    fine_rows_path = out_dir / f"fine__rows_{int(time.time())}.csv"
    save_df(fine_all, fine_rows_path)
    fine_agg = aggregate_rows(fine_all) if not fine_all.empty else pd.DataFrame()
    if not fine_agg.empty:
        fine_agg_path = out_dir / f"fine__agg_{int(time.time())}.csv"
        save_df(fine_agg, fine_agg_path)
        all_agg.append(fine_agg)
        save_df(_rank_top_k(fine_agg, 10), out_dir / f"fine__top10_{int(time.time())}.csv")
        save_df(_rank_top_k(fine_agg, 1),  out_dir / f"fine__top1_{int(time.time())}.csv")

    # Global winners across phases
    combined_agg = pd.concat(
        [x for x in [coarse_agg, fine_agg] if x is not None and not x.empty],
        ignore_index=True
    ) if ((coarse_agg is not None and not coarse_agg.empty) or (fine_agg is not None and not fine_agg.empty)) else pd.DataFrame()

    if not combined_agg.empty:
        winners_top10 = _rank_top_k(combined_agg, 10)
        winners_top1  = _rank_top_k(combined_agg, 1)
        save_df(winners_top10, out_dir / f"winners_top10_{int(time.time())}.csv")
        save_df(winners_top1,  out_dir / f"winners_top1_{int(time.time())}.csv")
        logger.info("Saved global winners (top10/top1).")


def _remaining_lambdas(
    all_lambdas: List[Tuple[float,float,float]],
    out_dir: Path,
    phase: str,
    dataset: str,
    backend: str,
    method_label: str,
    n_test_points: int,
) -> Tuple[List[Tuple[float,float,float]], Dict[Tuple[float,float,float], int]]:
    """
    Return (todo_lambdas, partial_next_q) for this group/phase.
    - todo_lambdas: ordered subset of all_lambdas that still need work
    - partial_next_q: map of λ -> next q_idx to start from (0..n_test_points-1)
    """
    existing = _load_group_rows(out_dir, phase, dataset, backend, method_label)
    done_triples, partial_next_q = _resume_map(existing, n_test_points=n_test_points)
    done_set = set(done_triples)
    # Keep order from the original grid
    todo_lambdas = [tuple(map(float, lam)) for lam in all_lambdas if tuple(map(float, lam)) not in done_set]
    return todo_lambdas, partial_next_q


def _pick_newest(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)

def _load_group_rows(out_dir: Path, phase: str, dataset: str, backend: str, method_label: str) -> pd.DataFrame:
    """Load newest rows for this group (prefer PART; fall back to newest phase final)."""
    phase_dir = out_dir / phase
    if phase_dir.exists():
        part = list(phase_dir.glob(f"rows_{dataset}__{backend}__{method_label}__*__PART.csv"))
        chosen = _pick_newest(part)
        if chosen is not None:
            try:
                return pd.read_csv(chosen)
            except Exception:
                pass
    # fallback: newest phase final
    finals = list(out_dir.glob(f"{phase}__rows_*.csv"))
    chosen = _pick_newest(finals)
    if chosen is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(chosen)
    except Exception:
        return pd.DataFrame()

def _resume_map(existing: pd.DataFrame, n_test_points: int) -> Tuple[set, Dict[Tuple[float,float,float], int]]:
    """Return (completed λ set, partial->next_q_idx map)."""
    if existing.empty:
        return set(), {}
    need = {"lambda1","lambda2","lambda3","q_idx"}
    if not need.issubset(existing.columns):
        return set(), {}
    df = existing.drop_duplicates(subset=["lambda1","lambda2","lambda3","q_idx"]).copy()
    cnt = df.groupby(["lambda1","lambda2","lambda3"])["q_idx"].nunique()
    done = set([tuple(map(float,k)) for k, v in cnt.items() if v >= n_test_points])
    partial = cnt[cnt < n_test_points]
    next_q: Dict[Tuple[float,float,float], int] = {}
    if not partial.empty:
        maxq = df.groupby(["lambda1","lambda2","lambda3"])["q_idx"].max().to_dict()
        for k in partial.index:
            lam = tuple(map(float,k))
            q = int(maxq.get(k, -1)) + 1
            next_q[lam] = max(0, min(q, n_test_points))
    return done, next_q

def _part_path(out_dir: Path, phase: str, dataset: str, backend: str, method_label: str) -> Path:
    phase_dir = out_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    existing = list(phase_dir.glob(f"rows_{dataset}__{backend}__{method_label}__*__PART.csv"))
    newest = _pick_newest(existing)
    if newest is not None:
        return newest
    return phase_dir / f"rows_{dataset}__{backend}__{method_label}__{int(time.time())}__PART.csv"

def _append_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (not path.exists()) or (path.stat().st_size == 0)
    df.to_csv(path, mode="a", header=header, index=False)

def _load_all_phase_rows(out_dir: Path, phase: str) -> pd.DataFrame:
    """Concat all PARTs and finals for a phase (drop exact duplicates)."""
    frames: List[pd.DataFrame] = []
    # finals
    for p in out_dir.glob(f"{phase}__rows_*.csv"):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    # parts
    phase_dir = out_dir / phase
    if phase_dir.exists():
        for p in phase_dir.glob("rows_*__PART.csv"):
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True).drop_duplicates()
    return all_df

def _has_complete_group(out_dir: Path, phase: str, dataset: str, backend: str, method_label: str, expected_rows: int) -> bool:
    """
    Return True if ANY existing PART or final phase CSV for this group already has
    >= expected_rows (e.g., coarse: 125 λ * 5 q = 625 rows). This lets us SKIP
    the whole group/phase at start.
    """
    # scan PARTs for this group
    phase_dir = out_dir / phase
    candidates: List[Path] = []
    if phase_dir.exists():
        candidates += list(phase_dir.glob(f"rows_{dataset}__{backend}__{method_label}__*__PART.csv"))
    # also scan final phase CSVs (in case you consolidated previously)
    candidates += list(out_dir.glob(f"{phase}__rows_*.csv"))
    for p in sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                n_lines = sum(1 for _ in fh)
            n_rows = max(0, n_lines - 1)
            if n_rows >= expected_rows:
                return True
        except Exception:
            continue
    return False


# -----------------------------
# Dataset & model loading
# -----------------------------

@task(retries=1, retry_delay_seconds=5)
def load_datasets() -> List[Tuple[pd.DataFrame, str, str]]:
    ds: List[Tuple[pd.DataFrame, str, str]] = [
        (helpers.load_compas_dataset(),  "twoyearrecid", "compas-recidivism"),
        (helpers.load_adult_income_dataset(), "income",  "adult-income"),
        (helpers.load_lending_club_dataset(), "loan_status", "lending-club"),
        (helpers.load_german_credit_dataset(), "credit_risk", "german-credit"),
    ]
    return ds


# --- Model loading for DiCE-Extended (from pickle the user already creates) ---

def _load_torch_model(model_path: str, in_features: int):
    sd = torch.load(model_path, map_location="cpu")
    sd = {f"model.{k}": v for k, v in sd.items()}
    model = neuralnetworks.PYTModel(in_features)
    model.load_state_dict(sd)
    return model

def _load_tf_model(model_path: str):
    model = neuralnetworks.TF2Model()
    model.load_weights(model_path)
    return model

@task(retries=1, retry_delay_seconds=5)
def load_dice_x_models(pickle_path: Path) -> Dict[str, Dict[str, Any]]:
    import pickle
    logger = get_run_logger()
    logger.info(f"Loading DiCE-X models from {pickle_path}")
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)

    backends = ["sklearn", "PYT", "TF2"]
    dataset_names = ["compas-recidivism", "adult-income", "lending-club", "german-credit"]

    dice_x_models: Dict[str, Dict[str, Any]] = {}
    for name in dataset_names:
        dice_x_models[name] = {}
        for backend in backends:
            if backend == "sklearn":
                dice_x_models[name][backend] = results[name][backend]["model"]
            elif backend == "PYT":
                model_path = os.path.join(
                    "/Users/volk/Documents/bau24-25/thesis/repos/DiCE-X/docs/source/notebooks",
                    results[name][backend]["model_path"]
                )
                num_features = results[name][backend]["metrics"]["num_features"]
                dice_x_models[name][backend] = _load_torch_model(model_path, num_features)
            elif backend == "TF2":
                model_path = os.path.join(
                    "/Users/volk/Documents/bau24-25/thesis/repos/DiCE-X/docs/source/notebooks",
                    results[name][backend]["model_path"]
                )
                dice_x_models[name][backend] = _load_tf_model(model_path)
    return dice_x_models

@task
def load_dice_baseline_models(pickle_path: Optional[Path]) -> Optional[Dict[str, Dict[str, Any]]]:
    if not pickle_path:
        return None
    import pickle
    
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)
    backends = ["sklearn", "PYT", "TF2"]
    dataset_names = ["compas-recidivism", "adult-income", "lending-club", "german-credit"]

    dice_models: Dict[str, Dict[str, Any]] = {}
    for name in dataset_names:
        dice_models[name] = {}
        for backend in backends:
            if backend == "sklearn":
                dice_models[name][backend] = results[name][backend]["model"]
            elif backend == "PYT":
                model_path = os.path.join(
                    "/Users/volk/Documents/bau24-25/thesis/repos/DiCE/docs/source/notebooks",
                    results[name][backend]["model_path"]
                )
                num_features = results[name][backend]["metrics"]["num_features"]
                dice_models[name][backend] = _load_torch_model(model_path, num_features)
            elif backend == "TF2":
                model_path = os.path.join(
                    "/Users/volk/Documents/bau24-25/thesis/repos/DiCE/docs/source/notebooks",
                    results[name][backend]["model_path"]
                )
                dice_models[name][backend] = _load_tf_model(model_path)
    return dice_models
    
    



# -----------------------------
# Lambda grid
# -----------------------------

@task
def build_lambda_grid(phase: str, cfg: GridConfig) -> List[Tuple[float, float, float]]:
    if phase == "coarse":
        vals = np.logspace(cfg.coarse_log_min, cfg.coarse_log_max, num=cfg.coarse_num)
        grid = list(itertools.product(vals, vals, vals))
    else:
        # not used (fine uses neighborhoods); keep for completeness
        grid = []
    if cfg.n_lambdas_limit is not None:
        grid = grid[: cfg.n_lambdas_limit]
    return grid


# -----------------------------
# Metrics (tasks)
# -----------------------------

@task
def compute_validity(exp_obj) -> float:
    return float(exp_obj.get_validity_percentage())

@task
def compute_mad(data_class: "dice_ml.Data", normalized: bool = False) -> Dict[str, float]:
    return data_class.get_valid_mads(normalized=normalized)

@task
def compute_continuous_proximity(C: pd.DataFrame, x: pd.DataFrame, data_class: "dice_ml.Data") -> float:
    mads = data_class.get_valid_mads(normalized=False)
    Xc = C[data_class.continuous_feature_names].values
    xc = x[data_class.continuous_feature_names].iloc[0].values
    mad_vec = np.array([mads[f] for f in data_class.continuous_feature_names]).reshape(1, -1)
    norm_diff = np.abs(Xc - xc) / mad_vec
    mean_per_cf = np.nanmean(norm_diff, axis=1)
    return float(-np.mean(mean_per_cf))  # higher = closer

@task
def compute_categorical_proximity(C: pd.DataFrame, x: pd.DataFrame, data_class: "dice_ml.Data") -> float:
    cats = data_class.categorical_feature_names
    if not cats:
        return 0.0
    xvals = x.iloc[0]
    diff = (C[cats] != xvals[cats]).astype(float).sum(axis=1)
    d_cat = len(cats)
    avg = diff.mean() / (d_cat * len(C)) if d_cat > 0 else 0.0
    return float(1 - avg)

@task
def compute_continuous_diversity(C: pd.DataFrame, data_class: "dice_ml.Data") -> float:
    mads = data_class.get_valid_mads(normalized=False)
    X = C[data_class.continuous_feature_names].values
    if X.size == 0 or X.shape[0] < 2:
        return 0.0
    diff = np.abs(X[:, None, :] - X[None, :, :])
    mad_vec = np.array([mads[f] for f in data_class.continuous_feature_names]).reshape(1, 1, -1)
    norm = diff / mad_vec
    pair = np.nanmean(norm, axis=2)
    iu = np.triu_indices(len(C), k=1)
    return float(np.mean(pair[iu])) if len(iu[0]) else 0.0

@task
def compute_categorical_diversity(C: pd.DataFrame, data_class: "dice_ml.Data") -> float:
    cats = data_class.categorical_feature_names
    if not cats:
        return 0.0
    X = C[cats].values
    if X.size == 0 or X.shape[0] < 2:
        return 0.0
    diff = (X[:, None, :] != X[None, :, :]).astype(np.float32)
    pair = np.mean(diff, axis=2)
    iu = np.triu_indices(len(C), k=1)
    return float(np.mean(pair[iu])) if len(iu[0]) else 0.0

@task
def compute_count_diversity(C: pd.DataFrame) -> float:
    X = C.values if isinstance(C, pd.DataFrame) else C
    if X.size == 0 or X.shape[0] < 2 or X.shape[1] == 0:
        return 0.0
    diff = (X[:, None, :] != X[None, :, :]).astype(np.float32)
    iu = np.triu_indices(X.shape[0], k=1)
    total = np.sum(diff[iu[0], iu[1], :])
    n_pairs = len(iu[0])
    return float(total / (n_pairs * X.shape[1]))

@task
def compute_sparsity(C: pd.DataFrame, x: pd.DataFrame, data_class: "dice_ml.Data") -> float:
    cont = data_class.continuous_feature_names
    cont_CFs = C[cont].to_numpy()
    cont_X = x[cont].to_numpy()
    k, d = cont_CFs.shape
    diff = (cont_CFs != cont_X[0])
    num_changed = diff.sum()
    return float(1 - (num_changed / (k * d)))

@task
def robustness_flip_rate(
    C: pd.DataFrame,
    target_col: str,
    data_iface: "dice_ml.Data",
    backend: str,
    model: Any,
    noise_sd: float = 0.10,
    cat_flip_p: float = 0.20,
    n_repeat: int = 50,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)

    def _predict_class(X_raw: pd.DataFrame) -> np.ndarray:
        if backend == "sklearn":
            return model.predict(X_raw)
        X_enc = data_iface.get_ohe_min_max_normalized_data(X_raw).values
        if backend == "PYT":
            logits = model(torch.tensor(X_enc, dtype=torch.float32)).detach().cpu().numpy().ravel()
        elif backend == "TF2":
            logits = model.predict(tf.constant(X_enc, dtype=tf.float32)).ravel()
        else:
            raise ValueError(f"Unknown backend {backend}")
        return (logits >= 0.5).astype(int)

    X_raw = C.drop(columns=[target_col], errors="ignore").reset_index(drop=True)
    y_orig = _predict_class(X_raw)

    cont_cols = data_iface.continuous_feature_names
    cat_cols  = data_iface.categorical_feature_names
    ranges    = data_iface.get_features_range_float()[1]
    cat_vals  = {c: data_iface.get_features_range()[1][c] for c in cat_cols}

    n_cf   = len(C)
    kept   = 0

    for _ in range(n_repeat):
        X_noisy = X_raw.copy()
        for col in cont_cols:
            lo, hi = ranges[col]
            span   = hi - lo
            X_noisy[col] = np.clip(
                X_noisy[col] + rng.normal(0, noise_sd * span, size=n_cf),
                lo, hi
            )
        for col in cat_cols:
            X_noisy[col] = X_noisy[col].astype(object)
            mask = rng.random(n_cf) < cat_flip_p
            if mask.any():
                X_noisy.loc[mask, col] = rng.choice(cat_vals[col], size=mask.sum())
        y_noisy = _predict_class(X_noisy)
        kept += np.sum(y_noisy == y_orig)

    return float(kept / (n_cf * n_repeat))


# -----------------------------
# DiCE object factory & CF generation
# -----------------------------

@task
def make_dice_objects(
    train_df: pd.DataFrame,
    target: str,
    dataset_name: str,
    backend: str,
    method_label: str,
    dice_x_models: Dict[str, Dict[str, Any]],
    dice_baseline_models: Optional[Dict[str, Dict[str, Any]]],
):
    if method_label == "DiCE-Ext":
        exp_module = dice_ml_x
        models = dice_x_models
    else:
        exp_module = dice_ml
        if dice_baseline_models is None:
            raise RuntimeError("Baseline models not provided but 'DiCE' method requested.")
        models = dice_baseline_models

    cont_feats = list(train_df.select_dtypes(include=np.number).columns.difference([target]))
    d = exp_module.Data(
        dataframe=train_df,
        continuous_features=cont_feats,
        outcome_name=target,
    )

    model_opts: Dict[str, Any] = dict(model=models[dataset_name][backend], backend=backend)
    method = "genetic" if backend == "sklearn" else "gradient"

    if backend != "sklearn":
        model_opts["func"] = "ohe-min-max"
        if backend == "PYT":
            model_opts["model"] = models[dataset_name][backend].model

    m = exp_module.Model(**model_opts)
    exp = exp_module.Dice(d, m, method=method)

    model_for_pred = m.model
    return exp_module, exp, d, method, model_for_pred

@task
def gen_kwargs_for_method(
    method_label: str,
    total_cfs: int,
    lam1: float, lam2: float, lam3: float,
    num_bins: int,
    backend: str,
) -> Dict[str, Any]:
    base_opts: Dict[str, Any] = {}
    if backend == "sklearn":
        base_opts["maxiterations"] = 500
    out = dict(
        total_CFs=total_cfs,
        desired_class="opposite",
        proximity_weight=lam1,
        diversity_weight=lam2,
        **base_opts,
    )
    if method_label == "DiCE-Ext":
        out["preprocessing_bins"] = num_bins
        out["robustness_weight"] = lam3
    return out

@task
def evaluate_one_setting(
    exp, data_iface, target: str, backend: str, model_for_pred: Any,
    test_df: pd.DataFrame, gen_kwargs: Dict[str, Any],
    robustness_cfg: Tuple[float, float, int, int],
) -> List[Dict[str, Any]]:
    noise_sd, cat_flip_p, n_repeat, seed = robustness_cfg
    rows: List[Dict[str, Any]] = []
    for q_idx, (_, row) in enumerate(test_df.iterrows()):
        x_full = row.to_frame().T
        x_query = x_full.drop(columns=[target])
        try:
            exp_obj = exp.generate_counterfactuals(x_query, **gen_kwargs)
            C = exp_obj.to_dataframe()
            if C.empty:
                rows.append(dict(q_idx=q_idx, error="empty_cf_set"))
                continue
            na_cols = C.columns[C.isna().any()]
            if len(na_cols) > 0:
                C[na_cols] = C[na_cols].fillna(x_query[na_cols].iloc[0])

            try:
                validity = float(exp_obj.get_validity_percentage())
            except Exception:
                validity = math.nan
            try:
                prox_cont = compute_continuous_proximity.fn(C, x_full, data_iface)
            except Exception:
                prox_cont = math.nan
            try:
                prox_cat = compute_categorical_proximity.fn(C, x_full, data_iface)
            except Exception:
                prox_cat = math.nan
            try:
                div_cont = compute_continuous_diversity.fn(C, data_iface)
            except Exception:
                div_cont = math.nan
            try:
                div_cat = compute_categorical_diversity.fn(C, data_iface)
            except Exception:
                div_cat = math.nan
            try:
                div_count = compute_count_diversity.fn(C)
            except Exception:
                div_count = math.nan
            try:
                spars = compute_sparsity.fn(C, x_full, data_iface)
            except Exception:
                spars = math.nan
            try:
                robust = robustness_flip_rate.fn(
                    C=C, target_col=target, data_iface=data_iface,
                    backend=backend, model=model_for_pred,
                    noise_sd=noise_sd, cat_flip_p=cat_flip_p, n_repeat=n_repeat, seed=seed
                )
            except Exception:
                robust = math.nan

            rows.append(dict(
                q_idx=q_idx,
                validity=validity,
                prox_cont_mad_negmean=prox_cont,
                prox_cat_similarity=prox_cat,
                div_cont_mad=div_cont,
                div_cat_hamming=div_cat,
                div_count=div_count,
                sparsity_cont=spars,
                robustness_keep_rate=robust,
                error="",
            ))
        except Exception as e:
            rows.append(dict(q_idx=q_idx, error=str(e)))
    return rows


# -----------------------------
# Aggregation & persistence
# -----------------------------

@task
def aggregate_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "validity": ["mean", "std"],
        "prox_cont_mad_negmean": ["mean", "std"],
        "prox_cat_similarity": ["mean", "std"],
        "div_cont_mad": ["mean", "std"],
        "div_cat_hamming": ["mean", "std"],
        "div_count": ["mean", "std"],
        "sparsity_cont": ["mean", "std"],
        "robustness_keep_rate": ["mean", "std"],
    }
    grouped = (rows_df[rows_df["error"].fillna("") == ""]
               .groupby(["phase","dataset","backend","method","lambda1","lambda2","lambda3"], dropna=False)
               .agg(agg_spec))
    grouped.columns = [f"{m}_{s}" for (m, s) in grouped.columns]
    grouped = grouped.reset_index()
    return grouped

@task
def save_df(df: pd.DataFrame, path: Path, msg: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# -----------------------------
# Main flow
# -----------------------------

@flow(name="DiCE Lambda Grid Search", task_runner=SequentialTaskRunner())
def lambda_grid_flow(
    out_dir: Path = DefaultPaths().out_dir,
    dice_x_pickle: Optional[Path] = DefaultPaths().dice_x_pickle,
    dice_baseline_pickle: Optional[Path] = DefaultPaths().dice_baseline_pickle,
    backends: List[str] = ("sklearn", "PYT", "TF2"),
    cfg: GridConfig = GridConfig(),
):
    logger = get_run_logger()
    np.random.seed(cfg.random_seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets/models
    datasets = load_datasets()
    if cfg.limit_datasets:
        datasets = [d for d in datasets if d[2] in set(cfg.limit_datasets)]
    if cfg.limit_backends:
        backends = [b for b in backends if b in set(cfg.limit_backends)]

    models_x = load_dice_x_models(dice_x_pickle)
    models_base = load_dice_baseline_models(dice_baseline_pickle)

    all_agg = []


    # ---------- PHASE: COARSE ----------
    coarse_lambdas = build_lambda_grid("coarse", cfg)
    logger.info(f"[coarse] λ-grid size = {len(coarse_lambdas)}")

    for df, target, ds_name in datasets:
        from sklearn.model_selection import train_test_split
        target_col = df[target]
        train_df, test_df_full = train_test_split(
            df, test_size=0.2, random_state=cfg.random_seed, stratify=target_col
        )
        if len(test_df_full) > cfg.n_test_points:
            test_df_full = test_df_full.sample(n=cfg.n_test_points, random_state=cfg.random_seed)

        expected_rows_coarse = len(coarse_lambdas) * len(test_df_full)

        for backend in backends:
            for method_label in ("DiCE-Ext", "DiCE"):
                if method_label == "DiCE" and models_base is None:
                    logger.warning("Baseline models not available; skipping DiCE baseline.")
                    continue

                # SKIP WHOLE GROUP IF COMPLETE
                if _has_complete_group(out_dir, "coarse", ds_name, backend, method_label, expected_rows_coarse):
                    logger.info(f"[skip/coarse] {ds_name}/{backend}/{method_label} already has >= {expected_rows_coarse} rows. Skipping.")
                    continue

                # Compute *remaining* λ and partial cursors up front
                todo_lambdas, partial_next_q = _remaining_lambdas(
                    coarse_lambdas, out_dir, "coarse", ds_name, backend, method_label,
                    n_test_points=len(test_df_full)
                )
                logger.info(f"[resume/coarse] {ds_name}/{backend}/{method_label}: "
                            f"remaining λ={len(todo_lambdas)}, partials={len(partial_next_q)} "
                            f"(grid size was {len(coarse_lambdas)})")

                if not todo_lambdas:
                    logger.info(f"[resume/coarse] Nothing to do for {ds_name}/{backend}/{method_label}.")
                    continue

                _, exp, data_iface, _, model_pred = make_dice_objects(
                    train_df, target, ds_name, backend, method_label, models_x, models_base
                )
                part_path = _part_path(out_dir, "coarse", ds_name, backend, method_label)

                pending_batch: List[Dict[str, Any]] = []

                for (lam1, lam2, lam3) in todo_lambdas:
                    start_q = partial_next_q.get((float(lam1), float(lam2), float(lam3)), 0)
                    if start_q >= len(test_df_full):
                        continue
                    test_df = test_df_full.iloc[start_q:].copy()

                    gen_kwargs = gen_kwargs_for_method(
                        method_label, cfg.k_cfs, lam1, lam2, lam3, backend
                    )
                    rows = evaluate_one_setting(
                        exp=exp, data_iface=data_iface, target=target, backend=backend,
                        model_for_pred=model_pred, test_df=test_df, gen_kwargs=gen_kwargs,
                        robustness_cfg=(cfg.noise_sd, cfg.cat_flip_p, cfg.n_repeat, cfg.random_seed),
                    )

                    for r in rows:
                        r["q_idx"] = int(r["q_idx"]) + int(start_q)
                        r.update(dict(
                            phase="coarse", dataset=ds_name, backend=backend, method=method_label,
                            lambda1=float(lam1), lambda2=float(lam2), lambda3=float(lam3),
                        ))
                    pending_batch.extend(rows)

                    while len(pending_batch) >= cfg.save_every_rows:
                        batch = pd.DataFrame(pending_batch[:cfg.save_every_rows])
                        _append_csv(part_path, batch)
                        pending_batch = pending_batch[cfg.save_every_rows:]

                # flush tail
                if pending_batch:
                    _append_csv(part_path, pd.DataFrame(pending_batch))

    # Consolidate all coarse rows and aggregate
    coarse_all = _load_all_phase_rows(out_dir, "coarse")
    coarse_rows_path = out_dir / f"coarse__rows_{int(time.time())}.csv"
    save_df(coarse_all, coarse_rows_path)
    coarse_agg = aggregate_rows(coarse_all) if not coarse_all.empty else pd.DataFrame()
    if not coarse_agg.empty:
        coarse_agg_path = out_dir / f"coarse__agg_{int(time.time())}.csv"
        save_df(coarse_agg, coarse_agg_path)
        all_agg.append(coarse_agg)
        create_markdown_artifact(
            key="coarse-summary",
            markdown=f"**coarse**: {len(coarse_all)} rows → {coarse_rows_path.name}, aggregated → {coarse_agg_path.name}",
            description="Lambda grid coarse phase summary"
        )
    else:
        create_markdown_artifact(
            key="coarse-summary",
            markdown=f"**coarse**: no rows found",
            description="Lambda grid coarse phase summary"
        )

    # ---------- Select top seeds for FINE ----------
    seeds_by_group = pick_top_seeds(coarse_agg, cfg) if not coarse_agg.empty else {}

    # ---------- PHASE: FINE (local neighborhoods around seeds) ----------
    for df, target, ds_name in datasets:
        from sklearn.model_selection import train_test_split
        target_col = df[target]
        train_df, test_df_full = train_test_split(
            df, test_size=0.2, random_state=cfg.random_seed, stratify=target_col
        )
        if len(test_df_full) > cfg.n_test_points:
            test_df_full = test_df_full.sample(n=cfg.n_test_points, random_state=cfg.random_seed)

        for backend in backends:
            for method_label in ("DiCE-Ext", "DiCE"):
                if method_label == "DiCE" and models_base is None:
                    continue

                key = (ds_name, backend, method_label)
                if key not in seeds_by_group or len(seeds_by_group[key]) == 0:
                    continue

                # Build local neighborhoods
                local_triples: List[Tuple[float,float,float]] = []
                for seed in seeds_by_group[key]:
                    local_triples.extend(make_local_neighborhood(seed, cfg.fine_scale))
                if cfg.n_lambdas_limit is not None:
                    local_triples = local_triples[: cfg.n_lambdas_limit]

                expected_rows_fine = len(local_triples) * len(test_df_full)

                # SKIP WHOLE GROUP IF COMPLETE
                if _has_complete_group(out_dir, "fine", ds_name, backend, method_label, expected_rows_fine):
                    logger.info(f"[skip/fine] {ds_name}/{backend}/{method_label} already has >= {expected_rows_fine} rows. Skipping.")
                    continue

                # Only the remaining λ's
                todo_lambdas, partial_next_q = _remaining_lambdas(
                    local_triples, out_dir, "fine", ds_name, backend, method_label,
                    n_test_points=len(test_df_full)
                )
                logger.info(f"[resume/fine] {ds_name}/{backend}/{method_label}: "
                            f"remaining λ={len(todo_lambdas)}, partials={len(partial_next_q)} "
                            f"(grid size was {len(local_triples)})")

                if not todo_lambdas:
                    logger.info(f"[resume/fine] Nothing to do for {ds_name}/{backend}/{method_label}.")
                    continue

                _, exp, data_iface, _, model_pred = make_dice_objects(
                    train_df, target, ds_name, backend, method_label, models_x, models_base
                )
                part_path = _part_path(out_dir, "fine", ds_name, backend, method_label)

                pending_batch: List[Dict[str, Any]] = []

                for (lam1, lam2, lam3) in todo_lambdas:
                    start_q = partial_next_q.get((float(lam1), float(lam2), float(lam3)), 0)
                    if start_q >= len(test_df_full):
                        continue
                    test_df = test_df_full.iloc[start_q:].copy()

                    gen_kwargs = gen_kwargs_for_method(
                        method_label, cfg.k_cfs, lam1, lam2, lam3, backend
                    )
                    rows = evaluate_one_setting(
                        exp=exp, data_iface=data_iface, target=target, backend=backend,
                        model_for_pred=model_pred, test_df=test_df, gen_kwargs=gen_kwargs,
                        robustness_cfg=(cfg.noise_sd, cfg.cat_flip_p, cfg.n_repeat, cfg.random_seed),
                    )

                    for r in rows:
                        r["q_idx"] = int(r["q_idx"]) + int(start_q)
                        r.update(dict(
                            phase="fine", dataset=ds_name, backend=backend, method=method_label,
                            lambda1=float(lam1), lambda2=float(lam2), lambda3=float(lam3),
                        ))
                    pending_batch.extend(rows)

                    while len(pending_batch) >= cfg.save_every_rows:
                        batch = pd.DataFrame(pending_batch[:cfg.save_every_rows])
                        _append_csv(part_path, batch)
                        pending_batch = pending_batch[cfg.save_every_rows:]

                if pending_batch:
                    _append_csv(part_path, pd.DataFrame(pending_batch))

    fine_all = _load_all_phase_rows(out_dir, "fine")
    fine_rows_path = out_dir / f"fine__rows_{int(time.time())}.csv"
    save_df(fine_all, fine_rows_path)
    fine_agg = aggregate_rows(fine_all) if not fine_all.empty else pd.DataFrame()
    if not fine_agg.empty:
        fine_agg_path = out_dir / f"fine__agg_{int(time.time())}.csv"
        save_df(fine_agg, fine_agg_path)
        all_agg.append(fine_agg)
        create_markdown_artifact(
            key="fine-summary",
            markdown=f"**fine**: {len(fine_all)} rows → {fine_rows_path.name}, aggregated → {fine_agg_path.name}",
            description="Lambda grid fine phase summary"
        )
    else:
        create_markdown_artifact(
            key="fine-summary",
            markdown=f"**fine**: no rows found",
            description="Lambda grid fine phase summary"
        )

    _finalize_and_rank(out_dir=out_dir, all_agg=all_agg, logger=logger, cfg=cfg)

    # ---------- Combine + rank ----------
    combined = pd.concat(all_agg, ignore_index=True) if all_agg else pd.DataFrame()
    if not combined.empty:
        combined["score"] = composite_score(combined)
        ranked = (combined.sort_values(["dataset","backend","method","score"], ascending=[True,True,True,False])
                          .groupby(["dataset","backend","method"], as_index=False, group_keys=False)
                          .head(10))
        ranked_path = out_dir / f"ranked_top10_{int(time.time())}.csv"
        save_df(ranked, ranked_path)
        logger.info(f"Saved ranked top-10 per dataset/backend/method → {ranked_path}")

    # ---------- Optional: high-fidelity re-eval on top-3 ----------
    if cfg.run_hi_fidelity and not combined.empty:
        for df, target, ds_name in datasets:
            from sklearn.model_selection import train_test_split
            target_col = df[target]
            train_df, test_df_full = train_test_split(
                df, test_size=0.2, random_state=cfg.random_seed, stratify=target_col
            )
            if len(test_df_full) > cfg.hi_n_test_points:
                test_df_full = test_df_full.sample(n=cfg.hi_n_test_points, random_state=cfg.random_seed)

            for backend in backends:
                for method_label in ("DiCE-Ext", "DiCE"):
                    if method_label == "DiCE" and models_base is None:
                        continue

                    final_top = (combined.sort_values(["dataset","backend","method","score"], ascending=[True,True,True,False])
                                         .query("dataset == @ds_name and backend == @backend and method == @method_label")
                                         .head(cfg.hi_top_n))
                    if final_top.empty:
                        continue

                    expected_rows_hi = len(final_top) * len(test_df_full)

                    # SKIP WHOLE GROUP IF COMPLETE
                    if _has_complete_group(out_dir, "hi", ds_name, backend, method_label, expected_rows_hi):
                        logger.info(f"[skip/hi] {ds_name}/{backend}/{method_label} already has >= {expected_rows_hi} rows. Skipping.")
                        continue

                    _, exp, data_iface, _, model_pred = make_dice_objects(
                        train_df, target, ds_name, backend, method_label, models_x, models_base
                    )
                    part_path = _part_path(out_dir, "hi", ds_name, backend, method_label)

                    # Resume existing rows for hi
                    existing = _load_group_rows(out_dir, "hi", ds_name, backend, method_label)
                    done_triples, partial_next_q = _resume_map(existing, n_test_points=len(test_df_full))
                    logger.info(f"[resume/hi] {ds_name}/{backend}/{method_label}: "
                                f"{len(done_triples)} λ done, {len(partial_next_q)} partial.")

                    pending_batch: List[Dict[str, Any]] = []

                    for _, row in final_top.iterrows():
                        lam1, lam2, lam3 = float(row.lambda1), float(row.lambda2), float(row.lambda3)
                        lam_key = (lam1, lam2, lam3)
                        if lam_key in done_triples:
                            continue

                        start_q = partial_next_q.get(lam_key, 0)
                        if start_q >= len(test_df_full):
                            continue
                        test_df = test_df_full.iloc[start_q:].copy()

                        gen_kwargs = gen_kwargs_for_method(method_label, cfg.hi_k_cfs, lam1, lam2, lam3, backend)
                        rows = evaluate_one_setting(
                            exp=exp, data_iface=data_iface, target=target, backend=backend,
                            model_for_pred=model_pred, test_df=test_df, gen_kwargs=gen_kwargs,
                            robustness_cfg=(cfg.noise_sd, cfg.cat_flip_p, cfg.hi_n_repeat, cfg.random_seed),
                        )

                        for r in rows:
                            r["q_idx"] = int(r["q_idx"]) + int(start_q)
                            r.update(dict(
                                phase="hi", dataset=ds_name, backend=backend, method=method_label,
                                lambda1=lam1, lambda2=lam2, lambda3=lam3,
                            ))
                        pending_batch.extend(rows)

                        while len(pending_batch) >= cfg.save_every_rows:
                            batch = pd.DataFrame(pending_batch[:cfg.save_every_rows])
                            _append_csv(part_path, batch)
                            pending_batch = pending_batch[cfg.save_every_rows:]

                    if pending_batch:
                        _append_csv(part_path, pd.DataFrame(pending_batch))

        hi_all = _load_all_phase_rows(out_dir, "hi")
        hi_rows_path = out_dir / f"hi__rows_{int(time.time())}.csv"
        save_df(hi_all, hi_rows_path)
        hi_agg = aggregate_rows(hi_all) if not hi_all.empty else pd.DataFrame()
        if not hi_agg.empty:
            hi_agg_path = out_dir / f"hi__agg_{int(time.time())}.csv"
            save_df(hi_agg, hi_agg_path)
            logger.info(f"Saved hi-fidelity aggregates → {hi_agg_path}")

    




# -----------------------------
# Script entrypoint
# -----------------------------
if __name__ == "__main__":
    lambda_grid_flow()  # run with defaults
