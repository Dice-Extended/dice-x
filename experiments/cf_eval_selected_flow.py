from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd
import csv
import math

dice_path    = "/Users/volk/Documents/bau24-25/thesis/repos/DiCE"
dice_x_path  = "/Users/volk/Documents/bau24-25/thesis/repos/DiCE-X"
for p in (dice_x_path, dice_path):
    if p not in sys.path:
        sys.path.insert(0, p)

import dice_ml
from dice_ml.utils import helpers as dhelpers, neuralnetworks as dnn
import dice_ml_x
from dice_ml_x.utils import helpers as xhelpers, neuralnetworks as xnn

import torch, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import OrderedDict, defaultdict


from prefect import flow, task, get_run_logger

from experiments.grid_search_experiment import (
    load_datasets,
    load_dice_x_models,
    load_dice_baseline_models,
    DefaultPaths,
)

# =========================
# Config
# =========================
@dataclass
class EvalConfig:
    n_test_points: int = 10
    k_cfs: int = 5
    random_seed: int = 42
    n_samples_fidelity: int = 1000
    radius_set: tuple = (0.5, 1.0, 2.0)

@dataclass
class Paths:
    out_dir: Path = Path("monte_carlo_results/metrics")
    optuna_dir: Path = DefaultPaths().out_dir / "optuna_mo"

# =========================
# Model loaders
# =========================
def load_torch_model(model_path: str | Path, in_features: int):
    state = torch.load(model_path)
    state = {f"model.{k}": v for k, v in state.items()}
    model = dnn.PYTModel(in_features)
    model.load_state_dict(state)
    return model

def load_tensorflow_model(model_path: str | Path):
    model = dnn.TF2Model()
    model.load_weights(str(model_path))
    return model

# =========================
# Helpers
# =========================

def merge_selected_results(result_path: str, out_path: str = "selected_lambdas") -> Path:

    root = Path(result_path)
    out_dir = (root / out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "selected_lambdas.csv"

    files: List[Path] = sorted(root.rglob("*__SELECTED.csv"))

    if not files:
        out_file.write_text("")
        return out_file
    
    all_rows = []
    all_fields = set()

    for f in files:
        try:
            with f.open("r", newline="") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    continue
                all_fields.update(reader.fieldnames)
                for row in reader:
                    if not any(row.values()):
                        continue
                    all_rows.append(row)
        except Exception as e:
            print(f"[MERGE] skipped {f}: {e}")

    if not all_rows:
        out_file.write_text("")
        return out_file
    
    preferred = [
        "dataset", "backend", "method",
        "lambda1", "lambda2", "lambda3",
        "robustness_keep_rate_mean",
        "div_cont_mad_mean",
        "prox_cont_mad_negmean_mean",
        "sparsity_cont_mean",
    ]

    remaining = [c for c in sorted(all_fields) if c not in preferred]
    fieldnames = [c for c in preferred if c in all_fields] + remaining

    with out_file.open("w", newline="") as outfh:
        writer = csv.DictWriter(outfh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            out_row = {}
            for k in fieldnames:
                v = row.get(k, "")
                if k.startswith("lambda") and v != "":
                    try:
                        out_row[k] = round(float(v), 2)
                    except Exception:
                        out_row[k] = v
                else:
                    out_row[k] = v
            writer.writerow(out_row)

    return out_file

def read_selected_table(merged_csv: Path) -> pd.DataFrame:
    """
    Load merged selected_lambdas.csv and return it as-is.
    Expected columns: dataset, backend, method, lambda1, lambda2, lambda3, ...
    """
    df = pd.read_csv(merged_csv)
    if df.empty:
        return df

    for col in ("lambda1", "lambda2", "lambda3"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "method" in df.columns and "lambda3" in df.columns:
        df.loc[df["method"] == "DiCE", "lambda3"] = 0.0

    return df

# =========================
# Metrics (your functions, trimmed to essentials)
# =========================
def compute_validity(exp) -> float:
    return exp.get_validity_percentage()

def compute_mad(data_class: dice_ml.Data, normalized=False) -> dict:
    return data_class.get_valid_mads(normalized=normalized)

def compute_continuous_proximity(C: pd.DataFrame, x: pd.DataFrame, data_class: dice_ml.Data) -> float:
    cont_feats = data_class.continuous_feature_names
    if len(cont_feats) == 0:
        return 0.0
    mads = compute_mad(data_class)
    diffs = np.abs(C[cont_feats].values - x[cont_feats].iloc[0].values)

    denom = np.array([mads[f] for f in cont_feats]) * len(C)
    norm = diffs / denom
    mean_per_cf = np.nanmean(norm, axis=1)
    return -float(np.mean(mean_per_cf))

def compute_categorical_proximity(C: pd.DataFrame, x: pd.DataFrame, data_class: dice_ml.Data) -> float:
    cats = data_class.categorical_feature_names
    if len(cats) == 0:
        return 0.0
    diff = (C[cats] != x[cats].iloc[0]).sum(axis=1)
    avg = diff.mean() / len(cats)
    return 1.0 - float(avg)

def compute_cont_count_diversity(C: pd.DataFrame, data_class: dice_ml.Data) -> float:
    feats = data_class.continuous_feature_names

    if len(feats) == 0 or len(C) < 2:
        return 0.0
    
    k = len(C)
    d = len(feats)

    X = C[feats].values    # shape: (k, d)
    diff_indicator = (X[:, None, :] != X[None, :, :]).astype(np.float32)   # shape: (k, k, d) 

    count_per_pair = diff_indicator.sum(axis=2)

    iu = np.triu_indices(k, k=1)

    total_diff_count = count_per_pair[iu].sum()
    n_pairs = len(iu[0])

    if n_pairs > 0 and d > 0:
        diversity = float(total_diff_count / (n_pairs * d))
    else:
        diversity = 0.0

    return diversity

def compute_continuous_diversity(C: pd.DataFrame, data_class: dice_ml.Data) -> float:
    feats = data_class.continuous_feature_names
    if len(feats) == 0 or len(C) < 2:
        return 0.0
    mads = compute_mad(data_class)
    X = C[feats].values
    diff = np.abs(X[:, None, :] - X[None, :, :])
    denom = np.array([mads[f] for f in feats]).reshape(1, 1, -1)
    norm = diff / denom
    pair = np.nanmean(norm, axis=2)
    iu = np.triu_indices(len(C), 1)
    return float(np.mean(pair[iu])) if len(iu[0]) else 0.0

def compute_categorical_diversity(C: pd.DataFrame, data_class: dice_ml.Data) -> float:
    feats = data_class.categorical_feature_names
    if len(feats) == 0 or len(C) < 2:
        return 0.0
    X = C[feats].values
    diff = (X[:, None, :] != X[None, :, :]).astype(np.float32)
    pair = diff.mean(axis=2)
    iu = np.triu_indices(len(C), 1)
    return float(np.mean(pair[iu])) if len(iu[0]) else 0.0

def compute_sparsity(C: pd.DataFrame, x: pd.DataFrame, data_class: dice_ml.Data) -> float:
    feats = data_class.continuous_feature_names
    if len(feats) == 0:
        return 0.0
    dif = (C[feats].to_numpy() != x[feats].to_numpy()[0])
    changed = dif.sum()
    k, d = C[feats].shape
    return 1.0 - (changed / (k * d))

def robustness_flip_rate(
    C: pd.DataFrame,
    target_col: str,
    data_iface: dice_ml.Data,
    backend: str,
    model,
    noise_sd: float = 0.10,
    cat_flip_p: float = 0.20,
    n_repeat: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> float:
    rng = rng or np.random.default_rng()
    def _predict_class(X_raw: pd.DataFrame) -> np.ndarray:
        if backend == "sklearn":
            return model.predict(X_raw)
        X_enc = data_iface.get_ohe_min_max_normalized_data(X_raw).values
        if backend == "PYT":
            logits = model(torch.tensor(X_enc, dtype=torch.float32)).detach().cpu().numpy().ravel()
        else:
            logits = model.predict(tf.constant(X_enc, dtype=tf.float32)).ravel()
        return (logits >= 0.5).astype(int)

    X_raw = C.drop(columns=[target_col], errors="ignore").reset_index(drop=True)
    y_orig = _predict_class(X_raw)

    cont_cols = data_iface.continuous_feature_names
    cat_cols  = data_iface.categorical_feature_names
    ranges    = data_iface.get_features_range_float()[1]
    cat_vals  = {c: data_iface.get_features_range()[1][c] for c in cat_cols}

    kept = 0
    n_cf = len(C)
    for _ in range(n_repeat):
        X_noisy = X_raw.copy()
        for col in cont_cols:
            lo, hi = ranges[col]; span = hi - lo
            X_noisy[col] = np.clip(
                X_noisy[col] + rng.normal(0, noise_sd * span, size=n_cf),
                lo, hi
            )
        for col in cat_cols:
            X_noisy[col] = X_noisy[col].astype(object)
            mask = rng.random(n_cf) < cat_flip_p
            if mask.any():
                X_noisy.loc[mask, col] = rng.choice(cat_vals[col], size=mask.sum())
        kept += np.sum(_predict_class(X_noisy) == y_orig)
    return float(kept / (n_cf * n_repeat))

def one_nn_fidelity(
    x_df: pd.DataFrame,
    cfs_df: pd.DataFrame,
    data_interface,
    model,
    backend: str,
    radius_mad: float,
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    rng = rng or np.random.default_rng()
    OUTCOME = data_interface.outcome_name
    def predict_classes(df_raw: pd.DataFrame) -> np.ndarray:
        X_raw = df_raw.drop(columns=[OUTCOME], errors="ignore")
        if backend == "sklearn":
            out = model.predict(X_raw)
            if out.ndim == 2 or (out.ndim == 1 and np.issubdtype(out.dtype, np.floating)):
                out = (out >= 0.5).astype(int).ravel()
            return out.astype(int)
        enc = data_interface.get_ohe_min_max_normalized_data(X_raw).values
        if backend == "PYT":
            preds = model(torch.tensor(enc, dtype=torch.float32)).detach().cpu().numpy().ravel()
        else:
            preds = model.predict(tf.constant(enc, dtype=tf.float32)).ravel()
        return (preds >= 0.5).astype(int)

    train_raw = pd.concat([x_df, cfs_df], ignore_index=True).drop(columns=[OUTCOME], errors="ignore")
    y_train   = predict_classes(train_raw)

    if backend == "sklearn":
        train_df = pd.get_dummies(train_raw).fillna(train_raw.mean(numeric_only=True))
        cols_ref = train_df.columns
        X_train  = train_df.values
    else:
        X_train  = data_interface.get_ohe_min_max_normalized_data(train_raw).values
        cols_ref = None

    knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)

    cont_cols = data_interface.continuous_feature_names
    cat_cols  = data_interface.categorical_feature_names
    mad_vals  = data_interface.get_valid_mads()
    cat_levels = {c: data_interface.get_features_range()[1][c] for c in cat_cols}

    rows = []
    x0 = x_df.drop(columns=[OUTCOME], errors="ignore").copy()
    for _ in range(n_samples):
        r = x0.copy()
        for c in cont_cols:
            r.at[r.index[0], c] += rng.uniform(-mad_vals[c]*radius_mad, mad_vals[c]*radius_mad)
        for c in cat_cols:
            r.at[r.index[0], c] = rng.choice(cat_levels[c])
        rows.append(r)
    synth_raw = pd.concat(rows, ignore_index=True)

    if backend == "sklearn":
        X_s = pd.get_dummies(synth_raw)
        for c in cols_ref:
            if c not in X_s.columns:
                X_s[c] = 0
        X_s = X_s[cols_ref].fillna(X_s.mean(numeric_only=True)).values
    else:
        X_s = data_interface.get_ohe_min_max_normalized_data(synth_raw).values

    orig = predict_classes(synth_raw)
    proxy = knn.predict(X_s)
    return float(np.mean(orig == proxy))

# =========================
# Read λ* from SELECTED.csv
# =========================
def read_selected_lambda(optuna_dir: Path, ds: str, backend: str, method: str) -> Optional[dict]:
    sel = optuna_dir / "selected_lambdas/selected_lambdas.csv"
    if not sel.exists():
        return None
    df = pd.read_csv(sel)
    if df.empty:
        return None
    row = df.iloc[0]
    return dict(lambda1=float(row["lambda1"]),
                lambda2=float(row["lambda2"]),
                lambda3=float(row.get("lambda3", 0.0)))

# =========================
# Evaluate ONE group with selected λ*
# =========================
@task
def evaluate_group_with_selected(
    df: pd.DataFrame, target: str, ds_name: str, backend: str, method_label: str,
    dice_models, dice_x_models,
    lambdas: dict, ecfg: EvalConfig, out_dir: Path
) -> Optional[dict]:
    logger = get_run_logger()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_vectors"; raw_dir.mkdir(parents=True, exist_ok=True)

    # resumable per-group CSV
    group_csv = out_dir / f"{ds_name}__{backend}__{method_label}.csv"
    if group_csv.exists():
        logger.info(f"[skip] {group_csv.name} already present.")
        return pd.read_csv(group_csv).iloc[0].to_dict()

    # sample fixed test subset
    y = df[target]
    train_df, test_df = train_test_split(df, test_size=0.2,
                                         random_state=ecfg.random_seed, stratify=y)

    # choose DiCE vs DiCE-X
    if method_label == "DiCE-Ext":
        exp_mod = dice_ml_x; models = dice_x_models
    else:
        exp_mod = dice_ml; models = dice_models

    d = exp_mod.Data(
        dataframe=train_df,
        continuous_features=list(train_df.select_dtypes(include=np.number).columns.difference([target])),
        outcome_name=target
    )

    # backend model
    m_opts = OrderedDict(model=models[ds_name][backend], backend=backend)
    method = 'genetic' if backend == 'sklearn' else 'gradient'
    if backend != 'sklearn':
        m_opts['func'] = 'ohe-min-max'
        if backend == 'PYT':
            m_opts['model'] = models[ds_name][backend].model

    m = exp_mod.Model(**m_opts)
    exp = exp_mod.Dice(d, m, method=method)

    # translate λ* → kwargs (λ1 prox, λ2 div, λ3 rob)
    gen_kwargs = dict(
        total_CFs=ecfg.k_cfs,
        proximity_weight=lambdas["lambda1"],
        diversity_weight=lambdas["lambda2"],
        maxiterations=500 if backend == "sklearn" else None,
    )
    if method_label == "DiCE-Ext":
        gen_kwargs["robustness_weight"] = lambdas.get("lambda3", 0.0)

    # pick N test points
    rng = np.random.default_rng(ecfg.random_seed)
    idxs = rng.choice(len(test_df), size=min(ecfg.n_test_points, len(test_df)), replace=False)

    metrics = defaultdict(list)
    for i in idxs:
        x_full  = test_df.iloc[i:i+1]
        x_query = x_full.drop(columns=[target])

        try:
            dice_exp = exp.generate_counterfactuals(x_query, **{k:v for k,v in gen_kwargs.items() if v is not None})
            C        = dice_exp.to_dataframe()
        except Exception as e:
            logger.warning(f"[warn] {ds_name}-{backend}-{method_label}: idx={i} failed ({e})")
            continue

        if C.empty:
            logger.warning(f"[warn] {ds_name}-{backend}-{method_label}: idx={i} empty CF set")
            continue

        # fill NaNs with original feature values
        na_cols = C.columns[C.isna().any()]
        if len(na_cols):
            C[na_cols] = C[na_cols].fillna(x_query[na_cols].iloc[0])

        # metrics
        vld = compute_validity(exp)
        continuous_prox = compute_continuous_proximity(C, x=x_full, data_class=d)
        categorical_prox = compute_categorical_proximity(C, x=x_full, data_class=d)
        spars = compute_sparsity(C, x=x_full, data_class=d)
        cdiv = compute_continuous_diversity(C, data_class=d)
        cont_count_diversity = compute_cont_count_diversity(C, data_class=d)
        stab = robustness_flip_rate(C, target, d, backend, m.model)

        metrics["validity"].append(vld)
        metrics["cont_prox"].append(continuous_prox)
        metrics["cat_prox"].append(categorical_prox)
        metrics["sparsity"].append(spars)
        metrics["diversity"].append(cdiv)
        metrics["cont_count_diversity"].append(cont_count_diversity)
        metrics["robustness"].append(stab)

        for r in ecfg.radius_set:
            f1 = one_nn_fidelity(
                x_query, C, d, m.model, backend=backend,
                radius_mad=r, n_samples=ecfg.n_samples_fidelity
            )
            metrics[f"fidelity_1nn_{r}"].append(f1)

    if len(metrics["validity"]) == 0:
        logger.warning(f"[warn] {ds_name}-{backend}-{method_label}: no successful points.")
        return None

    # aggregate + persist (idempotent)
    out = {f"{k}_mean": float(np.mean(v)) for k, v in metrics.items()}
    out.update({f"{k}_sd": float(np.std(v, ddof=1)) for k, v in metrics.items() if len(v) > 1})
    out.update(dict(dataset=ds_name, backend=backend, method=method_label, n=len(metrics["validity"])))

    pd.DataFrame([out]).to_csv(group_csv, index=False)
    # raw vectors checkpoint
    np.savez_compressed(
        out_dir / "raw_vectors" / f"{ds_name}__{backend}__{method_label}__{len(metrics['validity'])}.npz",
        **{k: np.asarray(v, dtype=float) for k, v in metrics.items()}
    )
    return out

# =========================
# Flow
# =========================
@flow(name="Evaluate CF metrics from Optuna-selected lambdas")
def evaluate_selected_flow(
    paths: Paths = Paths(),
    ecfg: EvalConfig = EvalConfig(),
    merged_file_name: str = "optuna_mo/selected_lambdas/selected_lambdas.csv",
    backends: List[str] = ("sklearn", "PYT", "TF2"),
):
    logger = get_run_logger()
    rng = np.random.default_rng(ecfg.random_seed)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    # resources
    datasets = load_datasets()
    dice_base = load_dice_baseline_models(DefaultPaths().dice_baseline_pickle)
    dice_ext = load_dice_x_models(DefaultPaths().dice_x_pickle)

    # iterate all (ds, backend, method) that have SELECTED.csv
    merged_csv = (DefaultPaths().out_dir / merged_file_name).resolve()
    sel_df = read_selected_table(merged_csv=merged_csv)
    if sel_df.empty:
        logger.info("[SKIP] merged selected_lambdas.csv is empty")

    for col in ("lambda1", "lambda2", "lambda3"):
        if col in sel_df.columns:
            sel_df[col] = pd.to_numeric(sel_df[col], errors="coerce")

    if {"method", "lambda3"}.issubset(sel_df.columns):
        sel_df.loc[sel_df["method"] == "DiCE", "lambda3"] = 0.0

    sel_df = sel_df[sel_df["backend"].isin(backends)].copy()
    sel_df = sel_df.dropna(subset=["lambda1","lambda2","lambda3"]).copy()
    
    ds_map = {name: (df, target) for (df, target, name) in datasets}

    futs = []
    for _, row in sel_df.iterrows():
        ds_name, backend, method = row["dataset"], row["backend"], row["method"]
        lambdas = dict(
            lambda1=row["lambda1"],
            lambda2=row["lambda2"],
            lambda3=row["lambda3"]
        )

        out_csv = paths.out_dir / f"{ds_name}__{backend}__{method}.csv"
        if out_csv.exists():
            logger.info(f"[SKIP] {out_csv} already present.")
            continue

        if ds_name not in ds_map:
            logger.info(f"[SKIP] dataset {ds_name} is not loaded")
            continue

        df, target = ds_map[ds_name]
        model_dict = dice_ext if method == "DiCE-Ext" else dice_base
        if model_dict is None or ds_name not in model_dict or backend not in model_dict[ds_name]:
            logger.info(f"[SKIP] missing model for {ds_name}/{backend}/{method}")
            continue

        logger.info(f"[λ*] Using lambdas for {ds_name}/{backend}/{method}: "
                    f"λ1={lambdas['lambda1']}, λ2={lambdas['lambda2']}, λ3={lambdas['lambda3']}")
        futs.append(
            evaluate_group_with_selected.submit(
                df=df, target=target, ds_name=ds_name, backend=backend, method_label=method,
                dice_models=dice_base, dice_x_models=dice_ext,
                lambdas=lambdas, ecfg=ecfg, out_dir=paths.out_dir
            )
        )
        
    _ = [f.result() for f in futs]

    # aggregate summary
    summary_files = [p for p in paths.out_dir.glob("*.csv") if p.name != "summary_metrics_all.csv" and p.name != "selected_lambdas.csv"]
    if summary_files:
        result_df = pd.concat([pd.read_csv(p) for p in summary_files], ignore_index=True)
        result_df.to_csv(paths.out_dir / "summary_metrics_all.csv", index=False)
        logger.info(f"[done] Wrote summary_metrics_all.csv with {len(result_df)} rows.")

# =========================
# CLI entry
# =========================
if __name__ == "__main__":
    # optional: make Prefect ephemeral (no server), comment if you prefer your profile
    # import os, tempfile, uuid
    # os.environ.setdefault("PREFECT_PROFILE", "default")
    # os.environ["PREFECT_API_URL"] = ""
    # os.environ["PREFECT_API_DATABASE_CONNECTION_URL"] = \
    #     f"sqlite+aiosqlite:///{tempfile.gettempdir()}/prefect-{uuid.uuid4().hex}.db"

    evaluate_selected_flow()
