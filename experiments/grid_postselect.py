"""
Post-selection for DiCE-X lambda grid
====================================

Prefect 2.x flow that reads aggregated CSV results, computes Pareto sets,
applies a one-standard-error rule on robustness, runs a simple knee detector
(Kneedle-style distance-to-chord) on robustness vs. λ3, and writes the chosen
(λ1, λ2, λ3) per dataset × backend (DiCE-Ext only).

Inputs
------
- A directory containing one or more aggregate CSVs produced by the grid search,
  e.g. `coarse__agg_*.csv`, `fine__agg_*.csv`, `hi__agg_*.csv`.

Outputs
-------
- `postselect/choices.csv` – final chosen λ per dataset×backend
- `postselect/debug/*.csv` – Pareto masks, knee curves (optional)

Run
---
    python prefect_dice_postselect.py 
    # or import and call postselect_flow(...)

Notes
-----
- Only DiCE-Ext (method=="DiCE-Ext") rows are considered.
- If a group has too few distinct λ3 values for knee detection (< 3), we skip knee and fall back to 1-SE / composite.
- Validity is assumed to be a percentage in [0,100]; if max≤1, the code scales threshold accordingly.
"""
from __future__ import annotations

import os, tempfile, uuid
os.environ.setdefault("PREFECT_PROFILE", "default")
os.environ.setdefault("PREFECT_API_URL", "")
os.environ.setdefault(
    "PREFECT_API_DATABASE_CONNECTION_URL",
    f"sqlite+aiosqlite:///{tempfile.gettempdir()}/prefect-postsel-{uuid.uuid4().hex}.db"
)

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

# -----------------------------
# Config
# -----------------------------

@dataclass
class PostSelectConfig:
    in_dir: Path = Path("lambda_grid_results")
    out_dir: Path = Path("lambda_grid_results/postselect")
    # constraints
    min_validity_percent: float = 95.0     # keep rows with validity_mean ≥ this (auto-scales if values in [0,1])
    proximity_within_frac: float = 0.05    # keep rows with proximity within 5% of the group's best
    use_pareto: bool = True                # enforce Pareto filter before rules
    # knee detection
    knee_source: str = "constrained"       # one of {"all","pareto","constrained"}
    knee_agg: str = "max"                  # how to aggregate robustness per λ3: "max" or "mean"
    knee_tol: float = 1e-6                 # tolerance when matching λ3 ≈ knee
    # selection
    tie_break_order: Tuple[str,...] = (
        "score", "prox_cont_mad_negmean_mean", "div_cont_mad_mean", "sparsity_cont_mean"
    )

# -----------------------------
# Utils
# -----------------------------

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


def _scale_validity_threshold(v: float, max_validity: float) -> float:
    # If validity in [0,1], convert 95% to 0.95 threshold
    return v/100.0 if max_validity <= 1.5 else v


def _pareto_mask(df: pd.DataFrame, maximize_cols: List[str]) -> np.ndarray:
    """Return boolean mask of non-dominated rows (True = on Pareto front)."""
    X = df[maximize_cols].to_numpy()
    n = X.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # dominated if any j has X[j] >= X[i] for all and > for at least one
        dom = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        dom[i] = False
        if np.any(dom):
            mask[i] = False
    return mask


def _distance_to_chord(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return perpendicular distance from each (x,y) to the line through endpoints.
    Assumes xs sorted ascending, ys aligned.
    """
    x0, y0 = xs[0], ys[0]
    x1, y1 = xs[-1], ys[-1]
    dx, dy = (x1 - x0), (y1 - y0)
    denom = math.hypot(dx, dy)
    if denom == 0:
        return np.zeros_like(xs)
    # distance formula: |dy*x - dx*y + (x1*y0 - y1*x0)| / sqrt(dx^2+dy^2)
    return np.abs(dy*xs - dx*ys + (x1*y0 - y1*x0)) / denom


def _knee_lambda3(df_group: pd.DataFrame, cfg: PostSelectConfig) -> Optional[float]:
    """Compute knee on robustness vs λ3. Aggregate over rows at same λ3 using cfg.knee_agg.
    Returns λ3 at max distance to chord, or None if <3 distinct λ3.
    """
    g = df_group.copy()
    if g.empty:
        return None
    # Aggregate robustness per λ3
    agg_fn = np.max if cfg.knee_agg == "max" else np.mean
    stats = (g.groupby("lambda3", as_index=False)["robustness_keep_rate_mean"].agg(agg_fn)
               .sort_values("lambda3"))
    if len(stats) < 3:
        return None
    xs = stats["lambda3"].to_numpy()
    ys = stats["robustness_keep_rate_mean"].to_numpy()
    # normalize to [0,1] for scale invariance
    xs_n = (xs - xs.min())/(xs.max()-xs.min()) if xs.max() > xs.min() else xs*0
    ys_n = (ys - ys.min())/(ys.max()-ys.min()) if ys.max() > ys.min() else ys*0
    d = _distance_to_chord(xs_n, ys_n)
    knee_idx = int(np.argmax(d))
    return float(xs[knee_idx])


# -----------------------------
# Tasks
# -----------------------------

@task
def find_aggregate_csvs(in_dir: Path) -> List[Path]:
    pats = ["coarse__agg_*.csv", "fine__agg_*.csv", "hi__agg_*.csv", "*__agg_*.csv"]
    files: List[Path] = []
    for pat in pats:
        files.extend(sorted(in_dir.glob(pat)))
    # de-duplicate while preserving order
    seen = set(); out = []
    for p in files:
        if p not in seen:
            out.append(p); seen.add(p)
    return out


@task
def load_and_combine(paths: List[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # keep DiCE-Ext only
    if "method" in df.columns:
        df = df[df["method"] == "DiCE-Ext"].copy()
    return df


@task
def filter_constraints(df: pd.DataFrame, cfg: PostSelectConfig) -> pd.DataFrame:
    if df.empty:
        return df
    # scale validity threshold
    max_val = float(df["validity_mean"].max()) if "validity_mean" in df.columns else 100.0
    vthr = _scale_validity_threshold(cfg.min_validity_percent, max_val)
    out = df.copy()
    if "validity_mean" in out.columns:
        out = out[out["validity_mean"] >= vthr]
    # proximity within fraction of best
    if "prox_cont_mad_negmean_mean" in out.columns and len(out):
        best = float(out["prox_cont_mad_negmean_mean"].max())
        # threshold = best - 5% of |best|
        thr = best - cfg.proximity_within_frac * max(1e-8, abs(best))
        out = out[out["prox_cont_mad_negmean_mean"] >= thr]
    return out


@task
def compute_pareto(df: pd.DataFrame, use_pareto: bool) -> pd.DataFrame:
    if df.empty or not use_pareto:
        return df
    cols = [
        "validity_mean",
        "prox_cont_mad_negmean_mean",
        "sparsity_cont_mean",
        "div_cont_mad_mean",
        "robustness_keep_rate_mean",
    ]
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return df
    mask = _pareto_mask(df[cols], cols)
    return df.loc[mask].copy()


@task
def choose_lambdas_for_group(df_group: pd.DataFrame, cfg: PostSelectConfig) -> Optional[Dict[str, float]]:
    """Apply 1-SE rule (robustness) and knee λ3, return a chosen row with metadata."""
    logger = get_run_logger()
    if df_group.empty:
        return None

    g = df_group.copy()
    g["score"] = composite_score(g)

    # 1-SE rule on robustness: pick minimal λ3 s.t. robustness ≥ best_mean - best_std
    best_idx = int(g["robustness_keep_rate_mean"].idxmax())
    best_mean = float(g.loc[best_idx, "robustness_keep_rate_mean"])
    best_sd = float(g.loc[best_idx, "robustness_keep_rate_std"]) if "robustness_keep_rate_std" in g.columns else 0.0
    one_se_thr = best_mean - best_sd
    one_se_cands = g[g["robustness_keep_rate_mean"] >= one_se_thr]
    one_se_pick = None
    if len(one_se_cands):
        one_se_pick = (one_se_cands.sort_values(["lambda3","score"], ascending=[True,False]).iloc[0]).to_dict()

    # Knee on robustness vs λ3
    # pick source
    if cfg.knee_source == "all":
        src = g
    elif cfg.knee_source == "pareto":
        src = compute_pareto.fn(g, True)
    else:  # constrained (already filtered)
        src = g
    knee_l3 = _knee_lambda3(src, cfg)
    knee_pick = None
    if knee_l3 is not None and np.isfinite(knee_l3):
        near = g.iloc[(g["lambda3"] - knee_l3).abs().argsort()[:5]].copy()
        if len(near):
            knee_pick = (near.sort_values(["lambda3","score"], ascending=[True,False]).iloc[0]).to_dict()

    # Final choice: prefer one-SE; if also knee matches within tol, label as both
    final = None
    how = None
    if one_se_pick is not None and knee_pick is not None:
        if abs(one_se_pick["lambda3"] - knee_pick["lambda3"]) <= cfg.knee_tol:
            final, how = one_se_pick, "oneSE+knee"
        else:
            final, how = one_se_pick, "oneSE"
    elif one_se_pick is not None:
        final, how = one_se_pick, "oneSE"
    elif knee_pick is not None:
        final, how = knee_pick, "knee"
    else:
        # fallback to composite best
        final = g.sort_values("score", ascending=False).iloc[0].to_dict()
        how = "composite"

    final["selected_by"] = how
    return {k: final[k] for k in final.keys() if k in (
        "dataset","backend","method","lambda1","lambda2","lambda3","selected_by",
        "validity_mean","prox_cont_mad_negmean_mean","sparsity_cont_mean",
        "div_cont_mad_mean","robustness_keep_rate_mean","score"
    )}


@task
def write_choices(rows: List[Dict[str, float]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = out_dir / "choices.csv"
    df.to_csv(path, index=False)
    return path


# -----------------------------
# Flow
# -----------------------------

@flow(name="DiCE-X Post-Selection", task_runner=SequentialTaskRunner())
def postselect_flow(cfg: PostSelectConfig = PostSelectConfig()):
    logger = get_run_logger()

    paths = find_aggregate_csvs(cfg.in_dir)
    logger.info(f"Found {len(paths)} aggregate CSV(s)")

    df = load_and_combine(paths)
    if df.empty:
        logger.warning("No aggregate data found.")
        return None

    # group by dataset x backend (DiCE-Ext only)
    rows = []
    for (ds, be), g0 in df.groupby(["dataset","backend"]):
        logger.info(f"Selecting for {ds} / {be}")
        # constraints + (optional) Pareto
        g1 = filter_constraints(g0, cfg)
        g2 = compute_pareto(g1, cfg.use_pareto)
        pick = choose_lambdas_for_group(g2, cfg)
        if pick is not None:
            rows.append(pick)

    out_path = write_choices(rows, cfg.out_dir)
    logger.info(f"Wrote selections → {out_path}")
    return out_path


# -----------------------------
# Script entrypoint
# -----------------------------
if __name__ == "__main__":
    postselect_flow()  # uses defaults; point in_dir/out_dir via PostSelectConfig if needed
