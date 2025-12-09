import sys

dice_path = "/Users/volk/Documents/bau24-25/thesis/repos/DiCE"
dice_x_path = "/Users/volk/Documents/bau24-25/thesis/repos/DiCE-X"
for p in (dice_x_path, dice_path):
    if p not in sys.path:
        sys.path.insert(0, p)

from typing import Optional

import dice_ml
from dice_ml.utils import helpers as dhelpers, neuralnetworks as dnn
import dice_ml_x
from dice_ml_x.utils import helpers as xhelpers, neuralnetworks as xnn

import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier



def compute_validity(explainer) -> float:
    return explainer.get_validity_percentage()

def compute_mad(data_class, normalized=False) -> dict:
    return data_class.get_valid_mads(normalized=normalized)

def compute_continuous_proximity(C: pd.DataFrame, x: pd.DataFrame, data_class: dice_ml.Data) -> float:
    cont_feats = data_class.continuous_feature_names
    if len(cont_feats) == 0:
        return 0.0
    mads = compute_mad(data_class=data_class)
    diffs = np.abs(C[cont_feats].values - x[cont_feats].iloc[0].values)
    # TODO: check the output
    denom = np.array([mads[f] for f in cont_feats])
    norm = diffs / denom
    mean_per_cf = np.nanmean(norm, axis=1)
    return -float(np.mean(mean_per_cf))

def compute_categorical_proximity(C: pd.DataFrame, x: pd.DataFrame, data_class: dice_ml.Data) -> float:
    cats = data_class.categorical_feature_names
    if len(cats) == 0:
        return 0.0
    diff = (C[cats] != x[cats].iloc[0]).sum(axis=1)
    avg = diff / (len(cats) * len(C))
    return 1.0 - float(avg)

def compute_continuous_diversity(self, C: pd.DataFrame) -> float:
    feats = self.data_class.continuous_feature_names
    if len(feats) == 0 or len(C) < 2:
        return 0.0
    mads = self.compute_mad()
    X = C[feats].values
    diff = np.abs(X[:, None, :] - X[None, :, :])
    denom = np.array([mads[f] for f in feats]).reshape(1, 1, -1)
    norm = diff / denom
    pair = np.nanmean(norm, axis=2)
    iu = np.triu_indices(len(C), 1)
    return float(np.mean(pair[iu])) if len(iu[0]) else 0.0

def compute_categorical_diversity(self, C: pd.DataFrame) -> float:
    feats = self.data_class.categorical_feature_names
    if len(feats) == 0 or len(C) < 2:
        return 0.0
    X = C[feats].values
    diff = (X[:, None, :] != X[None, :, :]).astype(np.float32)
    pair = diff.mean(axis=2)
    iu = np.triu_indices(len(C), 1)
    return float(np.mean(pair[iu])) if len(iu[0]) else 0.0

def compute_sparsity(self, C: pd.DataFrame, x: pd.DataFrame) -> float:
    feats = self.data_class.continuous_feature_names
    if len(feats) == 0:
        return 0.0
    dif = (C[feats].to_numpy() != x[feats].to_numpy()[0])
    changed = dif.sum()
    k, d = C[feats].shape
    return 1.0 - (changed / (k * d))

def robustness_flip_rate(
    self,
    C: pd.DataFrame,
    target_col: str,
    data_iface: "dice_ml.Data",
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
    cat_cols = data_iface.categorical_feature_names
    ranges = data_iface.get_features_range_float()[1]
    cat_vals = {c: data_iface.get_features_range()[1][c] for c in cat_cols}

    kept = 0
    n_cf = len(C)
    for _ in range(n_repeat):
        X_noisy = X_raw.copy()
        for col in cont_cols:
            lo, hi = ranges[col]
            span = hi - lo
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
    self,
    x_df: pd.DataFrame,
    cfs_df: pd.DataFrame,
    model,
    backend: str,
    radius_mad: float,
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    rng = rng or np.random.default_rng()
    OUTCOME = self.data_class.outcome_name

    def predict_classes(df_raw: pd.DataFrame) -> np.ndarray:
        X_raw = df_raw.drop(columns=[OUTCOME], errors="ignore")
        if backend == "sklearn":
            out = model.predict(X_raw)
            if out.ndim == 2 or (out.ndim == 1 and np.issubdtype(out.dtype, np.floating)):
                out = (out >= 0.5).astype(int).ravel()
            return out.astype(int)
        enc = self.data_class.get_ohe_min_max_normalized_data(X_raw).values
        if backend == "PYT":
            preds = model(torch.tensor(enc, dtype=torch.float32)).detach().cpu().numpy().ravel()
        else:
            preds = model.predict(tf.constant(enc, dtype=tf.float32)).ravel()
        return (preds >= 0.5).astype(int)

    train_raw = pd.concat([x_df, cfs_df], ignore_index=True).drop(columns=[OUTCOME], errors="ignore")
    y_train = predict_classes(train_raw)

    if backend == "sklearn":
        train_df = pd.get_dummies(train_raw).fillna(train_raw.mean(numeric_only=True))
        cols_ref = train_df.columns
        X_train = train_df.values
    else:
        X_train = self.data_class.get_ohe_min_max_normalized_data(train_raw).values
        cols_ref = None

    knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)

    cont_cols = self.data_class.continuous_feature_names
    cat_cols = self.data_class.categorical_feature_names
    mad_vals = self.data_class.get_valid_mads()
    cat_levels = {c: self.data_class.get_features_range()[1][c] for c in cat_cols}

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
        for c in cols_ref:    # type: ignore
            if c not in X_s.columns:
                X_s[c] = 0
        X_s = X_s[cols_ref].fillna(X_s.mean(numeric_only=True)).values
    else:
        X_s = self.data_class.get_ohe_min_max_normalized_data(synth_raw).values

    orig = predict_classes(synth_raw)
    proxy = knn.predict(X_s)    # type: ignore
    return float(np.mean(orig == proxy))
