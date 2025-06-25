import sys
dice_path = "/Users/volk/Documents/bau24-25/thesis/repos/DiCE-X"
sys.path.insert(0, dice_path)


from prefect import task, flow
from pathlib import Path
import torch
import pandas as pd
import numpy as np

import dice_ml_x
import dice_ml_x.utils.helpers as helpers
from dice_ml_x.benchmarking import Benchmarking
from dice_ml_x.utils.neuralnetworks import PYTModel
from ucimlrepo.fetch import fetch_ucirepo
import os
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Mapping
import json


@task
def load_raw_datasets() -> dict:
    diabetes_path = f"experiments/datasets/diabetes_train.csv"
    if not os.path.exists(diabetes_path):
        diabetes_raw = fetch_ucirepo(id=296)
        diabetes_df: pd.DataFrame = diabetes_raw.data.features 
        y = diabetes_raw.data.targets
        diabetes_df["readmitted"] = y
        diabetes_df["readmitted"] = diabetes_df["readmitted"].map({
            "NO": 0,
            "<30": 1,
            ">30": 1
        })
        diabetes_df.to_csv(diabetes_path)
    else:
        diabetes_df = pd.read_csv(diabetes_path)


    """
    "diabetes-readmission": {
            "data": diabetes_df,
            "target": "readmitted"
        }
    """    


    all_datasets = {
        "adult-income": {
            "data": helpers.load_adult_income_dataset(),
            "target": "income"
        }
        
    }
    return all_datasets

@task
def get_x_test(
    test_df: pd.DataFrame,
    target: str,
    cat_cols: list[str],
    cont_cols: list[str]
) -> pd.DataFrame:
    """
    Take a test DataFrame, drop the target, impute any missing values,
    and return X_test.
    """
    x_test = test_df.drop(columns=[target]).copy()

    # if there's at least one NaN in x_test, impute
    if x_test.isna().values.any():
        from sklearn.impute import SimpleImputer

        num_imp = SimpleImputer(strategy="median")
        cat_imp = SimpleImputer(strategy="most_frequent")

        # numeric
        x_test[cont_cols] = num_imp.fit_transform(x_test[cont_cols])

        # categorical
        x_test[cat_cols] = cat_imp.fit_transform(x_test[cat_cols])

    return x_test
    

@task
def load_and_split(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    return train_df, test_df


@task
def make_dataloaders(df, cont_feats, backend, target, batch_size):
    from dice_ml_x.benchmarking import Benchmarking
    from sklearn.preprocessing import MinMaxScaler
    benchmarking = Benchmarking(None, None)
    scaler = MinMaxScaler()
    train_loader, test_loader, *_ = benchmarking.preprocess_data(
        "PYT", df, cont_feats, target,
        batch_size, pyt_scaler=scaler
    )
    return train_loader, test_loader


@task
def save_history_plots(ds_name: str, history: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    epochs = history['epoch']

    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['test_acc'],  label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    acc_path = os.path.join(output_dir, f'{ds_name}_model_accuracy.png')
    plt.savefig(acc_path)
    plt.close()

    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['test_loss'],  label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    loss_path = os.path.join(output_dir, f'{ds_name}_model_loss.png')
    plt.savefig(loss_path)
    plt.close()


@task
def save_ae_grid_plot(errors: list, elbow: int, latent_sizes: list, out_dir: str):
    """
    Expects to find attributes on itself:
      save_ae_grid_plot.errors
      save_ae_grid_plot.latent_sizes
      save_ae_grid_plot.elbow
    """
    
    plt.plot(latent_sizes, errors, marker='o')
    if elbow in latent_sizes:
        idx = latent_sizes.index(elbow)
        plt.scatter([elbow], [errors[idx]], s=100, facecolors='none', edgecolors='red', linewidths=2, label='Elbow')

    plt.xlabel("Latent dimension")
    plt.ylabel("Validation MSE")
    plt.title("AE latent dim grid search")
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    ae_grid_plot_outdir = f"{out_dir}/ae_grid_plot.png"
    plt.savefig(ae_grid_plot_outdir)
    plt.close()


@task
def train_dp_autoencoder(train_loader, input_dim, latent_dim, epsilon=1.0, epochs=10, lr=1e-3):
    from dice_ml_x.autoencoders.dp_s_ae import DPStandardAutoEncoder
    ae = DPStandardAutoEncoder(input_dim, latent_dim)
    ae.train_autoencoder(train_loader, epochs=epochs, batch_size=train_loader.batch_size,
                         epsilon=epsilon, learning_rate=lr)
    return ae

@task
def select_best_autoencoder(ae_list, errors, latent_sizes, outdir):
    from kneed import KneeLocator
    kl = KneeLocator(latent_sizes, errors, curve='convex', direction='decreasing')
    idx = latent_sizes.index(kl.knee)
    save_ae_grid_plot(errors, kl.knee, latent_sizes, outdir)
    return ae_list[idx]

@task
def train_classifier(train_loader, test_loader, model_dir) -> tuple[PYTModel, torch.nn.Sequential]:
    benchmarking = Benchmarking(None, None)
    trainer, model = benchmarking.train_pytorch_model(
        train_loader, test_loader, model_dir
    )
    return trainer, model

@task
def generate_cfs(clf, ae, train_df, x_test, target, cont_feats,
                 cat_feats, backend, number_of_cfs, ds_name, cfs_outdir):
    gaussian_kwargs = {
        'continuous_features': cont_feats,
        'categorical_features': cat_feats,
        'std_dev': 0.3
    }
    d = dice_ml_x.Data(dataframe=train_df, continuous_features=cont_feats, 
                    categorical_features=cat_feats, outcome_name=target)

    m = dice_ml_x.Model(model=clf, backend=backend,  func="ohe-min-max")
    dice_x_opts = OrderedDict(data_interface=d, model_interface=m, method="gradient")
    if backend == "DP_PYT":
        dice_x_opts["dp_autoencoder"] = ae
    
    exp = dice_ml_x.DiceX(**dice_x_opts)
    input_instance = x_test[1:2]
    learning_rate = 0.03 if backend == "DP_PYT" and ds_name == "adult-income" else 0.05
    exp_options = OrderedDict(query_instances=input_instance, total_CFs=number_of_cfs, max_iter=1000, desired_class="opposite",
                                            perturbation_method="gaussian", algorithm="DiverseCF",
                                            learning_rate=learning_rate, **gaussian_kwargs)
    
    dice_exp = exp.generate_counterfactuals(**exp_options)
    save_cfs(dice_exp.to_dataframe(), cfs_outdir)
    return {
        "explainer": exp,
        "counterfactuals": dice_exp.to_dataframe(),
        "input_instance": input_instance,
        "data_class": d,
        "model_class": m
    } 

@task
def plot_explainer_loss_histories(
    histories: Mapping[str, Mapping[str, Mapping[str, list]]],
    output_path: str
) -> str:
    """
    Plot and save explainer loss histories, but only up to the 50th iteration.

    Args:
        histories: nested dict indexed by
            histories[dataset_name][backend_name] = {
                "iterations": [...],
                "y_loss": [...],
                "proximity_loss": [...],
                "diversity_loss": [...],
                "robustness_loss": [...],
                # "regularization_loss": [...]  # ignored
                "total_loss": [...]
            }
        output_path: directory in which to write the .png.

    Returns:
        The path to the saved figure.
    """
    datasets = list(histories.keys())
    backends = list(next(iter(histories.values())).keys())

    # which curves to draw
    loss_terms = [
        ("Proximity Loss",  "proximity_loss",  "o-"),
        ("Diversity Loss",  "diversity_loss",  "o-"),
        ("Y Loss",          "y_loss",          "^-"),
        ("Robustness Loss", "robustness_loss", "s-"),
    ]

    n_rows, n_cols = len(datasets), len(backends)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        constrained_layout=True
    )

    # normalize axes to 2D list
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, ds in enumerate(datasets):
        for j, bk in enumerate(backends):
            ax = axes[i][j]
            hist = histories[ds][bk]

            # only first 50 points
            iters = hist["iterations"][:50]

            for label, key, style in loss_terms:
                y = hist[key][:50]
                ax.plot(iters, y, style, label=label)

            ax.set_title(f"{bk}")
            ax.set_xlabel("Index (Iterations)")
            if j == 0:
                ax.set_ylabel(ds)
            # only show one legend
            if i == 0 and j == n_cols - 1:
                ax.legend(loc="upper right")

    # write to disk
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "explainer_plots.png")
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


@task
def save_cfs(cfs: pd.DataFrame, outdir: str):

    os.makedirs(outdir, exist_ok=True)
    cfs.to_csv(os.path.join(outdir, "counterfactuals.csv"), index=False)


@task
def compute_all_metrics(
    C_df: pd.DataFrame,
    x_df: pd.DataFrame,
    train_df: pd.DataFrame,
    data_class,
    predict_fn,
    C_prime_df: pd.DataFrame = None,
    noise_std: float = 1e-2,
    nn_k: int = 5,
) -> dict:
    """
    Compute proximity, sparsity, diversity, stability (optional),
    plausibility (via k-NN in train set), and robustness for a set of CFs.
    """
    outcome_col = data_class.outcome_name
    if outcome_col in train_df.columns:
        train_df = train_df.drop(columns=[outcome_col])
    if outcome_col in C_df.columns:
        C_df = C_df.drop(columns=[outcome_col])
    if outcome_col in x_df.columns:
        x_df = x_df.drop(columns=[outcome_col])

    x_ohe     = data_class.get_ohe_min_max_normalized_data(x_df)
    C_ohe     = data_class.get_ohe_min_max_normalized_data(C_df)
    train_ohe = data_class.get_ohe_min_max_normalized_data(train_df)

    x_t     = torch.from_numpy(x_ohe.values.astype(np.float32))      # (1, D)
    C_t     = torch.from_numpy(C_ohe.values.astype(np.float32))      # (k, D)
    train_t = torch.from_numpy(train_ohe.values.astype(np.float32))  # (n_train, D)

    k, D = C_t.shape

    proximity = torch.mean(torch.cdist(x_t, C_t, p=1)).item()

    cont = data_class.continuous_feature_names
    cont_C = C_df[cont].to_numpy()
    cont_x = x_df[cont].to_numpy()[0]
    diff   = (cont_C != cont_x[None, :])
    sparsity = 1 - diff.sum()/(k * len(cont))

    pdist = torch.cdist(C_t, C_t, p=2)
    diversity = ((pdist.sum() - torch.trace(pdist)) / (k*(k-1))).item()

    stability = None
    if C_prime_df is not None:
        C2_ohe   = data_class.get_ohe_min_max_normalized_data(C_prime_df.drop(columns=[outcome_col]))
        C2_t     = torch.from_numpy(C2_ohe.values.astype(np.float32))
        stability = torch.mean(torch.cdist(C_t, C2_t, p=2)).item()

    plausibility = None
    if train_t.numel() > 0:
        dists = torch.cdist(C_t, train_t, p=2)
        knn   = torch.topk(dists, k=nn_k, dim=1, largest=False).values
        plausibility = knn.mean().item()

    with torch.no_grad():
        # predict_fn returns a numpy array, convert to a tensor
        y0 = torch.tensor(predict_fn(C_t), dtype=torch.float32)

        noise = torch.randn_like(C_t) * noise_std

        # predict_fn returns a numpy array, convert to a tensor
        y1 = torch.tensor(predict_fn(C_t + noise), dtype=torch.float32)

        robustness = torch.mean(torch.abs(y1 - y0)).item()

    result = {
        "proximity":  round(proximity,  4),
        "sparsity":   round(sparsity,   4),
        "diversity":  round(diversity,  4),
        "robustness": round(robustness, 4),
    }
    if stability    is not None: result["stability"]    = round(stability,    4)
    if plausibility is not None: result["plausibility"] = round(plausibility, 4)

    return result


@task
def save_metrics_dict(metrics: dict, output_path: str) -> str:
    """
    Save the nested metrics dictionary to a JSON file.

    Args:
        metrics: The dictionary of metrics, e.g. {backend: {dataset: {metric: value, …}, …}, …}
        output_path: Full path to write the JSON file to.

    Returns:
        The path to the written file.
    """
    # ensure folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # write out
    with open(f"{output_path}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    return output_path

# -- 2. Assemble Flow --

@flow(name="Dice_X_Benchmark")
def dice_x_benchmark_flow():
    datasets = load_raw_datasets()
    backends = ["PYT", "DP_PYT"]
    histories = {backend: {} for backend in backends}
    metrics_dict = {backend: {} for backend in backends}
    for backend in backends:
        for ds_name, ds_info in datasets.items():
            df = ds_info["data"]
            target = ds_info["target"]
            train_df, test_df = load_and_split(df, target)
            cont_cols = df.select_dtypes(include=[np.number]).columns.difference([target]).tolist()
            cat_cols = df.columns.difference(cont_cols + [target])
            x_test = get_x_test(test_df, target, cat_cols, cont_cols)
            train_loader, test_loader = make_dataloaders(df, cont_cols, backend, target, batch_size=64)
            input_dim = next(iter(train_loader))[0].shape[1]
            latent_dims = [5, 10, 15, 20, 30]
            ae_list = []
            errors = []
            for latent_dim in latent_dims:
                dp_ae = train_dp_autoencoder(train_loader, input_dim, latent_dim)
                errors.append(dp_ae.history['loss'][-1])
                ae_list.append(dp_ae)

            best_ae = select_best_autoencoder(ae_list, errors, latent_dims,
                                              f"experiments/charts/{backend}/{ds_name}")
            trainer, dice_model = train_classifier(train_loader, test_loader, f"experiments/{backend}/{ds_name}")
            trainer.save_model(f"experiments/model_artefacts/{backend}/{ds_name}")
            save_history_plots(ds_name, trainer.history, f"experiments/charts/{backend}/{ds_name}")
            generation_result = generate_cfs(dice_model, best_ae, df if ds_name == "diabetes-readmission" else train_df, x_test,
                         target, cont_cols, cat_cols, backend, 10, ds_name, f"experiments/cfs/{backend}/{ds_name}")
            explainer = generation_result["explainer"]
            cfs = generation_result["counterfactuals"]
            input_instance = generation_result["input_instance"]
            data_interface = generation_result["data_class"]
            model_interface = generation_result["model_class"]

            histories[backend][ds_name] = explainer.loss_history

            metrics_dict[backend][ds_name] = compute_all_metrics(cfs, test_df.iloc[[1]], train_df, data_interface,
                                explainer.predict_fn, cfs.iloc[[1]])
            
            metrics_dict[backend][ds_name]["validity"] = explainer.get_validity_percentage()

    save_metrics_dict(metrics_dict, "experiments/metrics")
    plot_explainer_loss_histories(histories, "experiments/plots")
        

# -- 3. Running & Debugging --
if __name__ == "__main__":
    dice_x_benchmark_flow()
