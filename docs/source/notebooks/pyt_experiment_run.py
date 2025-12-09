import sys
dice_path = "/Users/volk/Documents/bau24-25/thesis/repos/DiCE-X"
sys.path.insert(0, dice_path)

import dice_ml_x
from dice_ml_x.utils import helpers, neuralnetworks
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import itertools
from tqdm import tqdm
import pandas as pd
import random
import tensorflow as tf
import os
from collections import OrderedDict


with open('benchmarking_results_23_01_2025-01_05.pkl', 'rb') as res_file:
    benchmarking_results = pickle.load(res_file)

backends = ['sklearn', 'PYT', 'TF2']
def load_torch_model(model_path, in_features):
    dummy_state_dict = torch.load(model_path)
    dummy_state_dict = {f'model.{key}': value for key, value in dummy_state_dict.items()}
    model = neuralnetworks.PYTModel(in_features, model_save_dir="")
    model.load_state_dict(dummy_state_dict)
    return model


def load_tensorflow_model(model_path: str):
    model = neuralnetworks.TF2Model()
    model.load_weights(str(model_path))
    return model

dataset_names = [
    "compas-recidivism",
    "adult-income",
    "lending-club",
    "german-credit"
]
models = {}
for name in dataset_names:
    models[name] = {}
    for backend in backends:
        if backend == 'sklearn':
            models[name][backend] = benchmarking_results[name][backend]['model']
        elif backend == 'PYT':
            model_path = benchmarking_results[name][backend]['model_path']
            num_features = benchmarking_results[name][backend]['metrics']['num_features']
            models[name][backend] = load_torch_model(model_path, num_features)
        elif backend == 'TF2':
            model_path = benchmarking_results[name][backend]['model_path']
            models[name][backend] = load_tensorflow_model(model_path)


datasets = [(helpers.load_compas_dataset(), "twoyearrecid", "compas-recidivism"),
            (helpers.load_adult_income_dataset(), "income", "adult-income"),
             (helpers.load_lending_club_dataset(), "loan_status", "lending-club"),
             (helpers.load_german_credit_dataset(), "credit_risk", "german-credit")]
backends = ['TF2']
root_folder = 'cfe_datasets_28_06_25'
os.environ["TQDM_DISABLE"] = "1"


cfe_datasets = {}

pbar = tqdm(datasets, desc="Generating counterfactual datasets...")
for exp_iteration in range(2):
    root_folder = f"{root_folder}_0{exp_iteration + 1}"
    os.makedirs(root_folder, exist_ok=True)
    for df, target_name, df_name in pbar:
        target_col = df[target_name]
        train_dataset, test_dataset, y_train, y_test = train_test_split(df, target_col, test_size=0.2,
                                                                            random_state=42, stratify=target_col)
        cont_feats = df.select_dtypes(include=[np.number]).columns.difference([target_name])
        d = dice_ml_x.Data(dataframe=train_dataset, continuous_features=list(cont_feats), outcome_name=target_name)
        x_test = test_dataset.drop(columns=[target_name])
        

        for backend in backends:
            file_path = os.path.join(root_folder, f"{df_name}_{backend}_cfe.csv")

            if os.path.isfile(file_path):
                existing_df = pd.read_csv(file_path)
                num_rows = len(existing_df)
                if num_rows >= 1000:
                    print(f"Skipping {df_name} for the backend {backend} because it already exists and have {num_rows} rows.")
                    continue
            else:
                existing_df = pd.DataFrame()
                
            print(f"Processing {df_name} with {backend} ...")

            model_options = OrderedDict(model=models[df_name][backend], backend=backend)
            method = 'genetic' if backend == 'sklearn' else 'gradient'

            if backend != 'sklearn':
                model_options['func'] = 'ohe-min-max'
                if backend == 'PYT':
                    model_options['model'] = models[df_name][backend].model

            m = dice_ml_x.Model(**model_options)
            exp = dice_ml_x.Dice(d, m, method=method)

            if backend == 'PYT':
                x_test_ohe_vector = torch.tensor(d.get_ohe_min_max_normalized_data(x_test).values, dtype=torch.float32)
            elif backend == 'TF2':
                x_test_ohe_vector = tf.constant(d.get_ohe_min_max_normalized_data(x_test).values, dtype=tf.float32)
            
            if backend == 'PYT':
                _0_d_indices = np.where(m.model(x_test_ohe_vector) < 0.5)[0].tolist()
            elif backend == 'TF2':
                _0_d_indices = np.where(m.model.predict(x_test_ohe_vector) < 0.5)[0].tolist()
            else:
                _0_d_indices = np.where(m.model.predict(x_test) < 0.5)[0].tolist()

            _1_d_indices = list((set(range(len(x_test)))) - set(list(_0_d_indices)))

            random.shuffle(_0_d_indices)
            random.shuffle(_1_d_indices)

            if not existing_df.empty:
                count_class_0 = len(existing_df.loc[existing_df[target_name] < 0.5])
                count_class_1 = len(existing_df.loc[existing_df[target_name] >= 0.5])
            else:
                count_class_0, count_class_1 = 0, 0

            target_count = 500
            
            cfe_dataset_list = []
    
            while count_class_0 < target_count or count_class_1 < target_count:
                try:
                    if count_class_0 < target_count:
                        rand_idx = random.choice(_0_d_indices)
                    else:
                        rand_idx = random.choice(_1_d_indices)

                    x = x_test[rand_idx : rand_idx + 1]

                    if backend == 'PYT':
                        x_ohe = torch.tensor(d.get_ohe_min_max_normalized_data(x).values, dtype=torch.float32)
                        desired_class = int(m.model(x_ohe) >= 0.5)
                    elif backend == 'TF2':
                        x_ohe = tf.constant(d.get_ohe_min_max_normalized_data(x).values, dtype=tf.float32)
                        desired_class = int(m.model.predict(x_ohe) >= 0.5)
                    else:
                        desired_class = int(m.model.predict(x) >= 0.5)

                    desired_class = 1 - desired_class

                    if (desired_class == 0 and count_class_0 >= target_count) or \
                    (desired_class == 1 and count_class_1 >= target_count):
                        continue

                
                    dice_exp = exp.generate_counterfactuals(
                        x, total_CFs=1, desired_class=desired_class, robustness_weight=0.4
                    )

                    cfe_sample = dice_exp.to_dataframe()

                    if desired_class == 0:
                        count_class_0 += 1
                    else:
                        count_class_1 += 1
                    
                    
                    cfe_dataset_list.append(cfe_sample)
                    
                    if len(cfe_dataset_list) % 2 == 0:
                        
                        combined_df = pd.concat(cfe_dataset_list, ignore_index=True).drop_duplicates()
                        
                        if os.path.isfile(file_path):
                            existing_df = pd.read_csv(file_path)
                            combined_df = pd.concat([combined_df, existing_df], ignore_index=True).drop_duplicates()
                        
                        count_class_0 = len(combined_df[target_name] < 0.5)
                        count_class_1 = len(combined_df[target_name] >= 0.5)

                        combined_df.to_csv(file_path, index=False)
                        
                        print(f"{df_name} - {backend}, {len(combined_df)} counterfactuals are saved to {file_path}")
                        cfe_dataset_list = []

                    pbar.set_postfix(OrderedDict(backend=backend, dataset=df_name,
                                                sample_count_class_0=count_class_0,
                                                sample_count_class_1=count_class_1))
                    pbar.update(1)
                except Exception as e:
                    print(f"Couldn't generate counterfactuals for {df_name}, {backend}")
                    continue