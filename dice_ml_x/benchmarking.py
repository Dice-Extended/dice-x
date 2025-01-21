import os
import dice_ml_x
from dice_ml_x.utils import helpers, neuralnetworks
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from time import time
import torch

from collections import OrderedDict
from typing import List


import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

class Benchmarking:
    def __init__(self, datasets: List[tuple], backends: list, metrics=None):
        self.datasets = datasets
        self.backends = backends
        self.metrics = metrics if metrics else ["fidelity", "proximity", "diversity", "robustness"]
        self.models = {}
        self.results = {}
        self.sklearn_pipeline = None
        
    def split_data(self, df, target_col, test_size=0.2, random_state=42):
        target = df[target_col]
        train_dataset, test_dataset, y_train, y_test = train_test_split(df,
                                                                target,
                                                                test_size=test_size,
                                                                random_state=random_state,
                                                                stratify=target)
        return train_dataset, test_dataset, y_train, y_test
      
        
    def preprocess_data(self, backend: str, df: pd.DataFrame, continuous_features: list, target_name: str, batch_size: int):
        train_dataset, test_dataset, y_train, y_test = self.split_data(df, target_name)
        
        x_train = train_dataset.drop(target_name, axis=1)
        x_test = test_dataset.drop(target_name, axis=1)

        if backend == "sklearn":
            categorical = x_train.columns.difference(continuous_features)

            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            transformations = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical)])

            self.sklearn_pipeline = Pipeline(steps=[('preprocessor', transformations),
                                ('classifier', RandomForestClassifier())])
            return x_train, x_test, train_dataset, test_dataset, y_train, y_test
        elif backend == "PYT":
            pyt_train_dataset = neuralnetworks.PYTDataset(df, target_column=target_name, train=True)
            pyt_test_dataset = neuralnetworks.PYTDataset(df, target_column=target_name, train=False)
            train_df = pyt_train_dataset.train_dataset_df
            test_df = pyt_train_dataset.test_dataset_df
            y_train_df = pyt_train_dataset.y_train_df
            y_test_df = pyt_train_dataset.y_test_df
            pyt_train_dataloader = DataLoader(pyt_train_dataset, batch_size=batch_size, shuffle=True)
            pyt_test_dataloader = DataLoader(pyt_test_dataset, batch_size=batch_size // 4, shuffle=False)
            
            return pyt_train_dataloader, pyt_test_dataloader, train_df, test_df, y_train_df, y_test_df
        elif backend == "TF2":
            categorical = x_train.columns.difference(continuous_features)

            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
        
            transformations = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical),
                    ('num', StandardScaler(), continuous_features)
                ],
                sparse_threshold=0
            )

            transformation_pipeline = transformations.fit(x_train)

            x_train_transformed_data = transformation_pipeline.transform(x_train)
            x_test_transformed_data = transformation_pipeline.transform(x_test)

            tf_train_dataset = tf.data.Dataset.from_tensor_slices((x_train_transformed_data, y_train.values))
            tf_test_dataset = tf.data.Dataset.from_tensor_slices((x_test_transformed_data, y_test.values))

            tf_train_dataset = tf_train_dataset.shuffle(len(x_train)).batch(batch_size)
            tf_test_dataset = tf_test_dataset.batch(batch_size=batch_size)

            return tf_train_dataset, tf_test_dataset, train_dataset, test_dataset, y_train, y_test
    
    def train_random_forest(self, x_train, y_train) -> Pipeline:
        self.sklearn_pipeline.fit(x_train, y_train)
        return self.sklearn_pipeline
    
    def train_pytorch_model(self, train_dataloader, test_dataloader):

        dummy_inputs, _ = next(iter(train_dataloader))
        in_features = dummy_inputs.shape[1]
        trainer = neuralnetworks.PYTModel(in_features)
        trainer.train(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
        return trainer.model
    
    def compute_pytorch_accuracy(self, model, test_dataloader, criterion=torch.nn.BCELoss()):
        correct_test_preds = 0.0
        test_loss = 0.0
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        model.eval()
        for _, test_batch in enumerate(test_dataloader):
            test_features, test_labels = test_batch
            test_features, test_labels = test_features.to(device), test_labels.float().to(device).unsqueeze(1)
            with torch.no_grad():
                
                test_outputs = model(test_features)
                loss = criterion(test_outputs, test_labels)
                test_preds = (test_outputs > 0.5).float()
            test_loss += loss.item()
            correct_test_preds += (test_preds == test_labels).sum().item()

        return correct_test_preds / len(test_dataloader.dataset)
    
    def train_keras_model(self, train_dataset, test_dataset, epochs=10):
        model = neuralnetworks.TF2Model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, verbose=0)
        return model


    def compute_keras_accuracy(self, model, tf_test_dataset):
        _, accuracy = model.evaluate(tf_test_dataset, verbose=0)
        return accuracy
    

    def train_model(self, backend, X, x_test, y, epochs=10):
        if backend == 'sklearn':
            return self.train_random_forest(X, y)
        elif backend == 'PYT':
            return self.train_pytorch_model(X, x_test)
        elif backend == 'TF2':
            return self.train_keras_model(X, x_test, epochs)
        
    def compute_stability(C_set_1: torch.Tensor, C_set_2: torch.Tensor, p: int=2) -> float:
        return torch.mean(torch.cdist(C_set_1, C_set_2, p=2)).item()
        
    def compute_proximity(self, original_instance: torch.Tensor, C: torch.Tensor) -> float:
        return torch.mean(torch.cdist(original_instance, C, p=1)).item()

    def compute_diversity(self, C_ohe: torch.Tensor):
        k = C_ohe.shape[0]
        pairwise_dist = torch.cdist(C_ohe, C_ohe, p=2)
        diversity = (torch.sum(pairwise_dist) - torch.sum(torch.diagonal(pairwise_dist))) / (k * (k - 1))
        return diversity.item()
    
    def compute_validity(self, CFs: pd.DataFrame) -> float:
        uniqe_rows, _ = CFs.drop_duplicates().shape
        rows, _ = CFs.shape
        return float(uniqe_rows) / float(rows)
    
    def do_perturbation(self, x: pd.DataFrame, data_class: dice_ml_x.Data):
        x_tensor = torch.tensor(x.values, dtype=torch.float32)
        continuous_feature_indexes = list(set(list(range(len(data_class.get_ohe_min_max_normalized_data(x).columns)))) - set(cat_cols))
        categorical_feature_indexes = data_class.get_encoded_categorical_feature_indexes()
        if continuous_feature_indexes:
            
            continuous_slice = x_tensor[:, continuous_feature_indexes]
            noise = continuous_slice * 0.1
            noise_mask = torch.zeros_like(x_tensor)
            noise_mask[:, continuous_feature_indexes] = noise
            x_tensor = x_tensor + noise_mask

        if categorical_feature_indexes:
            for cat_cols in categorical_feature_indexes:
                cat_slice = x_tensor[:, cat_cols]
                sample_size = cat_slice.shape[0]
                num_cats = cat_slice.shape[1]

                rand_idx = torch.randint(low=0, high=num_cats, size=(sample_size, ))

                cat_slice_perturbed = torch.nn.functional.one_hot(rand_idx, num_classes=num_cats).float()

                cat_mask = torch.zeros_like(x_tensor)
                cat_mask[:, cat_cols] = cat_slice_perturbed
                x_perturbed = x_tensor + cat_mask
        return x_perturbed

    def generate_perturbations(self, x_ohe: pd.DataFrame, data_class: dice_ml_x.Data,
                               model: any, max_iter=100, tol=1e-3, gamma=1e-2):
        x_ohe_tensor = torch.tensor(x_ohe.values, dtype=torch.float32, requires_grad=True)
        x_perturbed = self.do_perturbation(x_ohe, data_class)
        perturbation_optimizer = torch.optim.Adam([x_perturbed], lr=1e-3)

        prev_loss = np.inf
        for _ in range(max_iter):
            with torch.no_grad():
                model.model.eval()
                pred_i = model.model(x_ohe_tensor)
                pred_i_prime = model.model(x_perturbed)
            class_loss = torch.mean((pred_i - pred_i_prime) ** 2)
            distance = torch.norm(x_perturbed - x_ohe_tensor, p=2)
            loss = class_loss + gamma * distance


            perturbation_optimizer.zero_grad()
            loss.backward()

            perturbation_optimizer.step()
            if abs(loss.item() - prev_loss) < tol:
                break
            prev_loss = loss.item()
        return x_perturbed.detach()

    
    def load_and_train(self, batch_size, artefact_path=None):
        num_processes = len(self.datasets) * len(self.backends)
        
        if artefact_path is None:
            artefact_path = "benchmarking_artefact"
        if not os.path.isdir(artefact_path):
            os.mkdir(artefact_path)
        with tqdm(total=num_processes, desc="Benchmarking", leave=True) as d_pbar:
            for df, target_column, dataset_name in self.datasets:

                continuous_features = df.select_dtypes(include=[np.number]).columns.to_list()
                continuous_features.remove(target_column)
                self.results[dataset_name] = {}
                for backend in self.backends:
            
                    x_train_transformed, x_test_transformed, train_df, test_df, y_train, y_test = self.preprocess_data(backend=backend,
                                                                                                                    df=df,
                                                                                                                    continuous_features=continuous_features,
                                                                                                                    target_name=target_column,
                                                                                                                    batch_size=batch_size)
                    
                    model = self.train_model(backend, x_train_transformed, x_test_transformed, y_train)
                    
                    if backend == "sklearn":
                        accuracy = model.score(x_test_transformed, y_test)
                    elif backend == "PYT":
                        accuracy = self.compute_pytorch_accuracy(model, x_test_transformed)
                    elif backend == "TF2":
                        accuracy = self.compute_keras_accuracy(model, x_test_transformed)
                    
                    backend_results = {
                        'accuracy': accuracy,
                        'cfs': {},
                        'input_instance': {},
                        'time': {},
                        'exp_history': {}
                    }

                    if backend == "PYT":
                        model_path = os.path.join(artefact_path, f"{dataset_name}_{backend}_model.pth")
                        torch.save(model.state_dict(), model_path)
                        backend_results['model_path'] = model_path
                    elif backend == "TF2":
                        model_path = os.path.join(artefact_path, f"{dataset_name}_{backend}_model")
                        model.save_weights(model_path, save_format='tf')    
                        backend_results['model_path'] = model_path
                    else:  # sklearn
                        backend_results['model'] = model

                    
                    
                    print(f"the dataset is : {dataset_name}, the backend is : {backend}")
                        
                    cfs, input_instance, generation_time, exp_loss_history = self.generate_cfs(df,
                                                                                continuous_features,
                                                                                model,
                                                                                backend,
                                                                                target_column)
                    backend_results['cfs'] = cfs
                    backend_results['input_instance'] = input_instance
                    backend_results['time'] = generation_time
                    backend_results['exp_history'] = exp_loss_history
                    self.results[dataset_name][backend] = backend_results
                    d_pbar.set_postfix(OrderedDict(
                            dataset=dataset_name,
                            backend=backend
                    ))
                    d_pbar.update(1)

    
    def generate_cfs(self, dataset: pd.DataFrame,
                     continuous_features: list,
                     model: any, model_backend: str,
                     target_name: str, total_CFS=10, proximity_weight=0.5,
                     diversity_weight=1.0, robustness_weight=0.4,
                     algorithm="DiverceCF", desired_class="opposite"):
        train_dataset, test_dataset, _, _ = self.split_data(dataset, target_name)

        x_train = train_dataset.drop(target_name, axis=1)
        x_test = test_dataset.drop(columns=[target_name])

        numerical = continuous_features
        categorical = dataset.columns.difference(list(numerical))

        cat_features = {}
        for col in categorical:
            if col in dataset.columns:
                cat_features[col] = dataset[col].unique().tolist()

        if model_backend == "sklearn":
            exp_method = 'genetic'
            m = dice_ml_x.Model(model=model, backend=model_backend)
        else:
            exp_method = 'gradient'
            m = dice_ml_x.Model(model=model, backend=model_backend, func='ohe-min-max')
        
        d = dice_ml_x.Data(dataframe=train_dataset, continuous_features=list(numerical), outcome_name=target_name)
        
        exp = dice_ml_x.DiceX(d, m, method=exp_method)
        
        kwargs = {
            'gaussian': {
                'continuous_features': continuous_features,
                'categorical_features': cat_features,
                'std_dev': 0.3
            },
            'random': {
                'continuous_features': continuous_features,
                'categorical_features': cat_features,
                'feature_ranges': exp.data_interface.get_features_range_float()[1]
            },
            'spherical': {
                'continuous_features': continuous_features,
                'categorical_features': cat_features,
                'feature_ranges': exp.data_interface.get_features_range_float()[1] 
            }
        } 
        start = time()
        explainer_options = OrderedDict(query_instances=x_test[1:2], total_CFS=total_CFS,
                                        perturbation_method='gaussian', desired_class=desired_class, proximity_weight=proximity_weight,
                                        diversity_weight=diversity_weight, robustness_weight=robustness_weight,
                                        algorithm=algorithm, **kwargs['gaussian'])
        dice_exp = exp.generate_counterfactuals(explainer_options)
        end = time()
        generation_time = (end-start)
        CFs_df = dice_exp.to_dataframe()
        metrics_dict = self.compute_metrics(d, CFs_df, model, explainer_options['query_instances'],
                                            target_name, exp, explainer_options)
        return CFs_df, x_test[1:2], generation_time, exp.loss_history, metrics_dict
    
    def compute_metrics(self, data_class: dice_ml_x.Data, C: pd.DataFrame,
                        model: any, original_instance: pd.DataFrame,
                        target_name: str, explainer: dice_ml_x.DiceX, explainer_options: OrderedDict) -> dict:
        x_ohe_tensor = torch.tensor(data_class.get_ohe_min_max_normalized_data(original_instance).values, dtype=torch.float32)
        x_ohe_tensor_targetless = torch.tensor(data_class.get_ohe_min_max_normalized_data(original_instance.drop(columns=[target_name], inplace=True)).values, dtype=torch.float32)
        C_ohe = data_class.get_ohe_min_max_normalized_data(C)
        C_ohe_tensor = torch.tensor(C_ohe.values, dtype=torch.float32)
        proximity = self.compute_proximity(x_ohe_tensor, C_ohe_tensor)
        diversity = self.compute_diversity(C_ohe_tensor)
        x_ohe_prime_tensor = self.generate_perturbations(x_ohe_tensor_targetless, data_class, model)
        explainer_options['query_instances'] = x_ohe_prime_tensor
        C_prime_ohe_tensor = explainer.generate_counterfactuals(explainer_options)
        robustness = self.compute_stability(C_ohe_tensor, C_prime_ohe_tensor)
        return {
            'proximity': proximity,
            'diversity': diversity,
            'robustness': robustness
        }