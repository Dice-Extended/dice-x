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

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

class Benchmarking:
    def __init__(self, datasets: list, backends: list, perturbation_methods: list, metrics=None):
        self.datasets = datasets
        self.backends = backends
        self.perturbation_methods = perturbation_methods
        self.metrics = metrics if metrics else ["fidelity", "proximity", "diversity", "robustness"]
        self.models = {}
        self.results = {}
        self.sklearn_pipeline = None

    def load_dataset(self, name):
        if name == "compas-recidivism":
            return helpers.load_compas_dataset(), 'twoyearrecid'
        elif name == "adult-income":
            return helpers.load_adult_income_dataset(), 'income'
        elif name == "lending-club":
            return helpers.load_lending_club_dataset(), 'loan_status'
        elif name == "german-credit":
            return helpers.load_german_credit_dataset(), 'credit_risk'
        else:
            raise ValueError(f"Unkown dataset: {name}")
        
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
    
    def load_and_train(self, batch_size, artefact_path=None):
        num_processes = len(self.datasets) * len(self.backends)
        
        if artefact_path is None:
            artefact_path = "benchmarking_artefact"
        if not os.path.isdir(artefact_path):
            os.mkdir(artefact_path)
        with tqdm(total=num_processes, desc="Benchmarking", leave=True) as d_pbar:
            for dataset_name in self.datasets:
                
                df, target_column = self.load_dataset(dataset_name)
                
                continuous_features = df.select_dtypes(include=[np.number]).columns.to_list()
                continuous_features.remove(target_column)
                self.results[dataset_name] = {}
                model_items = {}
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
                        'time': {}
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

                    
                    for method in self.perturbation_methods:
                        print(f"the dataset is : {dataset_name}, the backend is : {backend}, the method is {method}")
                        
                        cfs, input_instance, generation_time = self.generate_cfs(df,
                                                                                 continuous_features,
                                                                                 model,
                                                                                 backend, method,
                                                                                 target_column)
                        backend_results['cfs'][method] = cfs
                        backend_results['input_instance'][method] = input_instance
                        backend_results['time'][method] = generation_time
                        self.results[dataset_name][backend] = backend_results
                    d_pbar.set_postfix(OrderedDict(
                            dataset=dataset_name,
                            backend=backend,
                            method=method
                    ))
                    d_pbar.update(1)

    
    def generate_cfs(self, dataset: pd.DataFrame,
                     continuous_features: list,
                     model: any, model_backend: str,
                     perturbation_method: str,
                     target_name: str):
        train_dataset, test_dataset, _, _ = self.split_data(dataset, target_name)

        x_train = train_dataset.drop(target_name, axis=1)
        x_test = test_dataset.drop(columns=[target_name])

        numerical = continuous_features
        categorical = dataset.columns.difference(list(numerical))

        cat_features = {}
        for col in categorical:
            if col in dataset.columns:
                cat_features[col] = dataset[col].unique().tolist()

        if target_name in ['credit_risk', 'loan_status']:
            d_frame = dataset
        else:
            d_frame = train_dataset
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
        dice_exp = exp.generate_counterfactuals(x_test[1:2], total_CFs=4, perturbation_method=perturbation_method,
                                        desired_class="opposite", **kwargs[perturbation_method])
        end = time()
        generation_time = (end-start)
        return dice_exp.to_dataframe(), x_test[1:2], generation_time
        
        d = dice_ml_x.Data(dataframe=train_dataset, continuous_features=continuous_features, outcome_name=target_name)
        m = dice_ml_x.Model(model=model, backend=model_backend)
        categorical = x_train.columns.difference(continuous_features)

        cat_features = {}
        for col in categorical:
            if col in dataset.columns:
                cat_features[col] = dataset[col].unique().tolist()

        if model_backend == "sklearn":
            exp_method = 'genetic'
        else:
            exp_method = 'gradient'

        exp = dice_ml_x.DiceX(d, m, method=exp_method)
        cfes = exp.generate_counterfactuals(x_test[1:2], total_CFs=5, desired_class="opposite",
                                        perturbation_method=perturbation_method, **kwargs[perturbation_method])
        
        return cfes.to_dataframe()
    
    def compute_metrics():
        return
        

