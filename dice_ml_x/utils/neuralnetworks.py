from torch import nn, sigmoid
import tensorflow as tf
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import \
(StandardScaler,
 OneHotEncoder,
 LabelEncoder)
import numpy as np
from collections import OrderedDict
import os
import time
import copy
class TF2Model(tf.keras.Model):

    """
    The model class is built accordingly with the loaded model summary in the repository.
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    dense (Dense)               (None, 20)                600       
                                                                    
    dense_1 (Dense)             (None, 1)                 21        
                                                                    
    =================================================================
    Total params: 621 (2.43 KB)
    Trainable params: 621 (2.43 KB)
    Non-trainable params: 0 (0.00 Byte)
    """
    def __init__(self):
        super(TF2Model, self).__init__()
        self.dense_1: tf.keras.layers.Dense = tf.keras.layers.Dense(20, activation=tf.keras.activations.relu, name='dense_1')
        self.dense_2: tf.keras.layers.Dense = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name='dense_2')

    def call(self, input):
        x = self.dense_1(input)
        output = self.dense_2(x)
        return output
    

class PYTDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, target_column: str, scaler=None, encoder=None, target_encoder=None):
        self.dataframe = dataframe.copy()
        self.target_column = target_column
        self.scaler = scaler
        self.encoder = encoder
        self.target_encoder = target_encoder

        self.features = self.dataframe.drop(columns=[target_column])
        self.target = self.dataframe[target_column]
        
        self.numerical_cols = self.features.select_dtypes(include=[np.number]).columns
        self.categorical_cols = self.features.columns.difference(self.numerical_cols)
        if self.scaler is None and len(self.numerical_cols) > 0:
            self.scaler = StandardScaler().fit(self.features[self.numerical_cols])

        if self.encoder is None and len(self.categorical_cols) > 0:
            self.encoder = OneHotEncoder(sparse_output=False,
                                         handle_unknown='ignore').fit(self.features[self.categorical_cols])

        if len(self.numerical_cols) > 0:
            self.features[self.numerical_cols] = self.scaler.transform(self.features[self.numerical_cols])
        
        if len(self.categorical_cols) > 0:
            encoded_cats = self.encoder.transform(self.features[self.categorical_cols])
            encoded_cat_df = pd.DataFrame(encoded_cats, columns=self.encoder.get_feature_names_out())
            encoded_cat_df.index = self.features.index
            self.features = self.features.drop(columns=list(self.categorical_cols))
            self.features = pd.concat([self.features, encoded_cat_df], axis=1)

        if self.target.dtype == 'object' or str(self.target.dtype) == 'category':
            self.target_encoder = LabelEncoder()
            self.target = self.target_encoder.fit_transform(self.target)
        else:
            self.target_encoder = None

        self.features = torch.tensor(self.features.values, dtype=torch.float32)
        self.target = torch.tensor(self.target.values, dtype=torch.long)

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, position):
        return self.features[position], self.target[position]


class PYTModel(nn.Module):
    def __init__(self, in_features):
        super(PYTModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )
        self.history = {
            'epoch': [],
            'train_acc': [],
            'train_loss': [],
            'test_acc': [],
            'test_loss': [],
            'model_state_dict': [],
            'optimizer_state_dict': []
        }
        self.best_val_acc = 0.0

        

    def forward(self, input):
        output = self.model(input)
        return output
    
    def train(self, epochs=10, criterion=nn.BCELoss(), optimizer="adam",
              learning_rate=1e-3,
              train_dataloader: torch.utils.data.DataLoader=None,
              test_dataloader: torch.utils.data.DataLoader=None,
              device='gpu' if torch.cuda.is_available() else 'cpu', save=False):
        train_steps = len(train_dataloader)
        test_steps = len(test_dataloader)
        self.model.zero_grad()
        self.model.to(device=device)
        if optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        
        for epoch in range(epochs):
            train_loss = 0.0
            test_loss = 0.0
            correct_train_preds = 0.0
            correct_test_preds = 0.0

            self.model.train()
            for batch_idx, batch in enumerate(train_dataloader):
                features, labels = batch
                features = features.to(device)
                labels = labels.float().to(device).unsqueeze(1)
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct_train_preds += (preds == labels).sum().item()
                

            epoch_train_loss = train_loss / train_steps
            epoch_train_acc = correct_train_preds / len(train_dataloader.dataset)
            self.history['epoch'].append(epoch)
            self.history['train_acc'].append(epoch_train_acc)
            self.history['train_loss'].append(epoch_train_loss)

            
            self.model.eval()
            
            for _, test_batch in enumerate(test_dataloader):
                test_features, test_labels = test_batch
                test_features, test_labels = test_features.to(device), test_labels.float().to(device).unsqueeze(1)
                with torch.no_grad():
                    
                    test_outputs = self.model(test_features)
                    loss = criterion(test_outputs, test_labels)
                    test_preds = (test_outputs > 0.5).float()
                test_loss += loss.item()
                correct_test_preds += (test_preds == test_labels).sum().item()


            epoch_test_loss = test_loss / test_steps
            epoch_test_acc = correct_test_preds / len(test_dataloader.dataset)

            self.history['test_acc'].append(epoch_test_acc)
            self.history['test_loss'].append(epoch_test_loss)
            self.history['optimizer_state_dict'].append(copy.deepcopy(optimizer.state_dict()))
            self.history['model_state_dict'].append(copy.deepcopy(self.model.state_dict()))
                
        if save:
            self.save_model(self.history)
    

    def save_model(self, history, model_path=None):
        artefacts_dir = 'artefacts'
        model_file_name = f'pyt_model_{time.time()}.pth'
        if model_path is not None and os.path.isdir(model_path):
            torch.save(history, os.path.join(model_path, model_file_name))
        else:
            if not os.path.isdir(artefacts_dir):
                os.mkdir(artefacts_dir)
            torch.save(history, os.path.join(artefacts_dir, model_file_name))

class FFNetwork(nn.Module):
    def __init__(self, input_size, is_classifier=True):
        super(FFNetwork, self).__init__()
        self.is_classifier = is_classifier
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear_relu_stack(x)
        out = sigmoid(out)
        if not self.is_classifier:
            out = 3 * out  # output between 0 and 3
        return out


class MulticlassNetwork(nn.Module):
    def __init__(self, input_size: int, num_class: int):
        super(MulticlassNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, num_class)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear_relu_stack(x)
        out = self.softmax(x)

        return out
