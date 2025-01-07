from base_autoencoder import _BaseAutoEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.adam import Adam
import numpy as np
from tqdm import tqdm


class VariationalAutoEncoder(_BaseAutoEncoder):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.encoder_fc = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparam(self, mu: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        s_dev = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(s_dev)
        return mu + s_dev * epsilon
    
    def forward(self, x: torch.Tensor) -> tuple:
        hidden_activation = torch.relu(self.encoder_fc(x))
        mu = self.fc_mu(hidden_activation)
        log_variance = self.fc_log_var(hidden_activation)
        z = self.reparam(mu, log_variance)
        decoded = self.decoder(z)
        return decoded, mu, log_variance
    
    def _compute_vae_loss(self, x: torch.Tensor, decoded_x: torch.Tensor, mu: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        reconstruction_loss = nn.functional.binary_cross_entropy(decoded_x, x)
        kl_loss = -0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())
        return reconstruction_loss + kl_loss
    
    def train_autoencoder(self, X, epochs = 10, batch_size = 16, learning_rate = 0.001, verbose = True, save_model=True, save_interval=5):
        optimizer = Adam(self.parameters(), lr=learning_rate)

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            
            for batch in progress_bar:
                x_batch = batch[0]
                optimizer.zero_grad()

                x_decoded, mu, log_variance = self.forward(x_batch)
                loss = self._compute_vae_loss(x_batch, x_decoded, mu, log_variance)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            average_loss = total_loss / len(dataloader)
            self._history['loss'].append(average_loss)

            if save_model and epoch % save_interval == 0:
                self.save_model(f"autoencoder_model_{epoch}")

            if verbose:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}")
        print("Training finished!")



        